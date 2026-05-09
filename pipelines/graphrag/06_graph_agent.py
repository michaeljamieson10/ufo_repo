"""GraphRAG stage 06 — multi-strategy retrieval over FalkorDB.

The "talk to the database" experience
-------------------------------------
Three retrieval strategies, picked per-query by an LLM router:

  1. CYPHER (text-to-Cypher) — for STRUCTURED queries:
       "incidents in 1947 involving the Air Force"
       "all witnesses who appear in more than one case file"
     Claude writes a Cypher query against our schema; we execute it
     against FalkorDB; results become the answer context.

  2. VECTOR — for SEMANTIC queries:
       "ramjet propulsion" → text similarity over Chunk.text_embedding
       "blurry triangular craft over water" → CLIP similarity over
        Frame.clip_embedding (visual)

  3. COMMUNITY — for GLOBAL queries ("themes", "patterns"):
       "what kinds of UFOs were most reported in the 1950s?"
     Embed the query, find top-K Communities by summary similarity,
     return their summaries as context.

Plus a HYBRID mode that runs all three and merges. The router picks
the strategy from the question; you can also force one with --mode.

LangGraph state machine
-----------------------
            START
              │
              ▼
       ┌──────────────┐
       │   classify   │  → query_type ∈ {structured, semantic, global, hybrid}
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ route by     │  ↳ cypher / vector / community / hybrid
       │ query_type   │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ retrieve_*   │  ← runs the chosen retrieval(s)
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │  synthesize  │  → Claude CLI writes answer w/ citations
       └──────┬───────┘
              ▼
             END
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

PIPELINES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINES))

from llm import build_chat_model

CYPHER_SCHEMA_BLURB = """
Schema (FalkorDB / Cypher):

Node labels:
  (:Document {file, kind, agency, title})       — kind in {pdf, video, image}
  (:Chunk {id, text, page, text_embedding})     — retrievable text span
  (:Frame {id, timestamp_s, caption, object_kind, clip_embedding, text_embedding})
  (:Agency {name})                              — FBI, DOW, NASA, NARA, DOS, etc.
  (:Person {name})
  (:Location {name})
  (:Object {name})                              — a UAP-like object mention
  (:CaseFile {name})                            — e.g. '62-HQ-83894'
  (:Date {name})                                — ISO YYYY-MM-DD or YYYY
  (:Incident {summary, date, location, agency, object_kind})
  (:Community {id, summary, size})

Relationships:
  (:Document)-[:HAS_CHUNK]->(:Chunk)
  (:Document)-[:FROM_AGENCY]->(:Agency)
  (:Frame)-[:IN_VIDEO]->(:Document {kind:'video'})
  (:Chunk)-[:MENTIONS]->(:Person|:Location|:Object|:Date|:Agency|:CaseFile)
  (:Incident)-[:REPORTED_IN]->(:Chunk)
  (:Incident)-[:WITNESSED_BY]->(:Person)
  (:Community)-[:CONTAINS]->(any entity)
"""


def get_db():
    from falkordb import FalkorDB
    return FalkorDB(host="localhost", port=6379).select_graph("uap")


# -- Router schema ------------------------------------------------------------


class QueryPlan(BaseModel):
    """How the agent should answer the question."""
    mode: str = Field(description="One of: structured, semantic, global, hybrid")
    rationale: str = Field(description="One sentence explaining the choice.")


# -- Nodes --------------------------------------------------------------------


class State(TypedDict, total=False):
    question: str
    mode: str
    cypher: str
    cypher_rows: list[Any]
    vector_hits: list[dict]
    community_hits: list[dict]
    answer: str


def classify(state: State) -> State:
    if state.get("mode") and state["mode"] != "auto":
        return {"mode": state["mode"]}
    llm = build_chat_model(timeout_seconds=60)
    sys_p = (
        "You route a UAP-corpus question to one retrieval strategy:\n"
        "- 'structured' if the question filters by entity/date/agency "
        "(e.g. '1947 FBI cases', 'incidents witnessed by named pilots').\n"
        "- 'semantic' if it's about content/meaning ('ramjet propulsion').\n"
        "- 'global' if it asks for themes/patterns/comparisons across the corpus.\n"
        "- 'hybrid' if it could need more than one."
    )
    plan = llm.with_structured_output(QueryPlan).invoke(
        [SystemMessage(content=sys_p), HumanMessage(content=state["question"])]
    )
    return {"mode": plan.mode}


def text_to_cypher(state: State) -> State:
    llm = build_chat_model(timeout_seconds=90)
    sys_p = (
        "You write read-only Cypher queries against a FalkorDB graph of "
        "declassified UAP files.\n\n" + CYPHER_SCHEMA_BLURB +
        "\n\nRules:\n"
        "- Only emit the Cypher. No prose, no markdown fences, no comments.\n"
        "- Always LIMIT 50.\n"
        "- Prefer MATCH over OPTIONAL MATCH unless null branches are required.\n"
        "- Use case-insensitive comparisons via toLower() on names.\n"
        "- Return concrete fields the answer can cite (file, page, summary, name, date)."
    )
    msg = llm.invoke(
        [SystemMessage(content=sys_p), HumanMessage(content=state["question"])]
    )
    cypher = str(msg.content).strip()
    if cypher.startswith("```"):
        cypher = cypher.split("```", 2)[1]
        if cypher.startswith("cypher"):
            cypher = cypher[len("cypher"):]
        cypher = cypher.strip()
    try:
        result = get_db().query(cypher)
        rows = [list(r) for r in result.result_set[:50]]
    except Exception as e:
        return {"cypher": cypher, "cypher_rows": [], "answer": f"Cypher failed: {e}"}
    return {"cypher": cypher, "cypher_rows": rows}


def vector_search(state: State) -> State:
    """Embed query, hit chunk + frame caption indexes, return top hits."""
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    qv = emb.embed_query(state["question"])
    g = get_db()
    hits: list[dict] = []
    chunk_q = (
        "CALL db.idx.vector.queryNodes('Chunk', 'text_embedding', 8, vecf32($v)) "
        "YIELD node, score "
        "MATCH (d:Document)-[:HAS_CHUNK]->(node) "
        "RETURN d.file, node.page, node.text, score"
    )
    try:
        res = g.query(chunk_q, {"v": qv}).result_set
        for file, page, text, score in res:
            hits.append({"kind": "chunk", "file": file, "page": page,
                         "text": text, "score": score})
    except Exception as e:
        print(f"  chunk vector failed: {e}", file=sys.stderr)
    frame_q = (
        "CALL db.idx.vector.queryNodes('Frame', 'text_embedding', 6, vecf32($v)) "
        "YIELD node, score "
        "MATCH (node)-[:IN_VIDEO]->(d) "
        "RETURN d.file, node.timestamp_s, node.caption, score"
    )
    try:
        res = g.query(frame_q, {"v": qv}).result_set
        for file, ts, caption, score in res:
            hits.append({"kind": "frame", "file": file, "timestamp_s": ts,
                         "caption": caption, "score": score})
    except Exception as e:
        print(f"  frame vector failed: {e}", file=sys.stderr)
    return {"vector_hits": hits}


def community_search(state: State) -> State:
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    qv = emb.embed_query(state["question"])
    q = (
        "CALL db.idx.vector.queryNodes('Community', 'text_embedding', 5, vecf32($v)) "
        "YIELD node, score "
        "RETURN node.id, node.size, node.summary, score"
    )
    try:
        rows = get_db().query(q, {"v": qv}).result_set
    except Exception as e:
        return {"community_hits": [], "answer": f"Community search failed: {e}"}
    hits = [
        {"id": r[0], "size": r[1], "summary": r[2], "score": r[3]}
        for r in rows
    ]
    return {"community_hits": hits}


def hybrid_search(state: State) -> State:
    s1 = text_to_cypher(state)
    s2 = vector_search(state)
    s3 = community_search(state)
    return {**s1, **s2, **s3}


def route_after_classify(state: State) -> str:
    return {
        "structured": "cypher",
        "semantic": "vector",
        "global": "community",
        "hybrid": "hybrid",
    }.get(state.get("mode", "hybrid"), "hybrid")


def synthesize(state: State) -> State:
    parts: list[str] = []
    if state.get("cypher_rows"):
        parts.append(
            "STRUCTURED ROWS (from Cypher):\n"
            + "\n".join(repr(r) for r in state["cypher_rows"][:30])
        )
    if state.get("vector_hits"):
        parts.append(
            "RELEVANT PASSAGES:\n"
            + "\n".join(
                f"[{i+1}] {h.get('kind')} {h.get('file')}"
                + (f" p.{h.get('page')}" if h.get('page') else "")
                + (f" t={h.get('timestamp_s'):.1f}s" if h.get('timestamp_s') else "")
                + f": {(h.get('text') or h.get('caption') or '')[:400]}"
                for i, h in enumerate(state["vector_hits"][:10])
            )
        )
    if state.get("community_hits"):
        parts.append(
            "COMMUNITY SUMMARIES (cluster-level themes):\n"
            + "\n".join(
                f"[C{c['id']} size={c['size']}] {c['summary']}"
                for c in state["community_hits"][:5]
            )
        )
    if not parts:
        return {"answer": "No retrieval results."}

    sys_p = (
        "You answer questions about declassified UAP/UFO files using the "
        "provided context. Cite by [N] for passage hits, [Cn] for community "
        "summaries, and inline-quote any concrete facts from Cypher rows. "
        "Only use information that is in the context."
    )
    user = state["question"] + "\n\n" + "\n\n".join(parts)
    msg = build_chat_model(timeout_seconds=180).invoke(
        [SystemMessage(content=sys_p), HumanMessage(content=user)]
    )
    return {"answer": str(msg.content)}


def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("cypher", text_to_cypher)
    g.add_node("vector", vector_search)
    g.add_node("community", community_search)
    g.add_node("hybrid", hybrid_search)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify, {
        "cypher": "cypher", "vector": "vector",
        "community": "community", "hybrid": "hybrid",
    })
    for n in ("cypher", "vector", "community", "hybrid"):
        g.add_edge(n, "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("auto", "structured", "semantic", "global", "hybrid"),
                    default="auto")
    ap.add_argument("question", nargs="+")
    args = ap.parse_args(argv)

    state: State = {"question": " ".join(args.question), "mode": args.mode}
    final = build_graph().invoke(state)

    print(f"\nQuestion: {state['question']}")
    print(f"Mode:     {final.get('mode')}")
    if final.get("cypher"):
        print(f"\nCypher:\n  {final['cypher']}")
    if final.get("cypher_rows"):
        print(f"\nRows: {len(final['cypher_rows'])}")
    if final.get("vector_hits"):
        print(f"Vector hits: {len(final['vector_hits'])}")
    if final.get("community_hits"):
        print(f"Community hits: {len(final['community_hits'])}")
    print(f"\nAnswer:\n{final.get('answer','(none)')}")


if __name__ == "__main__":
    main(sys.argv[1:])
