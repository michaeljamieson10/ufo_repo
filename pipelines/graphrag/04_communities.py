"""GraphRAG stage 04 — community detection + community summaries.

Why communities matter
----------------------
Microsoft's GraphRAG paper made one observation that's the main reason
to build a graph at all: vector retrieval is great for *local* questions
("what does file X say about Y?") and useless for *global* questions
("what are the major themes in this corpus?").

Their fix: cluster the entity graph with the **Leiden algorithm**, then
have an LLM summarize each cluster. At query time, a global question
hits the *community summaries* instead of raw chunks. You answer 4000
pages with 50 paragraphs.

This stage:
  1. Builds an in-memory entity co-occurrence graph from
     `chunk_entities.jsonl` — nodes are entities, edges are
     "appeared together in chunk X" (weighted by how many chunks).
  2. Runs the Leiden algorithm (via `graspologic`) to find communities.
  3. For each community, asks Claude CLI to write a 2–3 sentence
     summary covering its theme, key entities, and rough time period.
  4. Embeds each summary with bge-small for routing at query time.

Output
------
`data/communities.jsonl` — one row per community:
    { "id": int, "size": int, "members": [...entity ids...],
      "summary": "...", "summary_embedding": [...384...] }
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
ENTITIES_IN = DATA / "chunk_entities.jsonl"
OUT_PATH = DATA / "communities.jsonl"

sys.path.insert(0, str(PIPELINES))


def normalize_entity_id(t: str, name: str) -> str:
    """Stable id used to merge entity mentions across chunks. We
    normalize aggressively (lowercase, strip punctuation) because
    entity-resolution is otherwise the silent killer of graph quality."""
    n = name.strip().lower()
    n = "".join(c if c.isalnum() else "_" for c in n)
    return f"{t}::{n}"


def build_graph() -> tuple[dict, dict, list[set[str]]]:
    """Returns (entity_meta_by_id, edge_weight_map, chunks_by_id)."""
    entity_meta: dict[str, dict] = {}
    chunk_members: list[tuple[str, set[str]]] = []
    edge_w: dict[tuple[str, str], int] = defaultdict(int)

    if not ENTITIES_IN.exists():
        sys.exit(f"missing {ENTITIES_IN}; run 03_entity_extract.py first")

    for line in ENTITIES_IN.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        ext = row.get("extraction") or {}
        ents = ext.get("entities") or []
        members: set[str] = set()
        for e in ents:
            eid = normalize_entity_id(e["type"], e["name"])
            entity_meta.setdefault(eid, {"type": e["type"], "name": e["name"], "count": 0,
                                         "aliases": set(), "attrs": {}})
            entity_meta[eid]["count"] += 1
            for a in e.get("aliases") or []:
                entity_meta[eid]["aliases"].add(a)
            entity_meta[eid]["attrs"].update(e.get("attrs") or {})
            members.add(eid)
        chunk_members.append((row["chunk_id"], members))
        members_list = sorted(members)
        for i, a in enumerate(members_list):
            for b in members_list[i + 1:]:
                edge_w[(a, b)] += 1

    for v in entity_meta.values():
        v["aliases"] = sorted(v["aliases"])
    return entity_meta, edge_w, chunk_members


def detect_communities(entity_meta: dict, edge_w: dict) -> dict[str, int]:
    """Run Leiden on the weighted entity-cooccurrence graph. Returns
    {entity_id -> community_id}."""
    try:
        import networkx as nx
        from graspologic.partition import hierarchical_leiden
    except ImportError:
        sys.exit("pip install networkx graspologic")

    g = nx.Graph()
    for nid, meta in entity_meta.items():
        g.add_node(nid, **{k: v for k, v in meta.items() if k != "aliases"})
    for (a, b), w in edge_w.items():
        if w >= 1:
            g.add_edge(a, b, weight=w)

    if g.number_of_edges() == 0:
        return {}

    # max_cluster_size keeps communities Claude-summarizable in one
    # prompt. Hierarchical Leiden recursively splits any cluster that
    # exceeds it — much better than vanilla Leiden for skewed corpora.
    res = hierarchical_leiden(g, max_cluster_size=30)
    cmap: dict[str, int] = {}
    for r in res:
        cmap[r.node] = int(r.cluster)
    return cmap


def summarize_community(cid: int, members: list[dict], example_chunks: list[str]) -> str:
    from llm import ClaudeCLIChatModel
    from langchain_core.messages import HumanMessage, SystemMessage

    model = ClaudeCLIChatModel(model="sonnet", timeout_seconds=120)
    member_lines = [f"- {m['type']}: {m['name']}" for m in members[:30]]
    chunks_text = "\n---\n".join(example_chunks[:5])
    prompt = (
        "Summarize a cluster of entities from a declassified UAP/UFO corpus.\n\n"
        "ENTITIES IN THE CLUSTER:\n" + "\n".join(member_lines) + "\n\n"
        "EXAMPLE CHUNKS WHERE THESE ENTITIES CO-OCCUR:\n" + chunks_text + "\n\n"
        "Write 2–4 sentences covering: the cluster's theme, the key entities, "
        "the rough time period (if discernible), and any notable patterns. "
        "Be factual and specific. No prose preamble."
    )
    msg = model.invoke([SystemMessage(content="You write concise factual cluster summaries."),
                        HumanMessage(content=prompt)])
    return str(msg.content).strip()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[communities] building entity graph...", file=sys.stderr)
    entity_meta, edge_w, chunk_members = build_graph()
    print(f"  {len(entity_meta)} entities, {len(edge_w)} edges", file=sys.stderr)

    cmap = detect_communities(entity_meta, edge_w)
    if not cmap:
        sys.exit("no communities found")

    by_community: dict[int, list[str]] = defaultdict(list)
    for eid, cid in cmap.items():
        by_community[cid].append(eid)
    print(f"  {len(by_community)} communities", file=sys.stderr)

    # Map chunks -> community by majority membership for finding
    # representative chunks to feed the summarizer.
    chunks_for_community: dict[int, list[str]] = defaultdict(list)
    pdf_text_lookup = {}
    pdf_jsonl = PIPELINES / "data" / "extracted_ocr.jsonl"
    if not pdf_jsonl.exists():
        pdf_jsonl = PIPELINES / "data" / "extracted.jsonl"
    if pdf_jsonl.exists():
        for line in pdf_jsonl.read_text().splitlines():
            if line:
                rec = json.loads(line)
                for p in rec.get("pages", []):
                    pdf_text_lookup[(rec["file"], p["page"])] = (p.get("text") or "")[:1500]
    for chunk_id, members in chunk_members:
        if not members:
            continue
        votes: dict[int, int] = defaultdict(int)
        for m in members:
            if m in cmap:
                votes[cmap[m]] += 1
        if not votes:
            continue
        winner = max(votes.items(), key=lambda kv: kv[1])[0]
        # Pull source text from PDF if it's a PDF chunk.
        try:
            file, page, _ = chunk_id.split(":", 2)
            text = pdf_text_lookup.get((file, int(page)), "")
        except Exception:
            text = ""
        if text:
            chunks_for_community[winner].append(text)

    from langchain_huggingface import HuggingFaceEmbeddings
    text_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    with OUT_PATH.open("w") as out:
        for cid, members in sorted(by_community.items()):
            members_meta = [entity_meta[m] | {"id": m} for m in members]
            example_chunks = chunks_for_community.get(cid, [])
            print(f"  [community {cid}] {len(members)} members", file=sys.stderr)
            summary = summarize_community(cid, members_meta, example_chunks)
            embedding = text_emb.embed_query(summary)
            out.write(json.dumps({
                "id": cid,
                "size": len(members),
                "members": members,
                "summary": summary,
                "summary_embedding": embedding,
            }) + "\n")
    print(f"Output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
