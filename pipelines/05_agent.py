"""Stage 05 — LangGraph agent that answers UAP questions with citations.

Why LangGraph and not a plain LCEL chain
----------------------------------------
A naive RAG chain is one shot:  prompt → retrieve → synthesize → done.
That falls over on the queries this corpus invites:

  - "Compare the FBI's 1947 disc-sighting reports to the 2024 Navy
    encounters." → needs *two* retrievals with different filters.
  - "Was the Roswell incident mentioned?" → needs a yes/no decision
    *before* synthesizing 5 paragraphs of generic prose.
  - "What does file 65_HS1-... say about propulsion?" → file already
    named; retrieval should filter by ``file=...``.

LangGraph lets us model that as a state machine: a ``classify`` node
inspects the query and routes to one of several retrieval strategies,
then a ``synthesize`` node composes the answer with citations. It also
gives us a clean place to add follow-ups (multi-hop, tool calls) later.

The graph
---------
        START
          │
          ▼
   ┌──────────────┐
   │   classify   │  → produce {query, filter, intent}
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │   retrieve   │  → hybrid retriever w/ optional metadata filter
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  synthesize  │  → Claude CLI writes answer w/ [file p.N] citations
   └──────┬───────┘
          ▼
         END

Each node mutates the typed ``State`` dict. Easy to extend later with
``critique`` or ``follow_up`` nodes that loop back to ``retrieve``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

retrieve_mod = importlib.import_module("04_retrieve")
build_retriever = retrieve_mod.build_retriever

from llm import build_chat_model


class State(TypedDict, total=False):
    question: str
    filter: dict[str, Any]
    intent: str
    docs: list[Document]
    answer: str


class QueryPlan(BaseModel):
    """Lightweight router schema. The CLI returns JSON matching this."""

    intent: str = Field(description="One of: lookup, summarize, compare, factoid")
    file_filter: str | None = Field(
        default=None, description="Exact PDF basename if the query names one, else null"
    )
    agency_filter: str | None = Field(
        default=None,
        description="One of: FBI, DOW, NASA, NARA, DOS, or null if not specified",
    )


def _make_classify_chain():
    llm = build_chat_model(timeout_seconds=60)
    prompt = (
        "You are a query router for a search system over declassified UAP files. "
        "Read the user question and produce a QueryPlan JSON. "
        "Use file_filter only if the user names a specific PDF. "
        "Use agency_filter only if the user clearly restricts to one agency."
    )
    structured = llm.with_structured_output(QueryPlan)
    return prompt, structured


def classify(state: State) -> State:
    prompt, structured = _make_classify_chain()
    plan = structured.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["question"])]
    )
    flt: dict[str, Any] = {}
    if plan.file_filter:
        flt["file"] = plan.file_filter
    if plan.agency_filter:
        flt["agency"] = plan.agency_filter
    return {"intent": plan.intent, "filter": flt}


_RETRIEVER = None  # built once per process — embedding/cross-encoder load is heavy


def _retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = build_retriever(k=6)
    return _RETRIEVER


def retrieve(state: State) -> State:
    docs = _retriever().invoke(state["question"], filter=state.get("filter") or None)
    return {"docs": docs}


def synthesize(state: State) -> State:
    docs: list[Document] = state["docs"]
    if not docs:
        return {"answer": "No matching documents found."}

    context = "\n\n".join(
        f"[{i+1}] file={d.metadata.get('file')} page={d.metadata.get('page')} "
        f"agency={d.metadata.get('agency')} date={d.metadata.get('incident_date')}\n"
        f"{d.page_content}"
        for i, d in enumerate(docs)
    )
    prompt = (
        "You are a careful researcher answering questions from declassified "
        "UAP files. Use ONLY the provided sources. Cite each claim inline as "
        "[N] matching the source numbers. If the sources don't answer the "
        "question, say so. Keep the tone factual and precise.\n\n"
        f"SOURCES:\n{context}"
    )
    llm = build_chat_model(timeout_seconds=180)
    msg = llm.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["question"])]
    )
    return {"answer": msg.content}


def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: 05_agent.py 'your question'")
    question = " ".join(argv)
    graph = build_graph()
    final = graph.invoke({"question": question})

    print(f"\nQuestion: {question}")
    print(f"Intent:   {final.get('intent')}")
    print(f"Filter:   {final.get('filter')}")
    print(f"\nAnswer:\n{final['answer']}\n")
    print("Sources:")
    for i, d in enumerate(final["docs"], 1):
        print(
            f"  [{i}] {d.metadata.get('file')} p.{d.metadata.get('page')}  "
            f"({d.metadata.get('agency')}, {d.metadata.get('incident_date')})"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
