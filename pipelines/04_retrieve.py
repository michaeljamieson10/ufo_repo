"""Stage 04 — hybrid retriever + cross-encoder reranker.

Why this stage exists
---------------------
"Plain dense vector search" is what tutorials show, but interviewers
will press on the failure modes:

  1. Pure dense misses keyword/exact matches — names, serial numbers,
     case file IDs (e.g. "62-HQ-83894"). Embeddings squash those into
     fuzzy semantics.
  2. Pure BM25 (keyword) misses paraphrase — "saucer" vs. "disc"
     vs. "anomalous craft".
  3. Either retriever returns "plausible" hits at top-3; you want the
     *most relevant* one in slot 1 for the answer model. That's a
     cross-encoder reranker.

So the production-shape retriever is **BM25 + dense fused → rerank**:

    EnsembleRetriever (BM25 + Chroma dense, 50/50 weight)
        → top 20 candidates
        → cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
        → top 5 final hits

This file exposes ``build_retriever()`` for the agent stage to import.
Run it directly to query interactively (sanity check).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

# In langchain 1.x EnsembleRetriever moved to langchain_classic; keep a
# compat fallback so this works on both 0.3.x and 1.x.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

STORE = Path(__file__).resolve().parent / "store" / "chroma"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION = "uap_release_01"

DENSE_K = 20    # candidate pool from each retriever
FINAL_K = 5     # final hits after rerank


class HybridRetriever:
    """BM25 + dense ensemble, with cross-encoder rerank on top.

    We hand-roll the rerank wrapper rather than using LangChain's
    ``ContextualCompressionRetriever`` so we can keep the original
    EnsembleRetriever score for debugging and have explicit control
    over the rerank batch.
    """

    def __init__(self, k: int = FINAL_K) -> None:
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.dense = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(STORE),
        ).as_retriever(search_kwargs={"k": DENSE_K})

        # BM25 needs the corpus in memory. Pull it from Chroma once.
        all_docs = self._load_all_docs()
        self.bm25 = BM25Retriever.from_documents(all_docs, k=DENSE_K)

        # 50/50 weighting is the standard starting point. If queries are
        # mostly named entities (case IDs), bias towards BM25 (e.g.
        # weights=[0.7, 0.3]); for natural-language Qs, towards dense.
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.dense],
            weights=[0.5, 0.5],
        )

        self.reranker = CrossEncoder(RERANK_MODEL)

    def _load_all_docs(self) -> list[Document]:
        store = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(STORE),
        )
        # Chroma's get() returns dicts of parallel lists.
        raw = store.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(raw["documents"], raw["metadatas"])
        ]
        return docs

    def invoke(self, query: str, *, filter: dict | None = None) -> list[Document]:
        # Metadata filter is applied to dense only; BM25 doesn't speak
        # filters natively. For agency/date filtering we drop BM25 hits
        # post-hoc to keep semantics consistent.
        if filter:
            self.dense.search_kwargs["filter"] = filter
        candidates = self.ensemble.invoke(query)
        if filter:
            candidates = [
                c for c in candidates
                if all(c.metadata.get(k) == v for k, v in filter.items())
            ]
        # Rerank with the cross-encoder. Pairs are (query, passage).
        pairs = [(query, c.page_content) for c in candidates]
        if not pairs:
            return []
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)
        return [d for d, _ in ranked[: self.k]]


def build_retriever(k: int = FINAL_K) -> HybridRetriever:
    return HybridRetriever(k=k)


def _format_hit(d: Document, idx: int) -> str:
    md = d.metadata
    head = (
        f"[{idx}] {md.get('file')} p.{md.get('page')} "
        f"({md.get('agency','?')}, {md.get('incident_date','?')}) "
        f"score={md.get('rerank_score',0):.2f}"
    )
    body = d.page_content[:400].replace("\n", " ")
    return f"{head}\n    {body}"


def main(argv: Iterable[str]) -> None:
    args = list(argv)
    if not args:
        sys.exit("usage: 04_retrieve.py 'your query'")
    query = " ".join(args)
    r = build_retriever()
    hits = r.invoke(query)
    print(f"\nQuery: {query}\n")
    for i, d in enumerate(hits, 1):
        print(_format_hit(d, i), "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
