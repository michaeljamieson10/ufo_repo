"""Stage 03 — chunk → embed → persist to Chroma.

Why this stage exists
---------------------
Vector retrieval works on chunks, not whole PDFs. Two design knobs
matter for interview-grade RAG:

1) Chunk size + overlap
   Bigger chunks = more context per hit but blurrier semantic match.
   Smaller chunks = sharper match, more "needle-in-haystack" risk.
   Standard starting point: 1000 chars / 150 overlap with
   ``RecursiveCharacterTextSplitter``, which respects paragraph then
   sentence boundaries before falling back to char count.

2) Metadata
   Every chunk carries the file basename, page number, agency,
   incident date, and incident location. These let stage 04 filter
   ("only FBI files from 1947") *before* the dense search runs — much
   cheaper than post-filtering and matches what production RAG looks
   like.

Embedding model
---------------
``BAAI/bge-small-en-v1.5`` (384-dim, ~120MB) — fast on CPU, good
quality, and 100% local so we don't need any API key. The Codex/Claude
CLIs don't expose embedding endpoints, so a local HF model is the
right pairing for this corpus.

Vector store
------------
Chroma in persistent mode at ``store/chroma``. Re-runs are idempotent:
we tag each chunk with a deterministic id (``file:page:chunk_idx``) and
upsert.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
STORE = Path(__file__).resolve().parent / "store" / "chroma"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "uap_release_01"


def load_records() -> list[dict]:
    # Prefer the OCR'd file if stage 02 ran; fall back to raw extract.
    for name in ("extracted_ocr.jsonl", "extracted.jsonl"):
        p = DATA / name
        if p.exists():
            print(f"Reading {p.name}", file=sys.stderr)
            return [json.loads(l) for l in p.read_text().splitlines() if l]
    sys.exit("no extracted JSONL — run 01_ingest.py first")


def to_documents(records: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Order matters: paragraph → newline → sentence → space → char.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    for rec in records:
        base_meta = {
            "file": rec["file"],
            "title": (rec.get("title") or "")[:300],
            "agency": rec.get("agency") or "",
            "incident_date": rec.get("incident_date") or "",
            "incident_location": rec.get("incident_location") or "",
            "release_date": rec.get("release_date") or "",
        }
        for p in rec["pages"]:
            text = (p.get("text") or "").strip()
            if not text:
                continue
            for ci, chunk in enumerate(splitter.split_text(text)):
                meta = dict(base_meta)
                meta["page"] = p["page"]
                meta["chunk"] = ci
                meta["was_ocr"] = bool(p.get("ocr_text"))
                docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def main() -> None:
    records = load_records()
    print(f"Records: {len(records)}", file=sys.stderr)

    t0 = time.time()
    docs = to_documents(records)
    print(f"Chunks: {len(docs)} (in {time.time()-t0:.1f}s)", file=sys.stderr)

    print(f"Embedding with {EMBED_MODEL} (first run downloads ~120MB)...",
          file=sys.stderr)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine ≈ dot product
    )

    STORE.mkdir(parents=True, exist_ok=True)
    db = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(STORE),
    )

    # Stable IDs so re-runs upsert instead of duplicating.
    ids = [
        f"{d.metadata['file']}:{d.metadata['page']}:{d.metadata['chunk']}"
        for d in docs
    ]

    BATCH = 256
    t0 = time.time()
    for i in range(0, len(docs), BATCH):
        db.add_documents(docs[i : i + BATCH], ids=ids[i : i + BATCH])
        print(
            f"  embedded {min(i + BATCH, len(docs))}/{len(docs)} "
            f"({(time.time()-t0):.1f}s elapsed)",
            file=sys.stderr,
        )

    print(f"Persisted to {STORE}", file=sys.stderr)


if __name__ == "__main__":
    main()
