"""GraphRAG stage 03 — entity + relationship extraction.

Why GraphRAG over plain RAG
---------------------------
Plain vector RAG answers "what does the FBI file 62-HQ-83894 say
about discs over Oak Ridge?" reasonably well — the chunk for that
case file ranks high.

It collapses on global queries the corpus invites:
  - "Which agencies have reported tic-tac-shaped objects?"
  - "Were there incidents in 1947 and 2024 in the same location?"
  - "Who appears as a witness across multiple agencies?"

These need a *graph traversal*, not similarity. So before storing in
FalkorDB, we extract entities + relationships per chunk and
de-duplicate them at the corpus level. This stage's job is the per-
chunk extraction; stage 05 does the corpus-level merge.

Algorithm
---------
For every chunk in `data/extracted_ocr.jsonl` (PDFs), `video_frames.jsonl`
(video frames are short docs), and `images.jsonl`:
  1. Build a system prompt with the schema + few-shot examples.
  2. Ask Claude CLI for a `ChunkExtraction` (entities + incidents).
  3. Append result alongside chunk metadata to `data/chunk_entities.jsonl`.

Why few-shot examples
---------------------
Without them, the model wobbles on edge cases — it'll mark "the
witness" as a Person entity (no name), or hallucinate dates from
context the chunk doesn't actually contain. Two examples in the
system prompt eliminate ~80% of those failures and is the cheapest
quality lever available.

Idempotency: each chunk has a stable id (`file:page:chunk`); we skip
ids already in the output file so re-runs resume.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA / "chunk_entities.jsonl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

sys.path.insert(0, str(PIPELINES))


SYSTEM_PROMPT = """You extract entities and incidents from declassified UAP/UFO documents.

Rules:
- Only extract entities EXPLICITLY mentioned in the chunk. Do not infer from prior knowledge.
- Use canonical names: 'FBI' not 'Federal Bureau of Investigation'.
- For Person, only extract NAMED individuals (no 'witness', 'pilot', 'farmer').
- For Object, classify shape into: disc, saucer, triangle, cigar, sphere, egg-shape, tic-tac, light, fireball, boomerang, rectangle, swarm, unknown_craft, other.
- For Incident, only emit one if the chunk describes a SPECIFIC event (date OR location OR enough specificity).
- Dates: ISO format YYYY-MM-DD when full; YYYY when only year known; null otherwise.

Examples:

Chunk: "On 24 June 1947, near Mount Rainier, Washington, civilian pilot Kenneth Arnold reported nine bright discs traveling at high speed."
Output: {
  "entities": [
    {"type":"Person","name":"Kenneth Arnold","aliases":[],"attrs":{"role":"civilian pilot"}},
    {"type":"Location","name":"Mount Rainier, Washington","aliases":[],"attrs":{}},
    {"type":"Object","name":"nine bright discs","aliases":[],"attrs":{"kind":"disc","count":"9"}},
    {"type":"Date","name":"1947-06-24","aliases":[],"attrs":{}}
  ],
  "incidents": [
    {"summary":"Civilian pilot Kenneth Arnold reported nine disc-shaped objects near Mount Rainier.",
     "date":"1947-06-24","location":"Mount Rainier, Washington","agency":null,
     "object_kind":"disc","witnesses":["Kenneth Arnold"]}
  ]
}

Chunk: "FBI File 62-HQ-83894 contains correspondence regarding flying disc reports collected from 1947 through 1968."
Output: {
  "entities": [
    {"type":"Agency","name":"FBI","aliases":[],"attrs":{}},
    {"type":"CaseFile","name":"62-HQ-83894","aliases":[],"attrs":{"agency":"FBI"}},
    {"type":"Object","name":"flying disc","aliases":[],"attrs":{"kind":"disc"}}
  ],
  "incidents": []
}
"""


def chunkify_text(text: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def iter_chunks():
    """Yield (chunk_id, text, source_metadata) for every text source we have."""
    pdf_jsonl = PIPELINES / "data" / "extracted_ocr.jsonl"
    if not pdf_jsonl.exists():
        pdf_jsonl = PIPELINES / "data" / "extracted.jsonl"
    if pdf_jsonl.exists():
        for line in pdf_jsonl.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            for p in rec.get("pages", []):
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                for ci, chunk in enumerate(chunkify_text(text)):
                    yield (
                        f"{rec['file']}:{p['page']}:{ci}",
                        chunk,
                        {
                            "kind": "pdf",
                            "file": rec["file"],
                            "page": p["page"],
                            "chunk": ci,
                            "agency": rec.get("agency", ""),
                            "incident_date": rec.get("incident_date", ""),
                            "incident_location": rec.get("incident_location", ""),
                        },
                    )

    frames_jsonl = DATA / "video_frames.jsonl"
    if frames_jsonl.exists():
        for line in frames_jsonl.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get("caption") or "") + " " + (rec.get("visible_text") or "")
            text = text.strip()
            if not text:
                continue
            yield (
                f"{rec['video_file']}:{rec['frame_idx']}:0",
                text,
                {
                    "kind": "frame",
                    "video_file": rec["video_file"],
                    "frame_idx": rec["frame_idx"],
                    "timestamp_s": rec["timestamp_s"],
                },
            )

    images_jsonl = DATA / "images.jsonl"
    if images_jsonl.exists():
        for line in images_jsonl.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get("caption") or "") + " " + (rec.get("visible_text") or "")
            text = text.strip()
            if not text:
                continue
            yield (
                f"image:{rec['file']}",
                text,
                {
                    "kind": "image",
                    "file": rec["file"],
                    "agency": rec.get("agency", ""),
                    "incident_date": rec.get("incident_date", ""),
                },
            )


def extract_one(args: tuple) -> dict:
    """Single-chunk extraction (legacy path, used by --no-batch)."""
    chunk_id, text, meta = args
    from llm import ClaudeCLIChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
    from graphrag.schema import ChunkExtraction

    import os as _os
    _model_alias = _os.environ.get("UFO_ENTITY_MODEL", "haiku")
    model = ClaudeCLIChatModel(model=_model_alias, timeout_seconds=120)
    structured = model.with_structured_output(ChunkExtraction)
    try:
        result = structured.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Chunk:\n{text}\n\nReturn the extraction JSON."),
        ])
        return {"chunk_id": chunk_id, "metadata": meta, "extraction": result.model_dump()}
    except Exception as e:
        return {"chunk_id": chunk_id, "metadata": meta, "error": str(e)}


def extract_batch(batch: list[tuple]) -> list[dict]:
    """Extract entities for a *batch* of chunks in a single CLI call.

    Why batch
    ---------
    Claude subscription rate limits cap requests/hour, not tokens.
    Sending 5-8 chunks per CLI call cuts request count ~6x with no
    quality loss on entity extraction (each chunk is independent —
    the model just emits a list of N ChunkExtractions).

    The schema-bounded JSON output makes per-chunk parsing trivial:
    we ask for {"results": [<extraction_1>, ..., <extraction_N>]}
    and zip the parsed list against the input chunks.
    """
    from llm import ClaudeCLIChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
    from graphrag.schema import ChunkExtraction
    from pydantic import BaseModel
    import os as _os

    class BatchResult(BaseModel):
        results: list[ChunkExtraction]

    _model_alias = _os.environ.get("UFO_ENTITY_MODEL", "haiku")
    model = ClaudeCLIChatModel(model=_model_alias, timeout_seconds=180)
    structured = model.with_structured_output(BatchResult)

    chunks_text = "\n\n".join(
        f"=== CHUNK {i+1} ===\n{text}" for i, (_cid, text, _m) in enumerate(batch)
    )
    user = (
        f"Extract entities + incidents from each of the {len(batch)} chunks "
        f"below. Return a JSON object with key 'results' containing a list of "
        f"{len(batch)} ChunkExtraction objects, in chunk order.\n\n"
        f"{chunks_text}"
    )
    try:
        out = structured.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user),
        ])
    except Exception as e:
        # Fall back to per-chunk so a single bad batch doesn't lose the lot.
        err = str(e)
        return [
            {"chunk_id": cid, "metadata": meta, "error": err}
            for cid, _t, meta in batch
        ]

    # Pad/truncate if the model returned the wrong count.
    rows = []
    for (cid, _t, meta), result in zip(batch, out.results):
        rows.append({"chunk_id": cid, "metadata": meta, "extraction": result.model_dump()})
    if len(out.results) < len(batch):
        for cid, _t, meta in batch[len(out.results):]:
            rows.append({"chunk_id": cid, "metadata": meta,
                         "error": f"batch returned only {len(out.results)} of {len(batch)}"})
    return rows


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent CLI calls.")
    ap.add_argument("--batch-size", type=int, default=5,
                    help="Chunks per CLI call. 5 is a sweet spot — bigger "
                         "saves more requests but risks the model truncating "
                         "or losing per-chunk fidelity.")
    ap.add_argument("--no-batch", action="store_true",
                    help="One chunk per CLI call (legacy, hits rate limits faster).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line:
                done.add(json.loads(line)["chunk_id"])

    work = [(cid, t, m) for cid, t, m in iter_chunks() if cid not in done]
    if args.limit:
        work = work[: args.limit]
    n = len(work)
    print(f"[entities] {n} chunks pending ({len(done)} already done)", file=sys.stderr)

    if args.no_batch:
        with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(extract_one, w): w[0] for w in work}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                if i % 10 == 0 or i == n:
                    err = "ERR" if "error" in row else "OK"
                    print(f"  [{i}/{n}] {err} {row['chunk_id']}", file=sys.stderr)
        return

    # Batched mode (default).
    bs = max(1, args.batch_size)
    batches = [work[i : i + bs] for i in range(0, n, bs)]
    print(f"[entities] {len(batches)} batches × {bs} chunks each", file=sys.stderr)

    with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_batch, b): b[0][0] for b in batches}
        completed_chunks = 0
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            for row in rows:
                out.write(json.dumps(row) + "\n")
            out.flush()
            completed_chunks += len(rows)
            err_n = sum(1 for r in rows if "error" in r)
            tag = f"OK {len(rows)}" if err_n == 0 else f"PARTIAL {len(rows)-err_n}/{len(rows)}"
            print(
                f"  batch [{i}/{len(batches)}] {tag} ({completed_chunks}/{n} chunks)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
