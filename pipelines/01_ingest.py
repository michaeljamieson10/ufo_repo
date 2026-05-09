"""Stage 01 — PDF ingestion + scan/text classifier.

Why this stage exists
---------------------
About half the PDFs in this corpus have a real text layer (modern docs,
typed cables) and the other half are scanned (handwritten field reports,
1940s carbon copies). Throwing every page at OCR would be wasteful — OCR
on 4000+ pages on a CPU takes hours and produces noisier text than the
original layer when one exists.

So the ingestion contract is:
    For every PDF, walk page-by-page with PyMuPDF (``fitz``) and try to
    extract text. If the page has > MIN_CHARS of real characters we keep
    that text. If it has fewer (a near-empty extraction is the signal of
    an image-only scan), we mark the page as ``needs_ocr`` and stage 02
    will handle it.

Output
------
``data/extracted.jsonl`` — one JSON line per PDF with:
    {
        "file": "<basename>",
        "agency": ..., "incident_date": ..., "incident_location": ...,
        "title": ..., "description": ...,         # from the CSV manifest
        "pages": [
            {"page": 1, "text": "...", "needs_ocr": false},
            ...
        ]
    }

Why JSONL? Streamable, append-friendly, plays nicely with
``langchain_community.document_loaders.JSONLoader``. One row per file
keeps page-level structure intact for citation later.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.corpus import all_pdfs, manifest_rows, basename_from_url

OUT_PATH = Path(__file__).resolve().parent / "data" / "extracted.jsonl"

# Heuristic: pages with fewer than this many printable chars are almost
# certainly scans. Empty pages exist legitimately (cover sheets) but a
# false positive here just means stage 02 OCRs an empty page — cheap.
MIN_CHARS = 50


def load_manifest() -> dict[tuple[str, str], dict]:
    """Map (release_id, PDF basename) -> CSV row metadata. Tries multiple
    basename variants because the bash and python downloaders sanitized
    names differently (em-dashes, brackets, apostrophes)."""
    out: dict[tuple[str, str], dict] = {}
    for r, rid in manifest_rows():
        link = (r.get("PDF | Image Link") or "").strip()
        if not link.lower().endswith(".pdf"):
            continue
        raw = basename_from_url(link).replace(" ", "_")
        cleaned = "".join(c if c.isalnum() or c in "._-[]" else "_" for c in raw)
        row = {
            "title": (r.get("Title") or "").strip().replace("\n", " "),
            "agency": (r.get("Agency") or "").strip(),
            "incident_date": (r.get("Incident Date") or "").strip(),
            "incident_location": (r.get("Incident Location") or "").strip(),
            "description": (r.get("Description Blurb") or "").strip(),
            "release_date": (r.get("Release Date") or "").strip(),
            "release_id": rid,
        }
        out[(rid, raw)] = row
        out[(rid, cleaned)] = row
    return out


def extract_pdf(pdf_path: Path, release_id: str) -> dict:
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            pages.append(
                {
                    "page": i,
                    "text": text,
                    "needs_ocr": len(text) < MIN_CHARS,
                }
            )
    return {"file": pdf_path.name, "release_id": release_id, "pages": pages}


def _worker(args: tuple[str, str]) -> dict:
    path_str, rid = args
    return extract_pdf(Path(path_str), rid)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    pdfs = list(all_pdfs())
    print(f"Extracting {len(pdfs)} PDFs across "
          f"{len({rid for _, rid in pdfs})} release(s)...", file=sys.stderr)

    t0 = time.time()
    needs_ocr_count = 0
    with OUT_PATH.open("w") as out, ProcessPoolExecutor() as ex:
        futs = {ex.submit(_worker, (str(p), rid)): p for p, rid in pdfs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            meta = manifest.get((rec["release_id"], rec["file"]), {})
            rec.update(meta)
            n_ocr = sum(1 for p in rec["pages"] if p["needs_ocr"])
            if n_ocr > 0:
                needs_ocr_count += 1
            rec["pages_total"] = len(rec["pages"])
            rec["pages_needing_ocr"] = n_ocr
            out.write(json.dumps(rec) + "\n")
            if i % 10 == 0 or i == len(pdfs):
                print(f"  {i}/{len(pdfs)}  [{rec['release_id']}] {rec['file']} "
                      f"({n_ocr}/{len(rec['pages'])} pages need OCR)",
                      file=sys.stderr)

    print(
        f"Done in {time.time() - t0:.1f}s. "
        f"{needs_ocr_count}/{len(pdfs)} PDFs have scanned pages.",
        file=sys.stderr,
    )
    print(f"Output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
