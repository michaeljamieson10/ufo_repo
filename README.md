# UFO Repo — war.gov UAP Release Mirror

Local, rolling mirror of the U.S. Department of War's UAP/UFO file
releases. The DoW announced a multi-tranche release; each tranche
lives in its own self-contained directory under `releases/`.

Source: https://www.war.gov/ufo/

## Releases

| ID | Date | Files | Size | Source |
|----|------|------:|-----:|--------|
| [release_01](releases/release_01/INDEX.md) | 2026-05-08 | 115 PDFs, 138 images, 28 videos | 3.8 GB | [war.gov manifest](https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv) |

When new tranches drop, add a row + a `KNOWN_MANIFESTS` entry in
`scripts/download.py` and run `python scripts/download.py --release N`.

## Layout

```
ufo_repo/
├── releases/
│   └── release_01/
│       ├── pdfs/           # 115 PDFs (FBI / DoD / agency case files)
│       ├── images/          # 138 images (FBI sensor + slideshow + thumbs)
│       ├── videos/          # 28 DVIDS-hosted mp4s
│       ├── metadata/
│       │   ├── uap-csv.csv  # original war.gov manifest
│       │   ├── manifest.json
│       │   ├── pdf_urls.txt
│       │   ├── image_urls.txt
│       │   └── dvids_video_ids.txt
│       ├── download.log
│       └── INDEX.md
├── scripts/
│   ├── download.py          # `--release N` to mirror a tranche
│   └── corpus.py            # cross-release file discovery used by pipelines
├── pipelines/               # LangChain RAG (Chroma) — see pipelines/README.md
│   ├── 01_ingest.py … 06_multimodal.py
│   └── graphrag/            # FalkorDB GraphRAG stack — see graphrag/README.md
└── README.md
```

## Mirror a release

```bash
# release 1 — uses the known manifest URL
python scripts/download.py --release 1

# future releases — pass the manifest URL the war.gov page surfaces
python scripts/download.py --release 2 \
  --manifest-url https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv-r2.csv
```

The downloader is idempotent (skips existing files) and runs a
content-hash dedup at the end, so re-runs are cheap and safe.

## Pipelines

Two ingestion stacks live under `pipelines/`. Both walk every release
in `releases/*/` automatically (via `scripts/corpus.py`), so adding a
new tranche means re-running the pipeline, not rewiring it.

- **`pipelines/`** — baseline LangChain RAG. PyMuPDF + OCR, Chroma
  vectors, BM25 + dense + cross-encoder rerank, LangGraph agent with
  citations. Default LLM is the local Claude Code CLI (no API key).
- **`pipelines/graphrag/`** — FalkorDB-backed knowledge graph + CLIP
  frame index + Leiden community summaries + text-to-Cypher agent.

Each chunk in either pipeline is tagged with `release_id` so retrieval
can answer "what's new in release_NN?" out of the box.

## Notes on the war.gov source

- Page is rendered by Vue and exposes nothing static; the manifest
  CSV (`Portals/1/Interactive/2026/UFO/uap-csv.csv`) is where the file
  list actually lives.
- The site sits behind Akamai and 403s plain `curl`/`wget`. The
  downloader sends Chrome client-hint headers (`Sec-Ch-Ua*`,
  `Sec-Fetch-*`) which lets requests through.
- Videos aren't hosted on war.gov; the CSV provides DVIDS video IDs
  and the script scrapes `dvidshub.net/video/<id>` for the embedded
  CloudFront mp4 URL.
