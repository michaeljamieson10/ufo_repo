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

## Sharing this repo

Three layers, separated by cost-to-rebuild:

1. **Source files** (`releases/*/{pdfs,images,videos}/`) — gitignored.
   They're public domain, but heavy. Re-mirror with
   `python scripts/download.py --release N`. ~3.8 GB for release 01.
2. **Pipeline JSONLs** (`pipelines/data/*.jsonl`,
   `pipelines/graphrag/data/*.jsonl`) — **committed**. ~30 MB. Encodes
   every expensive LLM extraction (OCR, captions, entities,
   communities) so cloners don't pay for them again.
3. **Vector / graph DBs** (`pipelines/store/`,
   `pipelines/graphrag/data/falkordb/`) — gitignored. Either rebuild
   from JSONLs in ~5 min via `scripts/make_dbs.sh`, or download the
   pre-built tarball from a GitHub Release.

### Cloning the repo

```bash
git clone https://github.com/<you>/ufo_repo.git
cd ufo_repo
python scripts/download.py --release 1   # mirror source files (~3.8 GB)
scripts/make_dbs.sh                      # rebuild Chroma + FalkorDB from JSONLs
```

### Publishing a tarball release

```bash
scripts/release_tarball.sh
gh release create v$(date +%Y.%m.%d) ufo-dbs-*.tar.gz \
  --title "Pre-built DBs $(date +%Y-%m-%d)" \
  --notes "Chroma + FalkorDB built from release_01."
```

Downstream users can then skip the rebuild:
```bash
curl -L https://github.com/<you>/ufo_repo/releases/latest/download/ufo-dbs-*.tar.gz | tar -xzf -
```

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

### Why not the LangChain web loaders?

LangChain ships a stack of `*Loader` classes for fetching pages —
`WebBaseLoader`, `RecursiveUrlLoader`, `AsyncHtmlLoader`,
`SitemapLoader`, `PlaywrightURLLoader`. None of them are used here.
That's deliberate, and the reason matters for anyone reading this
repo and wondering if they should "swap to the LangChain way":

1. **Akamai bot detection.** `WebBaseLoader`, `RecursiveUrlLoader`,
   and `AsyncHtmlLoader` all wrap `requests` / `aiohttp` with a
   default User-Agent. war.gov 403s every request that doesn't carry
   a Chrome-shaped client-hint set (`Sec-Ch-Ua`, `Sec-Ch-Ua-Mobile`,
   `Sec-Ch-Ua-Platform`, `Sec-Fetch-*`, `Upgrade-Insecure-Requests`)
   together with a current desktop UA. You can pass `header_template=`
   to those loaders, but at that point you're hand-rolling the
   anti-bot headers anyway — the loader adds nothing.
2. **The page is Vue, not HTML.** `RecursiveUrlLoader` only sees
   what's in the static response; the file list is fetched by the
   Vue runtime from the CSV manifest. So even with the right headers,
   the loader returns ~0 file URLs. The actual extraction is "fetch
   the manifest CSV directly" — which is one `urllib.request` call,
   not a crawl.
3. **`PlaywrightURLLoader` would work but is overkill.** Headless
   Chrome handles the bot check and runs the Vue, but spinning up a
   browser to download a CSV plus 281 known file URLs costs orders
   of magnitude more time and disk than a flat-file loop. Kept in
   reserve in case war.gov ever escalates to a JS challenge.
4. **DVIDS video URLs need scraping a second site.** The CSV gives
   DVIDS video IDs, not playable URLs. `dvidshub.net/video/<id>`
   embeds a CloudFront mp4 inside a `<source>` tag. A custom regex
   scrape is ~10 lines; wiring a recursive loader to follow per-id
   pages and extract `<source src=...>` is more code, not less.

The repo's split is consistent everywhere: **LangChain where it adds
composability** (chains, agents, retrievers, ensemble + reranker),
**bare libraries where it would only add layers** (HTTP fetch,
PyMuPDF, Tesseract, PySceneDetect, FalkorDB Cypher). See
`pipelines/README.md` for the LangChain side and
`scripts/download.py` for the urllib side.
