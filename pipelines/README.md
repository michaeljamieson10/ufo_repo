# UAP RAG Pipeline

End-to-end LangChain + LangGraph pipeline that turns the war.gov UAP
release into a question-answering system with citations.

```
PDFs / videos / images
         │
         ▼
   01_ingest.py     PyMuPDF text extraction → flag scanned pages
         │
         ▼
   02_ocr.py        Tesseract for the ~half that need it
         │
         ▼
   03_chunk_embed.py  Recursive splitter + bge-small embeddings → Chroma
         │
         ▼
   04_retrieve.py   BM25 + dense ensemble + cross-encoder rerank
         │
         ▼
   05_agent.py      LangGraph: classify → retrieve → synthesize w/ cites

   06_multimodal.py (optional) Whisper for videos, Claude vision for images
```

## Why these choices

The pipeline targets the techniques interviewers probe most:

| Stage | Technique | What's interview-relevant |
|-------|-----------|---------------------------|
| 01 | Text-layer probe via PyMuPDF | "How do you tell a scanned PDF from a digital one?" |
| 02 | OCR with `--psm 6` for typed docs | "How would you handle low-quality scans?" |
| 03 | `RecursiveCharacterTextSplitter`, metadata filters | Chunk size/overlap trade-off; metadata-filtered retrieval |
| 04 | Hybrid (BM25 + dense) + reranker | The single most-asked production-RAG question |
| 05 | LangGraph state machine | "When would you reach for an agent vs a chain?" |
| 06 | Whisper + Claude vision | Multimodal RAG without CLIP joint embeddings |

## LLM backend

Defaults to the local **Claude Code CLI** (`claude -p ...`) — no API
key. The wrapper at `llm/cli_chat.py` is lifted from
`~/Code/decksmith-remote-work/python/voice_clone/lcgraph/llm/`. It
spawns `claude` (or `codex` with the `codex:` prefix) as a subprocess
and surfaces it as a LangChain `BaseChatModel`, including
`with_structured_output(Schema)` for JSON-shaped outputs.

Switch backends with one env var:

```bash
export UFO_LLM_MODEL=claude-cli:sonnet     # default — local Claude Code CLI
export UFO_LLM_MODEL=codex:gpt-5           # local Codex CLI
export UFO_LLM_MODEL=claude-sonnet-4-6     # cloud Anthropic SDK (needs API key)
export UFO_LLM_MODEL=gpt-4o                # cloud OpenAI SDK   (needs API key)
```

## Setup

```bash
cd ~/Code/ufo_repo/pipelines
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract            # for stage 02
./run_all.sh                      # runs 01 → 02 → 03
```

First run downloads:
- `BAAI/bge-small-en-v1.5` (~120 MB) — embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB) — reranker

After that, everything is local.

## Querying

```bash
# Just retrieval (sanity-check the corpus)
python 04_retrieve.py 'oak ridge sighting 1950'

# Full agent with citations
python 05_agent.py 'What does the FBI file 62-HQ-83894 say about disc-shaped craft?'
python 05_agent.py 'compare 1947 and 2024 sightings'
python 05_agent.py 'which incidents involved Navy aircraft over the Atlantic?'
```

## Notes per stage

### 01_ingest.py
PyMuPDF (`fitz`) is the right tool here: faster than `pypdf`, no Java
dependency like Tika, and it returns layout-preserving text rather
than raw bytes. The `MIN_CHARS = 50` cutoff for "this page is a scan"
is a heuristic — false positives just mean stage 02 OCRs a blank page,
which is cheap.

### 02_ocr.py
Rasterizes at 200 DPI which is the typewriter-era sweet spot.
`--psm 6` ("uniform block of text") beats Tesseract's default for
single-column declassified docs. For better OCR quality, swap in
`unstructured.io` or [`marker`](https://github.com/VikParuchuri/marker).

### 03_chunk_embed.py
Chunks are 1000/150. Each chunk carries `file`, `page`, `agency`,
`incident_date`, `incident_location` metadata so stage 04 can filter
("only FBI files from 1947") *before* the dense search runs. Stable
chunk IDs (`file:page:chunk`) make re-runs idempotent.

### 04_retrieve.py
Hybrid retrieval is the production answer: BM25 catches case file IDs
and proper nouns, dense catches paraphrase, the cross-encoder reranker
fixes the "good but not best" ordering. 50/50 weights are the
starting point — bias toward BM25 if your queries are mostly named
entities (case IDs), dense for natural-language questions.

### 05_agent.py
Three-node LangGraph: `classify` extracts a metadata filter from the
query, `retrieve` runs the hybrid search with that filter, `synthesize`
asks the Claude CLI to answer using only the retrieved chunks with
inline `[N]` citations. Easy to extend with a `critique` node that
loops back to `retrieve` for multi-hop questions.

### 06_multimodal.py
Optional. `faster-whisper` (small, int8) transcribes the 28 DVIDS
videos in ~10 minutes on a CPU. Claude vision via the CLI captions
each image with a 3-field schema (caption, entities, visible text).
Both outputs become JSONL with the same shape as `extracted.jsonl`,
so re-running stage 03 indexes them alongside the PDFs.

## Common interview probes the pipeline answers

- *Why hybrid retrieval?* — see 04, header comment.
- *How do you cite sources?* — see 05, `synthesize` prompt + State.docs.
- *What if you had no API budget?* — that's the default config: local
  embeddings + Claude Code CLI subprocess.
- *How would you handle scanned PDFs?* — stage 01's `needs_ocr` flag
  + stage 02 batched Tesseract.
- *Multi-hop / agent vs chain?* — LangGraph state in 05; trivial to
  add a critique→retrieve loop.
