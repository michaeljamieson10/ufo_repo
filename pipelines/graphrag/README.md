# GraphRAG layer for the UAP corpus

A FalkorDB-backed knowledge graph + multimodal frame index that you
can literally talk to with Cypher *and* natural language. Sits on top
of (and reuses) the simpler `pipelines/` Chroma flow.

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │            FalkorDB graph             │
                    │  (graph + vector indexes in one DB)   │
                    └──────────────────────────────────────┘
                                    ▲
                                    │ load
   ┌─────────────────────────┬──────┴─────┬──────────────────────────┐
   │                         │            │                          │
   ▼                         ▼            ▼                          ▼
01 video_frames        02 image_caption   03 entity_extract     04 communities
PySceneDetect →        Claude vision      Claude CLI →           Leiden →
Claude vision +        + CLIP             entities + incidents   summaries
CLIP per frame                            per chunk
                                                                       ▲
   ▲                         ▲                         ▲               │
   │                         │                         │               │
videos (28)           images (14 primary)        chunks of:           │
                                                  - PDFs (115)        │
                                                  - frame captions    │
                                                  - image captions    │
                                                                       │
                                                                       ▼
                                                          ┌─────────────────────┐
                                                          │  06 graph agent      │
                                                          │  text-to-Cypher │ vec│
                                                          │  │ community │ hybrid│
                                                          └─────────────────────┘
```

## Why this design (in one minute)

- **Plain RAG** answers "what does file X say" well; chokes on
  "compare 1947 vs 2024" or "show all FBI cases with named witnesses."
- **GraphRAG** (Microsoft, 2024) adds entity extraction + community
  detection: now you have a *structured* index alongside the vectors.
- **FalkorDB** holds graph + vectors in one Redis-protocol process.
  Single Docker container, fast Cypher, hot in 2026.
- **CLIP frame embeddings** make video footage retrievable by visual
  similarity, not just transcripts/captions.
- **LangGraph** state machine routes each question to the right
  retrieval strategy (Cypher / vector / community summary / hybrid).

## Quickstart

```bash
cd ~/Code/ufo_repo/pipelines/graphrag

# 1) Install (in addition to ../requirements.txt)
pip install -r requirements.txt

# 2) Start FalkorDB
docker compose up -d
# Browser UI: http://localhost:3000

# 3) Build everything (will take 1-3 hours of CLI time)
./run_all.sh
```

## Stages at a glance

| Stage | Reads | Writes | LLM time |
|-------|-------|--------|----------|
| 01 video_frames | `videos/*.mp4` | `data/video_frames.jsonl` + frame jpgs | ~20 min |
| 02 image_caption | `images/` (IMG-type rows) | `data/images.jsonl` | ~5 min |
| 03 entity_extract | PDFs jsonl + frame/image captions | `data/chunk_entities.jsonl` | 1-2 hr |
| 04 communities | `chunk_entities.jsonl` | `data/communities.jsonl` | ~10 min |
| 05 load_falkordb | all of the above | FalkorDB graph | < 1 min |
| 06 graph_agent | (live queries) | (printed answer) | per query |

## Talking to the database

Three retrieval modes, picked automatically per question:

```bash
# Structured — text-to-Cypher
python 06_graph_agent.py "incidents in 1947 with named witnesses"
# → Claude writes:
#     MATCH (i:Incident {date:'1947'})-[:WITNESSED_BY]->(p:Person)
#     RETURN i.summary, i.location, p.name LIMIT 50

# Semantic — vector over chunks + frame captions
python 06_graph_agent.py "ramjet and pulse-jet propulsion proposals"

# Global — community summaries
python 06_graph_agent.py --mode global "major themes in the FBI files"

# Force hybrid (run all three)
python 06_graph_agent.py --mode hybrid "compare 1947 disc reports to 2024 navy encounters"
```

## Schema reference

See `schema.py` for the full Pydantic + Cypher schema. Highlights:

- Nodes: `Document`, `Chunk`, `Frame`, `Agency`, `Person`, `Location`,
  `Object`, `CaseFile`, `Date`, `Incident`, `Community`
- Edges: `HAS_CHUNK`, `FROM_AGENCY`, `IN_CASEFILE`, `IN_VIDEO`,
  `MENTIONS`, `OCCURRED_AT`, `ON_DATE`, `INVOLVES`, `WITNESSED_BY`,
  `REPORTED_IN`, `HANDLED_BY`, `CONTAINS`
- Vector indexes:
  - `Chunk.text_embedding` (384-dim, bge-small-en-v1.5)
  - `Frame.clip_embedding` (512-dim, ViT-B/32 visual)
  - `Frame.text_embedding` (384-dim, caption text)
  - `Community.text_embedding` (384-dim, summary text)

## Interview talking points

The pipeline lights up almost every "modern RAG" probe:

| Concept | Where it lives |
|---------|----------------|
| Hybrid retrieval (BM25 + dense + rerank) | `../04_retrieve.py` |
| Multimodal RAG (CLIP + caption dual index) | `01_video_frames.py`, `02_image_caption.py` |
| GraphRAG (entity + community + summaries) | `03_entity_extract.py`, `04_communities.py` |
| Text-to-Cypher / NL→DSL | `06_graph_agent.py:text_to_cypher` |
| Leiden community detection | `04_communities.py:detect_communities` |
| LangGraph routing & state machines | `06_graph_agent.py:build_graph` |
| Local-first deployment (no API keys) | `pipelines/llm/cli_chat.py` |
| Schema-bounded extraction with Pydantic | `schema.py` |

## What's not done (yet)

- **Entity resolution beyond exact match** — "FBI" / "Federal Bureau"
  collapse correctly, but two spelling variants of a witness name
  remain separate. A name-blocking dedup pass would help.
- **Drift over time** — re-runs upsert, but if the manifest changes
  upstream we don't garbage-collect orphan nodes.
- **Streaming** — agent runs to completion before printing. LangGraph
  supports streaming if you want token-by-token output.
- **Eval** — no LLM-judge eval harness (yet). For interviews this is
  the obvious next module: a small set of question/expected-source
  pairs + ragas-style metrics.
