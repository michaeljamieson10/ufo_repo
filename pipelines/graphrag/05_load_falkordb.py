"""GraphRAG stage 05 — push everything into FalkorDB.

What this stage does
--------------------
Reads all the JSONL outputs from earlier stages and writes them as a
single graph in FalkorDB:

  Documents — every PDF, video, image
  Chunks    — every retrievable text span (PDF pages → chunks)
  Frames    — every keyframe
  Entities  — Agency / Person / Location / Object / CaseFile / Date
  Incidents — discrete events
  Communities — Leiden clusters

Plus the relationships from `schema.py`. Embeddings (chunk text,
frame CLIP, frame caption text, community summary) are stored as
node properties; vector indexes from `schema.CYPHER_INDEXES` make
them queryable.

Why batched MERGEs
------------------
FalkorDB's Cypher endpoint is fast on bulk inserts but each statement
incurs round-trip overhead. We batch ~500 nodes/edges per `UNWIND`
statement (single round-trip, server iterates internally). For 100k
chunks that's the difference between 10 seconds and 10 minutes.

Idempotency
-----------
Every node has a stable `id`/`name` field. We use `MERGE` rather than
`CREATE` everywhere so re-runs upsert. Re-running after fixing a
single chunk's extraction is cheap.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(PIPELINES))


def get_db():
    try:
        from falkordb import FalkorDB
    except ImportError:
        sys.exit("pip install falkordb")
    db = FalkorDB(host="localhost", port=6379)
    return db.select_graph("uap")


def normalize_entity_id(t: str, name: str) -> str:
    n = name.strip().lower()
    n = "".join(c if c.isalnum() else "_" for c in n)
    return f"{t}::{n}"


def run_indexes(g) -> None:
    from graphrag.schema import CYPHER_INDEXES
    for stmt in CYPHER_INDEXES:
        try:
            g.query(stmt)
        except Exception as e:
            # Indexes may already exist from a prior run; FalkorDB raises.
            msg = str(e).lower()
            if "exist" not in msg and "already" not in msg:
                print(f"  index warning: {e}", file=sys.stderr)


def load_documents_chunks_pdfs(g) -> None:
    """PDFs: create Document, Agency, Chunk nodes + edges. Embed chunks."""
    pdf_jsonl = PIPELINES / "data" / "extracted_ocr.jsonl"
    if not pdf_jsonl.exists():
        pdf_jsonl = PIPELINES / "data" / "extracted.jsonl"
    if not pdf_jsonl.exists():
        print("  no PDFs jsonl", file=sys.stderr)
        return

    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    BATCH = 200
    chunks_batch: list[dict] = []
    rels_batch: list[dict] = []

    def flush() -> None:
        if chunks_batch:
            g.query(
                "UNWIND $rows AS r "
                "MERGE (d:Document {file: r.file}) "
                "SET d.kind='pdf', d.title=r.title, d.agency=r.agency "
                "MERGE (a:Agency {name: r.agency}) "
                "MERGE (d)-[:FROM_AGENCY]->(a) "
                "MERGE (c:Chunk {id: r.cid}) "
                "SET c.text=r.text, c.page=r.page, c.text_embedding=vecf32(r.emb) "
                "MERGE (d)-[:HAS_CHUNK]->(c)",
                {"rows": chunks_batch},
            )
            chunks_batch.clear()

    n = 0
    for line in pdf_jsonl.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        for p in rec.get("pages", []):
            text = (p.get("text") or "").strip()
            if not text:
                continue
            for ci in range(0, len(text), 850):  # ~1000 with 150 overlap
                chunk = text[ci : ci + 1000]
                cid = f"{rec['file']}:{p['page']}:{ci // 850}"
                chunks_batch.append({
                    "cid": cid,
                    "file": rec["file"],
                    "title": (rec.get("title") or "")[:300],
                    "agency": rec.get("agency", "") or "OTHER",
                    "page": p["page"],
                    "text": chunk,
                    "emb": emb.embed_query(chunk),
                })
                n += 1
                if len(chunks_batch) >= BATCH:
                    flush()
                    print(f"  pdf chunks loaded: {n}", file=sys.stderr)
    flush()
    print(f"  pdf chunks total: {n}", file=sys.stderr)


def load_video_frames(g) -> None:
    fp = DATA / "video_frames.jsonl"
    if not fp.exists():
        return
    BATCH = 100
    rows: list[dict] = []

    def flush() -> None:
        if rows:
            g.query(
                "UNWIND $rows AS r "
                "MERGE (v:Document {file: r.video_file}) "
                "SET v.kind='video' "
                "MERGE (f:Frame {id: r.fid}) "
                "SET f.timestamp_s=r.ts, f.caption=r.caption, "
                "    f.object_kind=r.object_kind, "
                "    f.clip_embedding=vecf32(r.clip_emb), "
                "    f.text_embedding=vecf32(r.text_emb) "
                "MERGE (f)-[:IN_VIDEO]->(v)",
                {"rows": rows},
            )
            rows.clear()

    for line in fp.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        rows.append({
            "fid": f"{rec['video_file']}:{rec['frame_idx']}",
            "video_file": rec["video_file"],
            "ts": rec["timestamp_s"],
            "caption": rec["caption"],
            "object_kind": rec.get("object_kind") or "",
            "clip_emb": rec["clip_embedding"],
            "text_emb": rec["text_embedding"],
        })
        if len(rows) >= BATCH:
            flush()
    flush()


def load_images(g) -> None:
    fp = DATA / "images.jsonl"
    if not fp.exists():
        return
    BATCH = 50
    rows: list[dict] = []

    def flush() -> None:
        if rows:
            g.query(
                "UNWIND $rows AS r "
                "MERGE (d:Document {file: r.file}) "
                "SET d.kind='image', d.caption=r.caption, "
                "    d.clip_embedding=vecf32(r.clip_emb), "
                "    d.text_embedding=vecf32(r.text_emb) "
                "MERGE (a:Agency {name: r.agency}) "
                "MERGE (d)-[:FROM_AGENCY]->(a)",
                {"rows": rows},
            )
            rows.clear()

    for line in fp.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        rows.append({
            "file": rec["file"],
            "caption": rec.get("caption", ""),
            "agency": rec.get("agency", "") or "OTHER",
            "clip_emb": rec["clip_embedding"],
            "text_emb": rec["text_embedding"],
        })
        if len(rows) >= BATCH:
            flush()
    flush()


def load_entities(g) -> None:
    fp = DATA / "chunk_entities.jsonl"
    if not fp.exists():
        return

    # Entity type → label. We use `Object` (capitalized) as a label
    # rather than a generic Entity to keep Cypher pretty.
    LABEL = {
        "Agency": "Agency", "Person": "Person", "Location": "Location",
        "Object": "Object", "CaseFile": "CaseFile", "Date": "Date",
    }
    by_label: dict[str, list[dict]] = defaultdict(list)
    mentions: list[dict] = []
    incidents: list[dict] = []

    for line in fp.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        cid = row["chunk_id"]
        ext = row.get("extraction") or {}
        for e in ext.get("entities") or []:
            label = LABEL.get(e["type"])
            if not label:
                continue
            eid = normalize_entity_id(e["type"], e["name"])
            by_label[label].append({"id": eid, "name": e["name"]})
            mentions.append({"cid": cid, "eid": eid, "label": label})
        for i, inc in enumerate(ext.get("incidents") or []):
            incidents.append({
                "id": f"{cid}:inc{i}",
                "cid": cid,
                "summary": inc.get("summary", ""),
                "date": inc.get("date") or "",
                "location": inc.get("location") or "",
                "agency": inc.get("agency") or "",
                "object_kind": inc.get("object_kind") or "",
                "witnesses": inc.get("witnesses") or [],
            })

    for label, rows in by_label.items():
        for i in range(0, len(rows), 500):
            g.query(
                f"UNWIND $rows AS r MERGE (n:{label} {{id: r.id}}) SET n.name=r.name",
                {"rows": rows[i:i+500]},
            )

    # MENTIONS edges. Chunks must already exist (loaded earlier).
    for i in range(0, len(mentions), 500):
        g.query(
            "UNWIND $rows AS r "
            "MATCH (c:Chunk {id: r.cid}) "
            "MATCH (e {id: r.eid}) "
            "MERGE (c)-[:MENTIONS]->(e)",
            {"rows": mentions[i:i+500]},
        )

    # Incidents (one node + edges per incident).
    for i in range(0, len(incidents), 200):
        g.query(
            "UNWIND $rows AS r "
            "MERGE (i:Incident {id: r.id}) "
            "SET i.summary=r.summary, i.date=r.date, i.location=r.location, "
            "    i.agency=r.agency, i.object_kind=r.object_kind "
            "WITH i, r "
            "MATCH (c:Chunk {id: r.cid}) "
            "MERGE (i)-[:REPORTED_IN]->(c) "
            "FOREACH (w IN r.witnesses | "
            "  MERGE (p:Person {id: 'Person::' + toLower(replace(w, ' ', '_'))}) "
            "  SET p.name=w "
            "  MERGE (i)-[:WITNESSED_BY]->(p))",
            {"rows": incidents[i:i+200]},
        )


def load_communities(g) -> None:
    fp = DATA / "communities.jsonl"
    if not fp.exists():
        return
    rows = []
    contains = []
    for line in fp.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        rows.append({
            "id": rec["id"], "size": rec["size"], "summary": rec["summary"],
            "emb": rec["summary_embedding"],
        })
        for m in rec["members"]:
            contains.append({"cid": rec["id"], "eid": m})

    for i in range(0, len(rows), 100):
        g.query(
            "UNWIND $rows AS r "
            "MERGE (c:Community {id: r.id}) "
            "SET c.size=r.size, c.summary=r.summary, "
            "    c.text_embedding=vecf32(r.emb)",
            {"rows": rows[i:i+100]},
        )
    for i in range(0, len(contains), 500):
        g.query(
            "UNWIND $rows AS r "
            "MATCH (c:Community {id: r.cid}) "
            "MATCH (e {id: r.eid}) "
            "MERGE (c)-[:CONTAINS]->(e)",
            {"rows": contains[i:i+500]},
        )


def main() -> None:
    g = get_db()
    print("[falkordb] indexes...", file=sys.stderr)
    run_indexes(g)
    print("[falkordb] PDFs → Documents + Chunks + Agencies", file=sys.stderr)
    load_documents_chunks_pdfs(g)
    print("[falkordb] video frames", file=sys.stderr)
    load_video_frames(g)
    print("[falkordb] primary images", file=sys.stderr)
    load_images(g)
    print("[falkordb] entities + incidents", file=sys.stderr)
    load_entities(g)
    print("[falkordb] communities", file=sys.stderr)
    load_communities(g)

    # Sanity counts.
    for label in ("Document", "Chunk", "Frame", "Agency", "Person",
                  "Location", "Object", "CaseFile", "Incident", "Community"):
        try:
            r = g.query(f"MATCH (n:{label}) RETURN count(n) AS n").result_set
            print(f"  {label:>10}: {r[0][0]}", file=sys.stderr)
        except Exception:
            pass


if __name__ == "__main__":
    main()
