"""GraphRAG stage 02 — image vision-caption + CLIP embedding.

The corpus has 138 images but most are PDF thumbnails — already
covered by the PDF text indexing. We only want the *primary* IMG-type
images (FBI sensor photos + composite sketch) which the manifest tags
with Type='IMG'. Stage 02 reads the manifest, picks those, and runs
the same dual-embedding (caption + CLIP) treatment as stage 01 does
for frames.

Output
------
`data/images.jsonl` — same row shape as `video_frames.jsonl` minus
the video-specific fields, so stage 05 can merge the two.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PIPELINES = Path(__file__).resolve().parent.parent
META_CSV = REPO / "metadata" / "uap-csv.csv"
IMG_DIR = REPO / "images"
OUT_PATH = Path(__file__).resolve().parent / "data" / "images.jsonl"

sys.path.insert(0, str(PIPELINES))


def primary_image_files() -> list[tuple[Path, dict]]:
    """Manifest IMG-type rows -> (path, metadata)."""
    out: list[tuple[Path, dict]] = []
    with open(META_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("Type") or "").strip() != "IMG":
                continue
            link = (r.get("PDF | Image Link") or "").strip()
            if not link.startswith("http"):
                continue
            raw = urllib.parse.unquote(
                os.path.basename(urllib.parse.urlsplit(link).path)
            ).replace(" ", "_")
            for cand in (raw, "".join(c if c.isalnum() or c in "._-[]" else "_" for c in raw)):
                p = IMG_DIR / cand
                if p.exists():
                    out.append((p, {
                        "title": (r.get("Title") or "").strip(),
                        "agency": (r.get("Agency") or "").strip(),
                        "incident_date": (r.get("Incident Date") or "").strip(),
                        "incident_location": (r.get("Incident Location") or "").strip(),
                        "description": (r.get("Description Blurb") or "").strip(),
                    }))
                    break
    return out


def caption_image(image_path: Path):
    from llm import ClaudeCLIChatModel
    from langchain_core.messages import HumanMessage
    from graphrag.schema import FrameCaption

    model = ClaudeCLIChatModel(model="sonnet", timeout_seconds=90)
    structured = model.with_structured_output(FrameCaption)
    prompt = (
        f"Look at the attached image: @{image_path}\n\n"
        "This is a primary image from a declassified UAP/UFO release "
        "(FBI sensor photo, witness sketch, or similar). Describe it "
        "factually in one sentence. If a UAP-like object is depicted, "
        "classify its shape (disc/saucer/triangle/cigar/sphere/light/other "
        "or null). Transcribe any visible text (case numbers, captions, "
        "stamps). List salient entities."
    )
    return structured.invoke([HumanMessage(content=prompt)]).model_dump()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    targets = primary_image_files()
    print(f"[images] {len(targets)} primary images", file=sys.stderr)

    done: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line:
                done.add(json.loads(line)["file"])

    # Defer heavy embedding-model loads until we know we have work.
    from langchain_huggingface import HuggingFaceEmbeddings
    import open_clip
    import torch
    from PIL import Image

    text_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model.eval()

    with OUT_PATH.open("a") as out:
        for i, (path, meta) in enumerate(targets, 1):
            if path.name in done:
                continue
            cap = caption_image(path)
            text_vec = text_emb.embed_query(cap["caption"])
            with torch.no_grad():
                img = clip_pre(Image.open(path).convert("RGB")).unsqueeze(0)
                v = clip_model.encode_image(img)
                v = v / v.norm(dim=-1, keepdim=True)
                clip_vec = v[0].tolist()
            out.write(json.dumps({
                "file": path.name,
                "kind": "image",
                "caption": cap["caption"],
                "object_kind": cap.get("object_kind"),
                "visible_text": cap.get("visible_text", ""),
                "entities": cap.get("salient_entities", []),
                "text_embedding": text_vec,
                "clip_embedding": clip_vec,
                **meta,
            }) + "\n")
            print(f"  [{i}/{len(targets)}] {path.name}", file=sys.stderr)

    print(f"Output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
