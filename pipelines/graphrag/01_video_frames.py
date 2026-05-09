"""GraphRAG stage 01 — video → keyframes → vision caption + CLIP embedding.

Why scene-cut keyframes instead of every Nth frame
---------------------------------------------------
A 90-second declassified UAP clip might have one or two "the actual
thing" moments. Sampling 1 frame/second wastes 80+ vision-API calls on
near-duplicate frames; sampling 1 frame/30s misses the moment.

PySceneDetect's `ContentDetector` watches the HSV histogram for sudden
content shifts and emits a frame at each scene boundary — typically
3–10 keyframes per minute on real footage, vs. blanket-sampled rates
that produce 30–60. We then add the *first* and *middle* frame so even
a single-shot clip yields at least one frame to caption.

Two embeddings per frame
------------------------
1) CLIP visual embedding — for "find me UFO footage that looks like
   this" image-similarity queries. ViT-B/32 is the workhorse: 512-dim,
   fast on CPU, well-supported by `open_clip`.
2) bge-small text embedding of the caption — for natural-language
   queries ("triangular craft over water"). Lets a single text query
   hit the same retriever that PDF chunks use.

Storing both means stage 06 can answer either modality without
re-embedding at query time.

Output
------
`data/video_frames.jsonl` — one row per keyframe:
    {
      "video_file": "dvids_...mp4",
      "frame_idx": 42, "timestamp_s": 12.34,
      "caption": "...", "object_kind": "...",
      "visible_text": "...", "entities": [...],
      "clip_embedding": [...512 floats...],
      "text_embedding": [...384 floats...],
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PIPELINES = Path(__file__).resolve().parent.parent
VID_DIR = REPO / "videos"
OUT_PATH = Path(__file__).resolve().parent / "data" / "video_frames.jsonl"
FRAMES_DIR = Path(__file__).resolve().parent / "data" / "frames"

# Lower threshold = more sensitive = more keyframes. 27 is the
# PySceneDetect default; 22 picks up the subtle camera-cuts common in
# old gun-camera and dashcam footage.
SCENE_THRESHOLD = 22

sys.path.insert(0, str(PIPELINES))


def detect_keyframes(video_path: Path, out_dir: Path) -> list[tuple[int, float, Path]]:
    """Return a list of (frame_idx, timestamp_seconds, image_path)."""
    from scenedetect import detect, ContentDetector
    from scenedetect.video_splitter import save_images

    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = detect(str(video_path), ContentDetector(threshold=SCENE_THRESHOLD))

    if not scenes:
        # Single-shot clip — fall back to first + middle + last frames.
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        targets = [0, total // 2, max(0, total - 1)]
        out: list[tuple[int, float, Path]] = []
        for idx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            p = out_dir / f"{video_path.stem}_f{idx:06d}.jpg"
            cv2.imwrite(str(p), frame)
            out.append((idx, idx / fps, p))
        cap.release()
        return out

    # save_images writes one frame at the *start* of each scene.
    save_images(
        scenes, video_path, num_images=1, output_dir=out_dir,
        image_name_template="$VIDEO_NAME-Scene-$SCENE_NUMBER",
    )
    out = []
    for i, (start, _) in enumerate(scenes, start=1):
        # save_images uses 1-indexed scene numbers and the ORIGINAL stem
        path = out_dir / f"{video_path.stem}-Scene-{i:03d}-01.jpg"
        if not path.exists():
            # Older PySceneDetect versions use a different template
            for cand in out_dir.glob(f"{video_path.stem}-Scene-{i:03d}*.jpg"):
                path = cand
                break
        if path.exists():
            out.append((start.frame_num, start.get_seconds(), path))
    return out


def caption_frame(image_path: Path) -> dict:
    """Use Claude CLI vision via @-path attachment to caption a frame."""
    from llm import ClaudeCLIChatModel
    from langchain_core.messages import HumanMessage
    from graphrag.schema import FrameCaption

    model = ClaudeCLIChatModel(model="sonnet", timeout_seconds=90)
    structured = model.with_structured_output(FrameCaption)
    prompt = (
        f"Look at the attached image: @{image_path}\n\n"
        "This is a frame from declassified UAP/UFO footage. Describe it "
        "factually in one sentence. If a UAP-like object is visible, classify "
        "its shape (disc/saucer/triangle/cigar/sphere/light/other or null). "
        "Transcribe any visible text (radar overlays, timestamps, HUD). "
        "List salient entities (aircraft type, terrain, water, sky condition, "
        "instruments)."
    )
    return structured.invoke([HumanMessage(content=prompt)]).model_dump()


def embed_clip_image(image_paths: list[Path]) -> list[list[float]]:
    """Batch CLIP image embeddings."""
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    out: list[list[float]] = []
    with torch.no_grad():
        for p in image_paths:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            v = model.encode_image(img)
            v = v / v.norm(dim=-1, keepdim=True)  # cosine-friendly
            out.append(v[0].tolist())
    return out


def embed_caption(captions: list[str]) -> list[list[float]]:
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    return emb.embed_documents(captions)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Process only first N videos (debug).")
    args = ap.parse_args(argv)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VID_DIR.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"[frames] {len(videos)} videos", file=sys.stderr)

    # Resume support: skip videos already in output.
    done_videos: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line:
                done_videos.add(json.loads(line)["video_file"])

    with OUT_PATH.open("a") as out:
        for vi, video in enumerate(videos, 1):
            if video.name in done_videos:
                print(f"  [{vi}/{len(videos)}] SKIP {video.name}", file=sys.stderr)
                continue

            keyframes = detect_keyframes(video, FRAMES_DIR / video.stem)
            print(f"  [{vi}/{len(videos)}] {video.name}  {len(keyframes)} keyframes", file=sys.stderr)
            if not keyframes:
                continue

            captions = [caption_frame(p) for _, _, p in keyframes]
            captions_text = [c["caption"] for c in captions]
            clip_vecs = embed_clip_image([p for _, _, p in keyframes])
            text_vecs = embed_caption(captions_text)

            for (idx, ts, path), cap, cv, tv in zip(keyframes, captions, clip_vecs, text_vecs):
                row = {
                    "video_file": video.name,
                    "frame_path": str(path.relative_to(FRAMES_DIR.parent)),
                    "frame_idx": int(idx),
                    "timestamp_s": float(ts),
                    "caption": cap["caption"],
                    "object_kind": cap.get("object_kind"),
                    "visible_text": cap.get("visible_text", ""),
                    "entities": cap.get("salient_entities", []),
                    "clip_embedding": cv,
                    "text_embedding": tv,
                }
                out.write(json.dumps(row) + "\n")

    print(f"Output: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
