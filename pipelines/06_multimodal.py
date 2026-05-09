"""Stage 06 — multimodal extras (videos + images).

Why this stage exists
---------------------
The first five stages cover the PDF text. The corpus also has 28 DVIDS
videos (UAP encounter footage with audio narration) and 138 images
(FBI sensor photos, slideshow stills). To put those in the same
retriever, we need to convert each to text:

  - Videos → Whisper transcribes the audio narration. The transcript
    becomes a "document" with the same metadata shape (file, page=1,
    agency, etc.) so it slots into Chroma alongside the PDF chunks.
  - Images → an LLM that can see (Claude Sonnet via the API or `claude
    -p` with file mention) writes a short caption + entity list.

This is "multimodal RAG" the practical way: we don't run a CLIP joint
embedding (overkill for 166 assets). We just text-ify both modalities
and let the existing dense + BM25 + rerank pipeline handle them.

Run order
---------
    python 06_multimodal.py videos       # writes data/videos.jsonl
    python 06_multimodal.py images       # writes data/images.jsonl
    python 03_chunk_embed.py             # rerun to merge into Chroma

Whisper note
------------
This script uses ``faster-whisper`` because it's ~5× faster than
``openai-whisper`` on CPU and produces compatible transcripts. The
``small`` model is the sweet spot — ``tiny`` mangles names, ``medium``+
takes minutes per 90s clip on a CPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
VID_DIR = REPO / "videos"
IMG_DIR = REPO / "images"
PRIMARY_IMAGES = REPO / "metadata" / "uap-csv.csv"


def transcribe_videos() -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("pip install faster-whisper")

    out = DATA / "videos.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = WhisperModel("small", device="cpu", compute_type="int8")

    videos = sorted(VID_DIR.glob("*.mp4"))
    print(f"Transcribing {len(videos)} videos...", file=sys.stderr)
    with out.open("w") as f:
        for i, v in enumerate(videos, 1):
            print(f"  [{i}/{len(videos)}] {v.name}", file=sys.stderr)
            segments, info = model.transcribe(str(v), beam_size=5)
            text = " ".join(s.text.strip() for s in segments)
            f.write(json.dumps({
                "file": v.name,
                "kind": "video",
                "duration_s": info.duration,
                "language": info.language,
                "pages": [{"page": 1, "text": text, "needs_ocr": False}],
            }) + "\n")
    print(f"Output: {out}", file=sys.stderr)


def caption_images() -> None:
    """Use the Claude CLI as a vision backend.

    The CLI accepts ``@<path>`` references in the prompt body and
    auto-attaches the file. We feed each image with a fixed prompt and
    parse a tiny structured response (caption + entities). One CLI
    spawn per image — slow, but trivial to run overnight.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm import ClaudeCLIChatModel
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field

    class ImageDesc(BaseModel):
        caption: str = Field(description="One-sentence factual description.")
        entities: list[str] = Field(description="Salient nouns/objects, lowercase.")
        text_in_image: str = Field(description="Any visible text, '' if none.")

    model = ClaudeCLIChatModel(model="sonnet", timeout_seconds=120)
    structured = model.with_structured_output(ImageDesc)

    out = DATA / "images.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Captioning {len(images)} images via Claude CLI...", file=sys.stderr)

    with out.open("w") as f:
        for i, img in enumerate(images, 1):
            prompt = (
                f"Look at the attached image: @{img}\n"
                "Describe it factually. List salient entities. Transcribe any "
                "visible text. This is a declassified UAP/UFO record image."
            )
            try:
                desc = structured.invoke([HumanMessage(content=prompt)])
            except Exception as e:
                print(f"  [{i}/{len(images)}] FAIL {img.name}: {e}", file=sys.stderr)
                continue
            text = f"{desc.caption}\nEntities: {', '.join(desc.entities)}\n{desc.text_in_image}"
            f.write(json.dumps({
                "file": img.name,
                "kind": "image",
                "pages": [{"page": 1, "text": text, "needs_ocr": False}],
                "caption": desc.caption,
                "entities": desc.entities,
            }) + "\n")
            if i % 5 == 0 or i == len(images):
                print(f"  [{i}/{len(images)}] {img.name}", file=sys.stderr)
    print(f"Output: {out}", file=sys.stderr)


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: 06_multimodal.py [videos|images]")
    mode = argv[0]
    if mode == "videos":
        transcribe_videos()
    elif mode == "images":
        caption_images()
    else:
        sys.exit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main(sys.argv[1:])
