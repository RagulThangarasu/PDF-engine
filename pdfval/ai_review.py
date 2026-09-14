"""Optional, additive AI visual review.

A free, open-source, locally-run AI (via Ollama - https://ollama.com) looks at
each matched Production/Staging page and describes what differs, in its own
words. This runs ALONGSIDE the deterministic checks, never in place of them,
and only when a user explicitly opts in - nothing here is imported by the main
comparison pipeline, and a failure here (Ollama not running, a model missing,
a single page erroring) never fails the run.

Two calls per page, not one: a small local vision model (moondream) describes
ONE image reliably, but a two-image "spot the difference" prompt in a single
call was tried first and came back empty (the model has no real multi-image
reasoning) - so each page is captioned on its own, and a text model already
installed (llama3.2) compares the two captions instead.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("PDFVAL_OLLAMA_URL", "http://localhost:11434")
VISION_MODEL = os.environ.get("PDFVAL_AI_MODEL", "moondream")
DIFF_MODEL = os.environ.get("PDFVAL_AI_DIFF_MODEL", "llama3.2:3b")
REQUEST_TIMEOUT = 60  # seconds per call - a CPU-only local model is slow

CAPTION_PROMPT = (
    "Describe exactly what is on this document page: any heading, body text topic, "
    "tables, pictures or diagrams, icons and their colours, list markers, and overall "
    "layout. Be specific and factual, in 3-6 short sentences."
)

DIFF_PROMPT = (
    "Here are two descriptions of the same page from two versions of a document.\n\n"
    "PRODUCTION (baseline):\n{prod}\n\n"
    "STAGING (candidate):\n{stage}\n\n"
    "List only the VISUAL differences a reader would notice between them - a missing or "
    "extra picture, an icon or colour that changed, something moved, resized or "
    "misaligned, a missing table or list marker. Ignore differences that are only in how "
    "the description is worded. Answer in short bullet points, no more than 5. If there "
    "is no real difference, answer exactly: No visual differences."
)
_NO_DIFF = "no visual differences"


def available(timeout: float = 3.0) -> bool:
    """True when Ollama is reachable and both configured models are pulled."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as resp:
            tags = json.loads(resp.read())
    except Exception:
        return False
    names = {(m.get("name") or "").split(":")[0] for m in tags.get("models", [])}
    return VISION_MODEL.split(":")[0] in names and DIFF_MODEL.split(":")[0] in names


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


_CAPTION_IMAGE_WIDTH = 384  # pt: moondream's own PDF-page renders (2000px+ wide)
# came back empty or garbled in testing - this is close to the model's own
# native input resolution and gave the most coherent results.
_MIN_USABLE_CAPTION = 15  # characters: shorter than this is empty or garbage, not a real caption


def _b64_resized(path: str) -> str:
    """The page image, downscaled to the vision model's own working
    resolution, base64-encoded. A full-resolution page render is far larger
    than what a small local vision model expects and comes back empty or as
    unrelated noise - see the module docstring."""
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        if img.width > _CAPTION_IMAGE_WIDTH:
            h = round(_CAPTION_IMAGE_WIDTH * img.height / img.width)
            img = img.resize((_CAPTION_IMAGE_WIDTH, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return _b64(path)


def _generate(payload: dict) -> str:
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read())
    return (data.get("response") or "").strip()


def _caption(image_path: str) -> str:
    return _generate({
        "model": VISION_MODEL, "prompt": CAPTION_PROMPT, "images": [_b64_resized(image_path)], "stream": False,
    })


def _diff(prod_caption: str, stage_caption: str) -> str:
    return _generate({
        "model": DIFF_MODEL,
        "prompt": DIFF_PROMPT.format(prod=prod_caption or "(no caption)", stage=stage_caption or "(no caption)"),
        "stream": False,
    })


def review_pages(pdfview_dir: str, page_count: int, progress_cb=None) -> list[dict]:
    """One entry per page pair the model found something to say about:
    [{"page": n, "note": text}]. A page whose images are missing, whose
    request errors, or that comes back reporting no difference is skipped -
    never raised, so one bad page cannot fail the whole review."""
    out: list[dict] = []
    for i in range(1, page_count + 1):
        prod_image = os.path.join(pdfview_dir, f"prod_p{i}.png")
        stage_image = os.path.join(pdfview_dir, f"stage_p{i}.png")
        if not (os.path.isfile(prod_image) and os.path.isfile(stage_image)):
            continue
        if progress_cb:
            try:
                progress_cb(i, page_count)
            except Exception:
                pass
        try:
            prod_caption = _caption(prod_image)
            stage_caption = _caption(stage_image)
            # A caption this short is empty or garbled, not real information -
            # feeding it to the diff step anyway does not fail cleanly, it
            # invents plausible-sounding differences from nothing, which is
            # worse than saying nothing here.
            if len(prod_caption) < _MIN_USABLE_CAPTION or len(stage_caption) < _MIN_USABLE_CAPTION:
                continue
            note = _diff(prod_caption, stage_caption)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            continue
        if note and _NO_DIFF not in note.lower():
            out.append({"page": i, "note": note})
    return out
