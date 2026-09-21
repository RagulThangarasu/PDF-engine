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

# The same rules the deterministic checks work to, said in words the model can
# follow. Without them the AI reports exactly the false differences those checks
# were taught not to: a line that wraps in a different place, a page that
# paginates differently, a list number set tight against its word, a callout
# number one export draws and the other sets as text. What it is being asked is
# the same question the content check asks - is the content there or not.
_RULES = (
    "These two pages are the same page of the same manual, re-exported. They are "
    "TYPESET differently on purpose, so ignore every difference that is only "
    "typesetting:\n"
    "- text wrapping onto a different line, or a paragraph breaking in another place;\n"
    "- the same content sitting a little higher or lower, or the page being a "
    "different length;\n"
    "- a list number or bullet drawn tighter or looser against its text, or a "
    "different bullet glyph;\n"
    "- a number printed on a diagram in one and inside the picture in the other;\n"
    "- fonts, sizes and margins that only differ slightly.\n"
    "Report ONLY content that is genuinely present in one page and absent from the "
    "other, or genuinely changed: a picture, icon, table, row or block of text that "
    "is missing or extra, a picture that shows something else, a colour that changed."
)

DIFF_PROMPT = (
    "Here are two descriptions of the same page from two versions of a document.\n\n"
    "PRODUCTION (baseline):\n{prod}\n\n"
    "STAGING (candidate):\n{stage}\n\n"
    + _RULES + "\n"
    "Ignore differences that are only in how the two descriptions are worded - if "
    "both descriptions could be describing the same page, there is no difference. "
    "Answer in short bullet points, no more than 5, each naming what is missing, "
    "extra or changed. If there is no real difference, answer exactly: "
    "No visual differences."
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


def page_pairs(anchors: list[dict] | None, prod_pages: int, stage_pages: int) -> list[tuple[int, int]]:
    """Which Staging page to compare each Production page against, 1-based.

    Page 1 against page 1 is only right while the two documents paginate
    alike. They rarely do - this pair is 59 pages against 52 - and from the
    first inserted page onwards the AI was being shown two unrelated pages and
    asked what differed, which is a machine for producing false findings. The
    scroll-sync waypoints already say where the same content sits on both
    sides (matched headings, tables and paragraphs); the page mapping follows
    them, interpolating in between and falling back to page-for-page only when
    there are no waypoints at all.
    """
    placed = sorted(
        {(h["prod"]["page"], h["stage"]["page"]) for h in (anchors or []) if h.get("stage")}
    )
    if not placed:
        return [(i, i) for i in range(1, min(prod_pages, stage_pages) + 1)]
    out: list[tuple[int, int]] = []
    for prod in range(1, prod_pages + 1):
        before = [p for p in placed if p[0] <= prod]
        after = [p for p in placed if p[0] > prod]
        if before and after:
            (p0, s0), (p1, s1) = before[-1], after[0]
            stage = s0 + round((prod - p0) * (s1 - s0) / max(1, p1 - p0))
        elif before:
            p0, s0 = before[-1]
            stage = s0 + (prod - p0)
        else:
            p1, s1 = after[0]
            stage = s1 - (p1 - prod)
        if 1 <= stage <= stage_pages:
            out.append((prod, stage))
    return out


def review_pages(pdfview_dir: str, page_count: int, progress_cb=None,
                 pairs: list[tuple[int, int]] | None = None) -> list[dict]:
    """One entry per page pair the model found something to say about:
    [{"page": n, "note": text}]. A page whose images are missing, whose
    request errors, or that comes back reporting no difference is skipped -
    never raised, so one bad page cannot fail the whole review.

    `pairs` says which Staging page each Production page is compared against
    (see `page_pairs`); without it the two are paired page for page, which is
    only right when both documents paginate alike."""
    pairs = pairs or [(i, i) for i in range(1, page_count + 1)]
    out: list[dict] = []
    for n, (i, j) in enumerate(pairs, start=1):
        prod_image = os.path.join(pdfview_dir, f"prod_p{i}.png")
        stage_image = os.path.join(pdfview_dir, f"stage_p{j}.png")
        if not (os.path.isfile(prod_image) and os.path.isfile(stage_image)):
            continue
        if progress_cb:
            try:
                progress_cb(n, len(pairs))
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
            # Both page numbers: a reader checking the note needs to know
            # which Staging page it was actually compared against, which is
            # not "the same number" once the two paginate differently.
            out.append({"page": i, "stage_page": j, "note": note})
    return out
