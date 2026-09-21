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

The captions alone were not enough to review a page properly. A caption is a
lossy summary written by a small model: asked what differs between two of them,
the text model could only speak in generalities, and filled the gaps by
inventing plausible-sounding differences. So each page is ALSO measured - how
many images and at what size, how many shaded panels and in what colour, how
many tables and of what shape, how many links, how much text - straight off the
PDF, with no model involved. Those measurements are facts, and the model is
asked to explain and categorise the gaps between two sets of them rather than
to recall what it saw. What it reports is grouped the way a reviewer checks a
page: content, pictures and icons, tables, links, and the UI/layout of the page
itself.
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
    "- the same picture drawn as vector lines in one and embedded as an image in "
    "the other, or a picture a few points bigger or smaller;\n"
    "- a small difference in the icon count, which only reflects how the two "
    "pages group their drawing;\n"
    "- a word count that differs by a little, when the same topics are covered;\n"
    "- fonts, sizes and margins that only differ slightly.\n"
    "Report ONLY content that is genuinely present in one page and absent from the "
    "other, or genuinely changed: a picture, icon, table, row or block of text that "
    "is missing or extra, a picture that shows something else, a colour that changed."
)

DIFF_PROMPT = (
    "Two versions of the same page of a product manual are described below. For "
    "each side you get MEASUREMENTS taken from the PDF itself (these are facts) "
    "and a DESCRIPTION written by a vision model (this may be wrong or vague - "
    "trust the measurements over it).\n\n"
    "=== PRODUCTION (baseline), page {prod_page} ===\n{prod_facts}\n"
    "Description: {prod}\n\n"
    "=== STAGING (candidate), page {stage_page} ===\n{stage_facts}\n"
    "Description: {stage}\n\n"
    + _RULES + "\n\n"
    "Report every gap you can justify from the measurements, grouped under these "
    "headings, and use only the headings that have something under them:\n"
    "CONTENT: text, headings, notes or warnings present on one page and not the other.\n"
    "IMAGES & ICONS: a picture or icon missing, extra, resized, or showing something else.\n"
    "TABLES: a table missing, extra, or with a different number of rows or columns.\n"
    "LINKS: a hyperlink missing, extra, or pointing somewhere else.\n"
    "UI & LAYOUT: shaded note panels, coloured badges, callout styling, columns - "
    "the visual furniture of the page rather than its words.\n\n"
    "One short bullet per gap, naming the specific thing and which side has it. "
    "At most 10 bullets, and never drop a measured gap in favour of a guess. "
    "Do not repeat a measurement back without saying what it "
    "means. If nothing in the measurements shows a real gap, answer exactly: "
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


_FACT_TEXT_CHARS = 700  # kept for callers; the inventory owns the real limit


def page_facts(doc, page_index: int) -> dict:
    """The page's measured inventory - see `pdfval.validators.inventory`, which
    owns the measuring so the AI pass and the deterministic sweep can never
    disagree about what is on a page."""
    from pdfval.validators.inventory import page_inventory

    return page_inventory(doc, page_index)


def _facts_block(facts: dict) -> str:
    """The measurements as a few compact lines a small model can actually read."""
    def sizes(pairs):
        return ", ".join(f"{w}x{h}pt" for w, h in pairs) if pairs else "none"

    from pdfval.validators.inventory import _hex

    return (
        f"- text: {facts['words']} words on {facts['lines']} lines\n"
        f"- pictures ({len(facts['images'])}): {sizes(facts['images'])}\n"
        f"- small icons: {facts.get('icons', 0)}\n"
        f"- shaded panels ({len(facts.get('panels', []))}): "
        f"{', '.join(_hex(c) for c in facts.get('panels', [])) if facts.get('panels') else 'none'}\n"
        f"- tables ({len(facts['tables'])}): "
        f"{', '.join(f'{r} rows x {c} cols' for r, c in facts['tables']) if facts['tables'] else 'none'}\n"
        f"- hyperlinks ({len(facts['links'])}): "
        f"{', '.join(facts['links'][:6]) if facts['links'] else 'none'}\n"
        f"- text on the page: {facts['text']}"
    )


def measured_gaps(prod_facts: dict | None, stage_facts: dict | None) -> list[dict]:
    """The deterministic gaps between two pages, each with its place on the
    page. The model explains them; the boxes come from here, not from the
    model - a small local model cannot be trusted with coordinates, and it
    does not have to be when the measuring already knows them."""
    if not (prod_facts and stage_facts):
        return []
    try:
        from pdfval.validators.inventory import compare_pages_boxed

        return compare_pages_boxed(prod_facts, stage_facts)
    except Exception:
        return []


def _diff(prod_caption: str, stage_caption: str,
          prod_facts: dict | None = None, stage_facts: dict | None = None) -> str:
    empty = {"page": 0, "lines": 0, "words": 0, "text": "(not measured)",
             "images": [], "icons": 0, "panels": [], "tables": [], "links": [], "fonts": []}
    pf, sf = prod_facts or empty, stage_facts or empty
    # The gaps the deterministic sweep already found between these two pages,
    # handed over as a starting point. The sweep knows THAT something differs;
    # the model's job is to say what it means and whether it matters.
    gaps = measured_gaps(pf, sf)
    measured = ("\nMeasured gaps between these two pages:\n"
                + "\n".join(f"- {g['text']}" for g in gaps[:12]) if gaps else
                "\nMeasured gaps between these two pages: none found by measurement.")
    return _generate({
        "model": DIFF_MODEL,
        "prompt": DIFF_PROMPT.format(
            prod=prod_caption or "(no caption)", stage=stage_caption or "(no caption)",
            prod_facts=_facts_block(pf), stage_facts=_facts_block(sf) + measured,
            prod_page=pf["page"], stage_page=sf["page"],
        ),
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


def _facts_for(doc, page_index: int) -> dict | None:
    """`page_facts` for one side, or None when the document was not supplied or
    the page cannot be read - the review then falls back to captions alone."""
    if doc is None:
        return None
    try:
        return page_facts(doc, page_index)
    except Exception:
        return None


def review_pages(pdfview_dir: str, page_count: int, progress_cb=None,
                 pairs: list[tuple[int, int]] | None = None,
                 expected=None, actual=None) -> list[dict]:
    """One entry per page pair the model found something to say about:
    [{"page": n, "note": text}]. A page whose images are missing, whose
    request errors, or that comes back reporting no difference is skipped -
    never raised, so one bad page cannot fail the whole review.

    `pairs` says which Staging page each Production page is compared against
    (see `page_pairs`); without it the two are paired page for page, which is
    only right when both documents paginate alike.

    `expected` / `actual` are the two open PDFs. Given them, each page pair is
    MEASURED as well as captioned (see `page_facts`) and the model compares
    facts instead of two summaries - which is the difference between "these
    descriptions read differently" and "Production has two shaded note panels
    here, Staging has none"."""
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
            prod_facts = _facts_for(expected, i - 1)
            stage_facts = _facts_for(actual, j - 1)
            # With measurements in hand the captions no longer have to carry
            # the review on their own, so a page whose caption came back empty
            # is still worth comparing - it is only when there is NOTHING to go
            # on that the page is skipped.
            thin = (len(prod_caption) < _MIN_USABLE_CAPTION
                    or len(stage_caption) < _MIN_USABLE_CAPTION)
            if thin and not (prod_facts and stage_facts):
                continue
            note = _diff(prod_caption, stage_caption, prod_facts, stage_facts)
            gaps = measured_gaps(prod_facts, stage_facts)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            continue
        if note and _NO_DIFF not in note.lower():
            # Both page numbers: a reader checking the note needs to know
            # which Staging page it was actually compared against, which is
            # not "the same number" once the two paginate differently.
            out.append({
                "page": i, "stage_page": j, "note": note,
                # Where on each page the measured gaps are, so the viewer can
                # box what the note is talking about instead of leaving the
                # reader to find it.
                "prod_boxes": [{"page": i, "bbox": g["bbox"]}
                               for g in gaps if g["side"] == "prod" and g["bbox"]],
                "stage_boxes": [{"page": j, "bbox": g["bbox"]}
                                for g in gaps if g["side"] == "stage" and g["bbox"]],
                "gaps": [g["text"] for g in gaps],
            })
    return out
