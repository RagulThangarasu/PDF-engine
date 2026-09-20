"""The data behind pdf.html: Production (the baseline) and Staging whole, side by
side, every issue boxed where it is and listed in the left nav.

`pdfval.validators.chapter` finds the differences; this places them. Each issue
carries:

  * its category - Content, Images, Hyperlinks, Lists, Bold, Tables - the same
    one report.json files it under, so every count on every page agrees;
  * a box on each document it concerns, in PDF points, which the viewer draws
    as a light red outline with a short black comment saying exactly what is
    wrong at that spot;
  * each side's own text with the difference marked, for the nav.

The page images are plain renders: boxes and comments are drawn over them in
the browser, so they stay on the text at any zoom and can be switched off.

This replaces report.html, sections.html and chapters.html. Those described the
same two documents three ways, from two engines, with three different counts -
hyperlinks passed in one and failed in another.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import fitz

from pdfval.report.chapters import _TYPE_HELP, _range, _side_content
from pdfval.validators.chapter import CATEGORIES, category_of, content_anchors, table_anchors
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import heading_at, match_toc_entries

SUBDIR = "pdfview"
# Every page is rendered to the same pixel width, whatever its size in points.
# A fixed zoom gave an A5 Production page 630px against an A4 Staging page's
# 893px; both are shown at the pane's width, so Production was upscaled and
# looked pixelated. 2400px covers a half-screen pane on a 2x display with room
# to zoom in. PNG, not JPEG: lossless, so text edges stay as the PDF prints them.
RENDER_WIDTH_PX = 2400
MAX_ZOOM = 8.0
MAX_PAGES_PER_DOCUMENT = 400   # a runaway document cannot fill the run directory
MAX_BOXES_PER_ISSUE = 40
COMMENT_MAX = 90               # characters in the black comment beside a box

_CATEGORY_KEYS = [c["key"] for c in CATEGORIES]
_CATEGORY_LABEL = {c["key"]: c["label"] for c in CATEGORIES}
_BOXES_BEFORE_LIST = 3         # more word boxes than this on a side: one box, changes listed

# Noise: kinds of difference a reviewer may legitimately not want to read
# through on a given pass - the same idea as a text diff tool's "ignore
# whitespace" switch. Only ever HIDDEN, never dropped: the run still finds
# them, the report still carries them, and the toolbar says how many are out
# of sight, so ticking a box can never make a document quietly look cleaner
# than it is. Deliberately conservative - anything that changes what the
# document SAYS (wording, a missing row, a symbol, a picture's size) is not
# offered here at all, however cosmetic it might look in a particular run.
NOISE_GROUPS = [
    {"key": "space", "label": "Whitespace",
     "help": "A space added, dropped or doubled where the wording itself is identical."},
    {"key": "punctuation", "label": "Punctuation",
     "help": "A comma, full stop or bracket that differs where the wording does not."},
    {"key": "order", "label": "Reordering",
     "help": "The same content, printed in a different order - text out of sequence, "
             "a picture that moved under another heading."},
    {"key": "styling", "label": "Styling",
     "help": "Bold, italic, underline, shading, line spacing, marker size - how text is set, not what it says."},
    {"key": "layout", "label": "Layout",
     "help": "Indent, alignment and table shape - where content sits, not what it says."},
]
_NOISE_OF = {
    # Spaces, punctuation and quote marks are real issues (user rule), never
    # folded away as noise; nor is a merged cell Staging splits.
    "moved": "order", "table-cell-sequence": "order", "figure-wrong-section": "order",
    "bold-missing": "styling", "bold-added": "styling",
    "underline-missing": "styling", "underline-added": "styling",
    "italic-missing": "styling", "italic-added": "styling",
    "shading": "styling", "marker-size": "styling", "line-spacing": "styling", "icon-colour": "styling",
    "list-indent": "layout", "figure-alignment": "layout",
    "list-label-layout": "layout", "list-marker-glyph": "layout",
    "table-cell-layout": "layout", "table-shape": "layout",
    "table-header-repeat": "layout",
}

# Drawn orange. Everything else - missing or wrong content, a missing table or
# row, a/b/c instead of 1/2/3, a lost bullet, a missing picture or link - is
# red. Bold is red too, not orange: `confirm_bold` only ever keeps a
# "bold-missing"/"bold-added" candidate once it has measured the two sides'
# printed stroke weight and confirmed Staging's copy genuinely reads lighter
# or heavier, so by the time one reaches here it is exactly as real a defect
# as a wrong word - not a styling nuance to merely flag for review.
ORANGE_TYPES = {
    "list-indent", "table-merge", "table-columns", "table-cell-layout", "figure-alignment", "figure-size",
    "shading", "marker-size", "icon-changed",
    "underline-missing", "underline-added", "line-spacing", "glyph-render",
}

# An image or icon genuinely gone from one side - not resized, not moved, not
# rendering a little differently, but not there at all. Severity alone (4, the
# same tier as "figure oversized") would draw it in the ordinary, same-as-
# every-other-Images-finding blue; outright absence is content missing
# outright, same as a level-1 Content finding, and is drawn just as critical -
# thicker, opaque, and in red - so it never blends into a page full of routine
# size/alignment differences.
_ALWAYS_CRITICAL_TYPES = {"figure-missing", "figure-added", "icon-missing", "icon-added"}


# The content-validation label every text finding carries in the report - one
# of the seven kinds of content change a reviewer checks against Production.
CONTENT_LABELS = (
    "Added/removed text", "Changed words or numbers", "Changed paragraphs",
    "Page-level differences", "Tables/content changes", "Different reading order", "Missing text",
)
_PARAGRAPH_CHANGE_WORDS = 8  # more changed words than this in one finding: the paragraph was rewritten


def content_label(diff: dict) -> str:
    kind = diff.get("type") or ""
    if kind in ("missing", "section-missing"):
        return "Tables/content changes" if diff.get("kind") == "table" else "Missing text"
    if kind in ("added", "section-added"):
        return "Tables/content changes" if diff.get("kind") == "table" else "Added/removed text"
    if kind == "moved":
        return "Different reading order"
    if kind.startswith("table-"):
        return "Tables/content changes"
    if kind.startswith("page-") or kind == "scanned-page":
        return "Page-level differences"
    if kind == "text":
        gone, extra = list(diff.get("gone") or []), list(diff.get("extra") or [])
        changed = sum(len(str(w).split()) for w in gone + extra)
        if changed > _PARAGRAPH_CHANGE_WORDS:
            return "Changed paragraphs"
        if gone and extra:
            return "Changed words or numbers"
        return "Missing text" if gone else "Added/removed text"
    if kind == "page-ref-in-staging":
        return "Added/removed text"
    if kind == "paragraph-gap":
        return "Changed paragraphs"
    if kind in ("punctuation", "symbol", "numbering", "text-encoding", "glyph-render", "text-space"):
        return "Changed words or numbers"
    return ""


def _render(doc: fitz.Document, output_dir: str, prefix: str) -> list[dict]:
    """Every page as an image; `w`/`h` are the page's size in points, which is
    what the boxes are measured in."""
    os.makedirs(os.path.join(output_dir, SUBDIR), exist_ok=True)
    pages = []
    for index in range(min(doc.page_count, MAX_PAGES_PER_DOCUMENT)):
        page = doc[index]
        name = f"{prefix}_p{index + 1}.png"
        zoom = min(MAX_ZOOM, RENDER_WIDTH_PX / max(1.0, page.rect.width))
        try:
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).save(
                os.path.join(output_dir, SUBDIR, name)
            )
        except Exception:
            continue
        pages.append({"n": index + 1, "src": f"{SUBDIR}/{name}",
                      "w": page.rect.width, "h": page.rect.height})
    return pages


def _title(summary: str) -> str:
    """The issue's name, in the engine's own words ("Figure oversized in Staging")."""
    return summary.split(" — ")[0].rstrip(". ")


def _description(summary: str) -> str:
    """The summary without its name, which is shown on its own."""
    rest = summary.split(" — ", 1)[-1].strip()
    return rest[:1].upper() + rest[1:]


def _clip(text: str, limit: int = COMMENT_MAX) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _quoted(fragments: list[str]) -> str:
    shown = ", ".join(f"“{_clip(f, 40)}”" for f in fragments[:3])
    return shown + (" …" if len(fragments) > 3 else "")


def _comments(diff: dict, title: str, description: str) -> tuple[str, str]:
    """The black comment written beside each side's box: what is wrong at that
    exact spot, in a few words. Empty for a side the issue has no box on."""
    kind = diff.get("type")
    ops = diff.get("word_diff") or []
    gone = [op["text"] for op in ops if op["type"] == "del"]
    extra = [op["text"] for op in ops if op["type"] == "ins"]
    if gone or extra:
        prod = f"Missing in Staging: {_quoted(gone)}" if gone else "Staging adds words here"
        stage = f"Extra in Staging: {_quoted(extra)}" if extra else f"Missing here: {_quoted(gone)}"
        return _clip(prod), _clip(stage)
    if kind in ("missing", "added", "table-row-missing", "table-row-added"):
        # No counterpart element exists on the other side to box, so there is
        # nothing to write a DIFFERENT comment about there - both sides get
        # the exact same detailed line: what is missing/extra, quoted, so a
        # reviewer sees the same explanation whichever document is on screen.
        lost = kind in ("missing", "table-row-missing")
        src = diff.get("exp") if lost else diff.get("act")
        el_text = getattr(src, "text", "") if src is not None else ""
        quoted = _quoted([el_text or diff.get("detail") or ""])
        label = "Missing in Staging" if lost else "Extra in Staging"
        msg = _clip(f"{label}: {quoted}") if quoted else _clip(title)
        return msg, msg
    fixed = {
        "figure-missing": ("Picture missing in Staging", ""),
        "figure-added": ("", "Picture not in Production"),
        "bold-missing": ("Bold in Production", "Not bold in Staging"),
        "bold-added": ("Regular in Production", "Bold added in Staging"),
        "italic-missing": ("Italic in Production", "Not italic in Staging"),
        "italic-added": ("Upright in Production", "Italic added in Staging"),
        "link-missing": ("Link in Production", "Link missing in Staging"),
        "link-added": ("No link in Production", "Extra link in Staging"),
        "link-broken": ("", "Link does not work in Staging"),
        "list-marker-missing": ("List item in Production", "Bullet / number missing in Staging"),
        "list-marker-added": ("Plain text in Production", "Bullet / number added in Staging"),
        "table-as-text": ("Table in Production", "Table missing in Staging — rows printed as plain text"),
        "table-fill-missing": ("Background in Production", "Background missing in Staging"),
        "icon-missing": ("Icon in Production", "Icon missing in Staging"),
        "icon-added": ("No icon in Production", "Icon added in Staging"),
        "icon-colour": ("Icon colour in Production", "Icon colour changed in Staging"),
    }
    if kind in fixed:
        return fixed[kind]
    if kind == "shading":
        shading_titles = {
            "Background shading added in Staging": ("Plain page in Production", "Shaded box added in Staging"),
            "Background shading missing in Staging": ("Shaded box in Production", "Shading missing in Staging"),
            "Note styling missing in Staging": ("Icon note in Production", "Note styling missing in Staging"),
            "Extra content pulled into Staging's note box": ("Separate text in Production", "Pulled into note box in Staging"),
        }
        if title in shading_titles:
            return shading_titles[title]
        return (("Plain page in Production", "Shaded box added in Staging") if "added" in title
                else ("Shaded box in Production", "Shading missing in Staging"))
    if kind == "link-page-ref-dropped":
        # `detail` carries the exact phrase ("on page 32") - short and
        # specific about which side has it, instead of the generic
        # "<title>: <description>" fallback repeating the whole summary
        # sentence (already shown once, in full, right beside the box) and
        # then clipping it mid-word.
        ref = diff.get("detail") or "the page number"
        return f"Names “{ref}”", "Drops it — no page named here"
    if kind == "link-page-ref-inconsistent":
        return "", "Keeps “on page” here, unlike its sibling items"
    if kind == "link-target":
        # detail reads "Production: section: X · Staging: section: Y" - or, for
        # a link to an outside address, "Production: https://... · Staging: ...".
        # A URL is the one piece of a comment that must never be clipped short:
        # a path cut off mid-way ("…/software/display-pilot-2/spec.h…") reads as
        # a whole different, unreadable address rather than the real one, and
        # is exactly the detail a reviewer opened this comment to check. Each
        # side is clipped on its own, generously, instead of the assembled
        # sentence at a fixed length that a long URL alone can blow past.
        sides = (diff.get("detail") or "").split(" · ")
        if len(sides) == 2:
            where = [_clip(re.sub(r"^\s*section\s*:\s*", "", s.split(":", 1)[-1]).strip(), 200) for s in sides]
            both = f"Production links to “{where[0]}”; Staging links to “{where[1]}”"
            return both, both
    # "The figure beside “LCD monitor”: 24×87pt in Production, 30×105pt in
    # Staging." - what differs is the part after the figure's name.
    tail = description.split("”: ", 1)[-1] if "”: " in description else description
    text = _clip(f"{title}: {tail}")
    return text, text


def _boxes(el, marks: list | None = None) -> list[dict]:
    """[{page (1-based), bbox (points)}] - the marks when the issue has its own
    exact places (the spots on a picture, the words that lost their bold), the
    element's boxes otherwise."""
    places = marks if marks else (el.boxes if el is not None else [])
    return [{"page": p + 1, "bbox": [round(float(v), 2) for v in b]} for p, b in places]


# --- only what differs ----------------------------------------------------
#
# A reworded sentence is boxed on the WORDS that changed, on each page, each box
# with its own comment - the way a diff tool marks a line - not as the whole
# paragraph, which left the reader to find the one changed word by eye.

def _mark(page_index: int, rect, note: str) -> dict:
    r = fitz.Rect(rect)
    return {"page": page_index + 1, "bbox": [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)],
            "note": note}


# A "missing"/"added" finding has no element on one side to box - nothing was
# printed there. Left with no box at all, that side showed nothing when a
# reviewer had it open on its own: no pin, no comment, no way to tell the
# issue exists without switching to the other document first.
_MARKER_TYPES = {"missing", "added", "table-row-missing", "table-row-added", "figure-missing", "figure-added"}


_MARKER_DROP = 18.0  # pt below the anchor: past the heading line it points at


def _marker_box(doc: fitz.Document, point: tuple, note: str) -> dict | None:
    """A small pin at roughly where the missing/extra content would sit on
    this side, carrying the SAME comment as the side that has it - so the
    issue is visible, and reads the same, whichever document is on screen."""
    page_index, y = point
    try:
        rect = doc[page_index].rect
    except Exception:
        return None
    # In the MARGIN, below the anchor line. Drawn where the anchor itself
    # points, a 90pt-wide pin landed square on the section's heading, which
    # reads as "this heading is wrong" - the one thing it does not mean. A
    # thin tab clear of the text column says "the content belongs along here"
    # without marking any words as changed.
    x0 = rect.x0 + 8
    x1 = min(rect.x1 - 8, x0 + 12)
    y0 = min(max(rect.y0, y + _MARKER_DROP), max(rect.y0, rect.y1 - 18))
    return {"page": page_index + 1, "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y0 + 18, 2)],
            "note": note, "marker": True}


def _hits_in(doc: fitz.Document, el, phrase: str) -> list[tuple[int, "fitz.Rect"]]:
    """Every rect where `phrase` prints inside the element, in page order."""
    out: list[tuple[int, fitz.Rect]] = []
    for page_index, bbox in el.boxes:
        try:
            found = doc[page_index].search_for(phrase, clip=fitz.Rect(bbox) + (-3, -3, 3, 3))
        except Exception:
            continue
        out.extend((page_index, r) for r in found)
    return out


def _first_occurrence(hits: list[tuple[int, "fitz.Rect"]]) -> list[tuple[int, "fitz.Rect"]]:
    """The rects of the first match only: a phrase that wraps comes back as one
    rect per line, and those continue on the line below, starting further left."""
    if not hits:
        return []
    out = [hits[0]]
    for page_index, r in hits[1:]:
        prev_page, prev = out[-1]
        if page_index == prev_page and prev.y1 - 2 <= r.y0 <= prev.y1 + prev.height and r.x0 < prev.x0:
            out.append((page_index, r))
        else:
            break
    return out


def _locate(doc: fitz.Document, el, fragment: str, before: str) -> list[tuple[int, "fitz.Rect"]]:
    """Where `fragment` prints in the element - just after the words that come
    before it when those are known, so a word printed twice in the paragraph is
    found in the right place. Empty when it cannot be found."""
    fragment = " ".join(fragment.split())
    if not fragment:
        return []
    if before:
        occurrence = _first_occurrence(_hits_in(doc, el, f"{before} {fragment}"))
        if occurrence:
            page_index = occurrence[0][0]
            region = fitz.Rect(occurrence[0][1])
            for _, r in occurrence[1:]:
                region |= r
            try:
                inside = doc[page_index].search_for(fragment, clip=region + (-1, -1, 1, 1))
            except Exception:
                inside = []
            if inside:
                return [(page_index, inside[-1])]
    found = _first_occurrence(_hits_in(doc, el, fragment))
    if found or len(fragment.split()) > 3:
        return found
    # The page prints the words with other punctuation around them - "LAN."
    # in the compared text, "LAN on page 36." on the page once the page
    # reference is set aside: the words alone, still after their context.
    bare = fragment.strip(".,;:!?\"'“”‘’()")
    if bare and bare != fragment:
        return _locate(doc, el, bare, before)
    longest = max(fragment.split(), key=len)
    return _first_occurrence(_hits_in(doc, el, longest)) if len(longest) >= 4 else []


def _word_marks(diff: dict, a, b, expected: fitz.Document, actual: fitz.Document) -> tuple[list[dict], list[dict]]:
    """A box on every run of words that differs, on each side, with its comment.
    Where one side has no words at all (words dropped, words added) a thin mark
    is put at the spot they belong, right after the words before them."""
    ops = diff.get("word_diff") or []
    if not ops or a is None or b is None:
        return [], []
    prod: list[dict] = []
    stage: list[dict] = []
    before_p: list[str] = []
    before_s: list[str] = []

    def spot(doc, el, words):
        if not words:
            return None
        at = _locate(doc, el, words[-1], " ".join(words[:-1]))
        if not at:
            return None
        page_index, r = at[-1]
        # A thin insertion mark in the gap after the word, clear of the next one.
        return page_index, fitz.Rect(r.x1 + 0.5, r.y0, r.x1 + 2, r.y1)

    def spot_before(doc, el, j):
        """A thin mark just LEFT of the words that follow position `j` - where
        an insertion really goes when the words before it end the line above."""
        nxt = next((o for o in ops[j:] if o["type"] == "equal"), None)
        words = nxt["text"].split()[:3] if nxt else []
        if not words:
            return None
        at = _locate(doc, el, " ".join(words), "")
        if not at:
            return None
        page_index, r = at[0]
        return page_index, fitz.Rect(r.x0 - 2, r.y0, r.x0 - 0.5, r.y1)

    i = 0
    while i < len(ops):
        if ops[i]["type"] == "equal":
            words = ops[i]["text"].split()
            before_p, before_s = (before_p + words)[-3:], (before_s + words)[-3:]
            i += 1
            continue
        gone: list[str] = []
        added: list[str] = []
        while i < len(ops) and ops[i]["type"] != "equal":
            (gone if ops[i]["type"] == "del" else added).append(ops[i]["text"])
            i += 1
        gone_text, added_text = " ".join(gone), " ".join(added)
        # Whitespace or punctuation alone is no missing word: no box, no mark.
        has_word = lambda t: any(ch.isalnum() for ch in t)  # noqa: E731
        gone_text = gone_text if has_word(gone_text) else ""
        added_text = added_text if has_word(added_text) else ""
        if not gone_text and not added_text:
            continue
        if gone_text and added_text and gone_text.replace(" ", "") == added_text.replace(" ", ""):
            lost = gone_text.count(" ") > added_text.count(" ")
            note_p = (f"Space {'missing' if lost else 'added'} in Staging: “{_clip(gone_text, 40)}” → "
                      f"“{_clip(added_text, 40)}”")
            note_s = (f"Space {'missing' if lost else 'added'} here — Production prints “{_clip(gone_text, 40)}”")
        elif gone_text and added_text:
            note_p = f"“{_clip(gone_text, 40)}” changed to “{_clip(added_text, 40)}” in Staging"
            note_s = f"“{_clip(added_text, 40)}” — Production has “{_clip(gone_text, 40)}”"
        elif gone_text:
            note_p = f"“{_clip(gone_text, 50)}” missing in Staging"
            note_s = f"Missing here: “{_clip(gone_text, 50)}”"
        else:
            note_p = f"Staging adds “{_clip(added_text, 50)}” here"
            note_s = f"“{_clip(added_text, 50)}” extra in Staging"
        if gone_text:
            prod += [_mark(p, r, note_p) for p, r in _locate(expected, a, gone_text, " ".join(before_p))]
        else:
            at = spot_before(expected, a, i) or spot(expected, a, before_p)
            if at:
                prod.append(_mark(*at, note_p))
        if added_text:
            stage += [_mark(p, r, note_s) for p, r in _locate(actual, b, added_text, " ".join(before_s))]
        else:
            at = spot(actual, b, before_s)
            if at:
                stage.append(_mark(*at, note_s))
        before_p = (before_p + gone_text.split())[-3:]
        before_s = (before_s + added_text.split())[-3:]
    return prod, stage


def _numbering_marks(diff: dict, expected: fitz.Document, actual: fitz.Document) -> tuple[list[dict], list[dict]]:
    """A box on each list marker that changed style - the "1." and the "a." -
    rather than on the whole list."""
    items = diff.get("items") or []
    if not items:
        return [], []
    # One comment for the list, on its first marker on each page: a note per
    # marker ("1." -> "a.", "2." -> "b.", ...) stacked four deep over the table.
    note = (f"Numbering: Production {', '.join(i[0] for i in items)} → "
            f"Staging {', '.join(i[1] for i in items)}")
    prod: list[dict] = []
    stage: list[dict] = []
    for _, _, key, exp_el, act_el in items:
        phrase = " ".join(key.split()[:4])
        for doc, el, out in ((expected, exp_el, prod), (actual, act_el, stage)):
            hits = _hits_in(doc, el, phrase) if el is not None else []
            if hits:
                page_index, r = hits[0]
                first_on_page = all(m["page"] != page_index + 1 for m in out)
                mark = _mark(page_index, fitz.Rect(r.x0 - 24, r.y0, r.x0 - 1, r.y1), note if first_on_page else "")
                out.append(mark)
    return prod, stage


def _y_share(doc: fitz.Document, page_index: int, y: float) -> float:
    height = doc[page_index].rect.height or 1.0
    return max(0.0, min(1.0, y / height))


def _proportional_page(page: int, own_pages: list[int], other_pages: list[int]) -> int | None:
    """Roughly where `page` (a position within `own_pages`, this chapter's own
    page range on one side) falls on the OTHER side's copy of the same
    chapter - the same technique the figure reconciliation pass already uses
    for a figure with no matched caption, applied here as a fallback anchor
    for any "missing"/"added" finding with nothing to box on the other side
    at all. Without it, that finding's pin falls all the way back to the
    section's own heading, which can be most of a page away from where the
    reader would actually look - a diagram's own dimension labels, missing on
    one side, land near the section's opening paragraph instead of near the
    diagram they belong to.
    """
    if not own_pages or not other_pages or page not in own_pages:
        return None
    frac = own_pages.index(page) / max(1, len(own_pages) - 1)
    return other_pages[round(frac * (len(other_pages) - 1))]


def _proportional_anchor(
    own_doc: fitz.Document, page: int, y: float,
    own_pages: list[int], other_pages: list[int], other_doc: fitz.Document,
) -> tuple[int, float] | None:
    """`(page, y)` in POINTS on the OTHER side for an element with no
    counterpart there at all: the page from `_proportional_page`, at the SAME
    relative height down that page the element sits at on its own - not the
    top of the page, which reads as "the section heading" regardless of where
    in the section the missing content actually was."""
    prop_page = _proportional_page(page, own_pages, other_pages)
    if prop_page is None:
        return None
    share = _y_share(own_doc, page, y)
    return (prop_page, share * (other_doc[prop_page].rect.height or 1.0))


def _order_anchors(out: list[dict]) -> list[dict]:
    """Sort a set of waypoints into Production's reading order and drop any
    `stage` side that cannot be part of one forward-running sequence - a
    heading or table Staging prints earlier than one already placed cannot be
    a waypoint, or scrolling forward in Production would jump Staging back.

    Which ones to drop is the whole question. Scanning forward and keeping
    whatever comes next let ONE bad match ruin everything after it: these
    manuals repeat near-identical "Item | Function | Range" tables under every
    menu, one of them matched a table seven pages too far down, and every
    correct waypoint for the eleven pages that followed was then "backwards"
    and thrown away - leaving exactly the stretch where the two panes drift
    apart with nothing to hold them together. So the sequence kept is the
    LONGEST run of waypoints that does move forward on both sides, which
    drops the single odd one out instead of the many that disagree with it.
    """
    out.sort(key=lambda h: (h["prod"]["page"], h["prod"]["y"]))
    placed = [i for i, h in enumerate(out) if h["stage"] is not None]
    keys = [(out[i]["stage"]["page"], out[i]["stage"]["y"]) for i in placed]
    # Patience sorting: `tails[k]` is the smallest ending key of an increasing
    # run of length k+1, `back` remembers each item's predecessor in its run.
    tails: list[int] = []
    back: list[int] = [-1] * len(keys)
    for n, key in enumerate(keys):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[tails[mid]] < key:
                lo = mid + 1
            else:
                hi = mid
        back[n] = tails[lo - 1] if lo else -1
        if lo == len(tails):
            tails.append(n)
        else:
            tails[lo] = n
    kept: set[int] = set()
    n = tails[-1] if tails else -1
    while n >= 0:
        kept.add(placed[n])
        n = back[n]
    for i, h in enumerate(out):
        if h["stage"] is not None and i not in kept:
            h["stage"] = None  # out of order - a Production-only waypoint instead of a wrong one
    return out


def _anchors(expected: fitz.Document, actual: fitz.Document, exp_entries, act_entries) -> list[dict]:
    """Production's own table of contents, whole - the nav lists every heading
    the baseline has, not just the ones Staging also has, and clicking one
    always moves Production there. `stage` is filled in only where Staging
    has a usable match for it, in step with the matches before it: the JS
    side interpolates Staging's position from its neighbours when a heading
    is Production-only, the same way it already does for a one-sided issue."""
    out: list[dict] = []
    for m in match_toc_entries(exp_entries, act_entries):
        if m.expected_index is None:
            continue
        e = exp_entries[m.expected_index]
        if not (0 <= e.page < expected.page_count):
            continue
        stage = None
        if m.actual_index is not None:
            a = act_entries[m.actual_index]
            if 0 <= a.page < actual.page_count:
                stage = {"page": a.page + 1, "y": _y_share(actual, a.page, a.y)}
        out.append({
            "title": e.title, "level": e.level,
            "prod": {"page": e.page + 1, "y": _y_share(expected, e.page, e.y)},
            "stage": stage,
        })
    return _order_anchors(out)


# A dense area - a heading with several paragraph edits, a run of bullets
# whose markers all changed, a page of similar list items, a spec table whose
# cells split into a different number of lines - can produce several issues
# under one heading, each boxed on the page and lined to its own comment.
# Past a handful, those lines cross every other one's and the page stops
# being readable, even though each individual finding is real. A couple in
# one section stays as-is (specific enough to be worth its own comment); past
# that, the rest fold into ONE comment per (section, CATEGORY) - not further
# split by the finding's own type, so a heading with a dropped paragraph, a
# renumbered bullet and an out-of-line indent next to it reads as one cluster
# of content problems, not three separate comments pointing at the same
# handful of lines - listing each one as its own bullet and keeping every
# one's own boxes lit up, instead of drawing a separate, overlapping line for
# each.
_CONSOLIDATE_MIN = 2  # this many (or more) in one section fold into one card


def _bounding_boxes(boxes: list[dict]) -> list[dict]:
    """One box per page spanning every box on it, instead of one per issue.

    A consolidated card unions every member's own boxes - for a genuine
    pile-up that is dozens of small boxes again, each with its own line back
    to the one card, which is the exact clutter consolidating the CARDS was
    meant to remove; it just moved from the sidebar onto the page. This is
    the same "box the paragraph once" rule `build_issue_report` already
    applies within a single diff's own word-marks, extended to a merged
    card's boxes too.
    """
    by_page: dict[int, list[dict]] = {}
    for b in boxes:
        by_page.setdefault(b["page"], []).append(b)
    out = []
    for page, page_boxes in by_page.items():
        # Only boxes that touch or nearly touch share one: a paragraph's
        # several marks become one box, but five captions scattered around a
        # page of pictures keep their own - one box around all of them
        # outlined half the page and pointed at none of them.
        clusters: list[list[float]] = []
        pairs: list[set] = []
        for b in sorted(page_boxes, key=lambda b: (b["bbox"][1], b["bbox"][0])):
            x0, y0, x1, y1 = b["bbox"]
            for c in clusters:
                # Same LINE only. Joining boxes that merely sit near each other
                # vertically swallowed the lines between them: three marked
                # words on three bullets became one block over the whole note,
                # which reads as "all of this differs" when only those words
                # do. Two marks join when they overlap on the page's y axis -
                # the same line of text - and are close across it.
                overlap = min(y1, c[3]) - max(y0, c[1])
                if overlap > _LINE_OVERLAP * min(y1 - y0, c[3] - c[1]) \
                        and x0 <= c[2] + _BOX_JOIN_GAP and x1 >= c[0] - _BOX_JOIN_GAP:
                    c[:] = [min(c[0], x0), min(c[1], y0), max(c[2], x1), max(c[3], y1)]
                    pairs[clusters.index(c)].add(b.get("pair"))
                    break
            else:
                clusters.append([x0, y0, x1, y1])
                pairs.append({b.get("pair")})
        for c, owners in zip(clusters, pairs):
            box = {"page": page, "bbox": c}
            known = [p for p in owners if p is not None]
            if known:
                box["pair"] = min(known)  # the earliest finding merged into it
            out.append(box)
    return out


_BOX_JOIN_GAP = 6.0  # pt: boxes this close are one place on the page
# Two boxes are on the same line when they share this much of the shorter
# one's height; below it they are separate lines and stay separate boxes.
_LINE_OVERLAP = 0.5


def _consolidate_issues(issues: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for issue in issues:
        # A shaded NOTE callout (or its missing styling) is boxed, not
        # tinted, in pdf.html - specifically BECAUSE its own type is
        # "shading"/"note-label" (see pdf_template.html). Folding it into a
        # same-category, different-type "formatting" card would hand it a
        # type that is no longer either of those, so it would quietly lose
        # that treatment and get tinted like ordinary formatting text - kept
        # to its own (section, type) key, like every category did before
        # merge-by-category-alone, so it never shares a card with anything
        # that changes its type.
        key = (issue["section"], issue["type"]) if issue["type"] in ("shading", "note-label") \
            else (issue["category"], issue["section"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(issue)

    out: list[dict] = []
    for key in order:
        members = groups[key]
        if len(members) < _CONSOLIDATE_MIN:
            out.extend(members)
            continue
        merged = dict(members[0])
        merged["occurrences"] = sum(m["occurrences"] for m in members)
        # One critical member makes the whole card critical: folded in behind a
        # routine member's wording it would be drawn as the routine one, and the
        # finding a reader must not miss is exactly the one that disappears.
        if any(m.get("critical") for m in members):
            merged["critical"], merged["minor"], merged["noise"] = True, False, ""
        prod_boxes = [b for m in members for b in m["prod_boxes"]]
        stage_boxes = [b for m in members for b in m["stage_boxes"]]
        # Pictures keep their own boxes: one box around several figures would
        # outline the text between them too.
        if merged["category"] != "images":
            if len(prod_boxes) > _BOXES_BEFORE_LIST:
                prod_boxes = _bounding_boxes(prod_boxes)
            if len(stage_boxes) > _BOXES_BEFORE_LIST:
                stage_boxes = _bounding_boxes(stage_boxes)
        merged["prod_boxes"] = prod_boxes[:MAX_BOXES_PER_ISSUE]
        merged["stage_boxes"] = stage_boxes[:MAX_BOXES_PER_ISSUE]
        # A mixed-type group (a dropped paragraph, a renumbered bullet, an
        # indent, say) is not any single one of those - the first member's
        # own wording would misname the rest, so it takes a generic title
        # instead. A same-type group (the common case) keeps the specific
        # wording as before.
        kinds = {m["type"] for m in members}
        title = merged["title"] if len(kinds) == 1 else f"{_CATEGORY_LABEL.get(merged['category'], 'Content')} differences"
        merged["title"] = f"{title} ({len(members)} places)"
        if len(kinds) > 1:
            merged["noise"], merged["minor"] = "", False
        # ONE message, not one per place: the per-member text is near-identical
        # boilerplate once merged ("Figure oversized in Staging: WxH vs WxH",
        # N times over with only the numbers changing) - the box the reader is
        # already looking at says WHICH one, so the comment says how many,
        # and - when the group mixes kinds - how many of each, once each, not
        # N times over.
        if len(kinds) == 1:
            merged["description"] = f"{merged['description']} ({len(members)} places - see each box)."
        else:
            label_of = {m["type"]: m["title"] for m in members}
            counts = Counter(m["type"] for m in members)
            merged["description"] = ", ".join(f"{n} \u00d7 {label_of[t]}" for t, n in counts.most_common()) + "."
        # Each place keeps its own content label: a card holding a missing
        # sentence and a changed number is both, and says how many of each.
        labels = Counter(m.get("content_label") for m in members if m.get("content_label"))
        merged["content_label"] = " · ".join(labels)
        if set(labels) == {"Tables/content changes"} and len(kinds) > 1:
            merged["title"] = f"Table issues ({len(members)} places)"
        if len(labels) > 1 and len(kinds) == 1:
            merged["description"] = ", ".join(f"{n} \u00d7 {lab}" for lab, n in labels.most_common()) + "."
        elif len(kinds) > 1:
            # Named finding by finding ("2 × Table head missing on the continued
            # page"), never by the generic label they share: the label says
            # which check ran, the title says what is wrong.
            label_of = {m["type"]: m["title"] for m in members}
            counts = Counter(m["type"] for m in members)
            merged["description"] = ", ".join(f"{n} \u00d7 {label_of[t]}" for t, n in counts.most_common()) + "."
        merged["changes"] = []
        merged["comment_prod"] = merged["comment_stage"] = ""
        out.append(merged)
    return out


def _page_level_issues(expected: fitz.Document, actual: fitz.Document, start: int,
                       chapter_rows: list[dict]) -> list[dict]:
    """Differences of the documents as a whole - page count, pages only one
    side has, pages that are scanned images with no text layer."""
    from pdfval import ocr

    def page_box(doc: fitz.Document, index: int) -> list[dict]:
        r = doc[index].rect
        return [{"page": index + 1, "bbox": [0, 0, round(r.width, 2), round(r.height, 2)], "note": ""}]

    found: list[tuple] = []  # (type, title, description, prod_boxes, stage_boxes)
    n_exp, n_act = expected.page_count, actual.page_count
    # One finding, not one per page: a denser layout ends earlier without any
    # page going missing - content itself is compared heading by heading, and
    # anything truly lost is reported there.
    if n_exp != n_act:
        diff = n_exp - n_act
        found.append(("page-count", "Page count differs",
                      f"Production has {n_exp} pages, Staging has {n_act} "
                      f"({abs(diff)} {'fewer' if diff > 0 else 'more'}). Content is compared heading by heading; "
                      f"anything actually missing is listed under its heading.", [], []))
    for doc, name in ((expected, "Production"), (actual, "Staging")):
        for i in ocr.scanned_pages(doc):
            note = ("its text was read by OCR and compared" if ocr.available()
                    else "OCR is not installed, so its text could not be compared")
            found.append(("scanned-page", f"Scanned page {i + 1} in {name}",
                          f"{name} page {i + 1} is a scanned image with no text layer - {note}.",
                          page_box(doc, i) if name == "Production" else [],
                          page_box(doc, i) if name == "Staging" else []))
    chapter_id = chapter_rows[0]["id"] if chapter_rows else ""
    out = []
    for k, (kind, title, description, prod_boxes, stage_boxes) in enumerate(found):
        out.append({
            "id": start + k, "category": "content", "type": kind, "kind": "page",
            "content_label": "Page-level differences", "title": title, "description": description,
            "detail": "", "help": "", "chapter": "Whole document", "chapter_id": chapter_id, "section": "",
            "prod_anchor": None, "stage_anchor": None,
            "prod_page": prod_boxes[0]["page"] if prod_boxes else None,
            "stage_page": stage_boxes[0]["page"] if stage_boxes else None,
            "prod_boxes": prod_boxes, "stage_boxes": stage_boxes,
            "comment_prod": "", "comment_stage": "", "occurrences": 1,
            "prod_parts": [], "stage_parts": [], "changes": [],
            "noise": "", "minor": kind == "scanned-page", "critical": kind != "scanned-page",
        })
    return out


def build_issue_report(chapters: list, expected: fitz.Document, actual: fitz.Document,
                       output_dir: str | None,
                       expected_path: str | None = None, actual_path: str | None = None) -> dict:
    """The data behind pdf.html."""
    exp_entries, act_entries = resolve_entries(expected, actual)
    issues: list[dict] = []
    chapter_rows: list[dict] = []
    for ci, chapter in enumerate(chapters):
        before = len(issues)
        for diff in chapter.differences:
            a, b = diff.get("exp"), diff.get("act")
            repeats = list(diff.get("repeats") or [])
            # Only what differs is boxed: the changed words, the changed list
            # markers, the spots on a picture, the words that lost their bold -
            # the whole element only when nothing finer can be placed.
            prod_marks, stage_marks = [], []
            if diff.get("word_diff"):
                prod_marks, stage_marks = _word_marks(diff, a, b, expected, actual)
            elif diff.get("type") == "numbering":
                prod_marks, stage_marks = _numbering_marks(diff, expected, actual)
            prod_boxes = prod_marks or _boxes(a, diff.get("exp_marks"))
            stage_boxes = stage_marks or _boxes(b, diff.get("marks"))
            # Many small boxes on one paragraph are noise in the COMMENTS: one
            # note per changed word stacks half a dozen deep over the page. The
            # boxes themselves stay on the words - highlighting the paragraph
            # instead says every word of it differs, when the reader can see
            # most of it is identical - and every change is listed once, in the
            # single comment beside them.
            changes = [m["note"] for m in prod_boxes if m.get("note")] if prod_boxes else []
            changes += [m["note"] for m in stage_boxes if m.get("note") and not prod_boxes]
            if len(prod_boxes) > _BOXES_BEFORE_LIST or len(stage_boxes) > _BOXES_BEFORE_LIST:
                prod_boxes = [{k: v for k, v in m.items() if k != "note"} for m in prod_boxes]
                stage_boxes = [{k: v for k, v in m.items() if k != "note"} for m in stage_boxes]
            else:
                changes = []
            # A change made in several places is one issue, boxed everywhere -
            # on the same changed words there, not on the whole paragraph.
            for rep_a, rep_b in repeats:
                rep_p, rep_s = [], []
                if diff.get("word_diff"):
                    rep_p, rep_s = _word_marks(diff, rep_a, rep_b, expected, actual)
                prod_boxes += rep_p or _boxes(rep_a)
                stage_boxes += rep_s or _boxes(rep_b)
            # Which finding each box came from. Cards are consolidated from
            # several findings, and a finding with a box on one side only (a
            # figure missing in Staging, text only Production has) then left
            # the two sides' box lists out of step: the viewer opened
            # Production at its first box and Staging at a box belonging to a
            # different finding, pages apart, so the panes no longer mirrored.
            # With this, the viewer can open both at the same finding.
            pair = len(issues)
            for box in prod_boxes + stage_boxes:
                box.setdefault("pair", pair)
            # The issue's topic - the matched heading it sits under on both sides -
            # and where that topic starts in each document, so a click opens both
            # at the same topic even when one side has nothing to box.
            topic = (getattr(chapter, "topics", None) or {}).get(diff.get("section"))
            section = topic["title"] if topic else ""
            # A figure only one side prints still often has its caption text on
            # both - closer to the reader's actual place than the top of the
            # whole heading the figure falls under.
            exp_near, act_near = diff.get("exp_anchor"), diff.get("act_anchor")
            # A plain "missing"/"added" text finding (nothing to box on the
            # OTHER side at all) never gets one of those - it falls straight
            # to the topic heading below unless given a closer guess here:
            # roughly where the element's own page falls, proportionally,
            # within the other side's copy of the same chapter.
            # A finding under a matched section pins to that SAME section's
            # heading on the other side; a chapter-wide guess could land in a
            # neighbouring section (a table above, instead of "Audio format").
            in_section = bool(topic) and section != chapter.title
            if not in_section and act_near is None and a is not None and a.boxes and chapter.exp_pages and chapter.act_pages:
                page, y = a.boxes[0][0], a.boxes[0][1][1]
                act_near = _proportional_anchor(
                    expected, page, y, chapter.exp_pages, chapter.act_pages, actual)
            if not in_section and exp_near is None and b is not None and b.boxes and chapter.exp_pages and chapter.act_pages:
                page, y = b.boxes[0][0], b.boxes[0][1][1]
                exp_near = _proportional_anchor(
                    actual, page, y, chapter.act_pages, chapter.exp_pages, expected)
            prod_anchor = (
                {"page": exp_near[0] + 1, "y": _y_share(expected, exp_near[0], exp_near[1])}
                if exp_near and 0 <= exp_near[0] < expected.page_count else
                {"page": topic["prod"][0] + 1, "y": _y_share(expected, topic["prod"][0], topic["prod"][1])}
                if topic and 0 <= topic["prod"][0] < expected.page_count else None
            )
            stage_anchor = (
                {"page": act_near[0] + 1, "y": _y_share(actual, act_near[0], act_near[1])}
                if act_near and 0 <= act_near[0] < actual.page_count else
                {"page": topic["stage"][0] + 1, "y": _y_share(actual, topic["stage"][0], topic["stage"][1])}
                if topic and 0 <= topic["stage"][0] < actual.page_count else None
            )
            title, description = _title(diff["summary"]), _description(diff["summary"])
            if content_label(diff) == "Tables/content changes" and not title.startswith("Table issue"):
                title = f"Table issue: {title[:1].lower()}{title[1:]}"
            comment_prod, comment_stage = _comments(diff, title, description)
            if diff.get("type") in _MARKER_TYPES:
                def _pick_point(near, topic_key, doc):
                    if near and 0 <= near[0] < doc.page_count:
                        return near
                    if topic and 0 <= topic[topic_key][0] < doc.page_count:
                        return topic[topic_key]
                    return None
                if not prod_boxes:
                    point = _pick_point(exp_near, "prod", expected)
                    marker = _marker_box(expected, point, comment_prod) if point else None
                    if marker:
                        prod_boxes = [marker]
                if not stage_boxes:
                    point = _pick_point(act_near, "stage", actual)
                    marker = _marker_box(actual, point, comment_stage) if point else None
                    if marker:
                        stage_boxes = [marker]
            prod_parts, stage_parts = _side_content(diff, a, b)
            issues.append({
                "id": len(issues),
                "category": category_of(diff),
                "type": diff.get("type"),
                "kind": diff.get("kind"),
                "content_label": content_label(diff),
                "title": title,
                "description": description,
                "detail": diff.get("detail") or "",
                "help": _TYPE_HELP.get(diff.get("type"), ""),
                "chapter": chapter.title,
                "chapter_id": f"ch{ci}",
                "section": "" if section == chapter.title else section,
                "prod_anchor": prod_anchor,
                "stage_anchor": stage_anchor,
                "prod_page": prod_boxes[0]["page"] if prod_boxes else None,
                "stage_page": stage_boxes[0]["page"] if stage_boxes else None,
                "prod_boxes": prod_boxes[:MAX_BOXES_PER_ISSUE],
                "stage_boxes": stage_boxes[:MAX_BOXES_PER_ISSUE],
                "comment_prod": comment_prod if prod_boxes else "",
                "comment_stage": comment_stage if stage_boxes else "",
                "occurrences": 1 + len(repeats),
                "prod_parts": prod_parts,
                "stage_parts": stage_parts,
                "changes": changes,
                # Red: something missing or wrong in Staging. Orange: an expected
                # kind of difference to confirm - indent, merge, alignment, style
                # - or a "missing"/"added" text the OCR-softening pass could not
                # confirm either way (a spec caption baked into the other side's
                # artwork as pixels). `chapter.py` already computed this from the
                # diff's real severity (which factors in `review_only`); reading
                # only `type in ORANGE_TYPES` here duplicated that logic and fell
                # out of step with it, so a softened finding still rendered as an
                # urgent red failure.
                # A diff that marks itself critical overrides its type's usual
                # tier: it is neither noise to be filtered away nor an orange
                # "confirm this" - see `severity_of` in chapter.py.
                "noise": "" if diff.get("critical") else _NOISE_OF.get(diff.get("type"), ""),
                "minor": not diff.get("critical")
                         and (diff.get("type") in ORANGE_TYPES or diff.get("minor", False)),
                # Level 1 is what the document actually says or lacks outright -
                # the one kind of finding a reader must not miss among the rest
                # that also fail the run, so it is marked to draw darker.
                "critical": (
                    (diff.get("severity") == 1 and diff.get("type") not in ORANGE_TYPES
                     and not diff.get("review_only"))
                    or diff.get("type") in _ALWAYS_CRITICAL_TYPES
                    or bool(diff.get("critical"))
                ),
            })
        issues[before:] = _consolidate_issues(issues[before:])
        chapter_rows.append({
            "id": f"ch{ci}", "title": chapter.title, "count": len(issues) - before,
            "prod_range": _range(chapter.exp_pages), "stage_range": _range(chapter.act_pages),
        })

    issues.extend(_page_level_issues(expected, actual, len(issues), chapter_rows))

    # Numbered in the order the nav lists them - by category, then as printed -
    # so issue #12 is the twelfth line in the nav and the "#12" on the page.
    issues.sort(key=lambda it: (_CATEGORY_KEYS.index(it["category"]), it["id"]))
    for index, item in enumerate(issues):
        item["id"], item["n"] = index, index + 1

    counts = {key: 0 for key in _CATEGORY_KEYS}
    noise_counts = {g["key"]: 0 for g in NOISE_GROUPS}
    for item in issues:
        counts[item["category"]] += 1
        if item["noise"]:
            noise_counts[item["noise"]] += 1
    anchors = _anchors(expected, actual, exp_entries, act_entries)
    if expected_path and actual_path:
        # Table waypoints on top of heading ones: a chapter that packs several
        # tables onto one page in Production and spreads them across several
        # in Staging (or excludes a whole section, like a per-country RoHS
        # declaration, from Content Validation altogether) has no other
        # landmark between its surrounding headings, and drifts out of step
        # without one.
        anchors = _order_anchors(anchors + table_anchors(expected, actual, expected_path, actual_path))
    # Paragraph waypoints under both: headings and tables put the two panes at
    # the same SECTION, and between two of them the viewer can only interpolate
    # by proportion - which drifts as soon as the two documents paginate a
    # chapter differently. One per matched paragraph keeps them level all the
    # way down. Added last and re-ordered, so a paragraph that would run the
    # Staging side backwards is dropped exactly like a table or heading.
    anchors = _order_anchors(anchors + content_anchors(chapters, expected, actual))
    return {
        "total": len(issues),
        "passed": not issues,
        "categories": [{**c, "count": counts[c["key"]]} for c in CATEGORIES],
        # Only the kinds this pair actually produced: a switch for something
        # the run never found is a switch that does nothing.
        "noise_groups": [
            {**g, "count": noise_counts[g["key"]]} for g in NOISE_GROUPS if noise_counts[g["key"]]
        ],
        "groups": [{"key": c["key"], "label": c["label"], "help": c["help"],
                    "ids": [it["id"] for it in issues if it["category"] == c["key"]]} for c in CATEGORIES],
        "issues": issues,
        "chapters": chapter_rows,
        "anchors": anchors,
        "prod_pages": _render(expected, output_dir, "prod") if output_dir else [],
        "stage_pages": _render(actual, output_dir, "stage") if output_dir else [],
    }
