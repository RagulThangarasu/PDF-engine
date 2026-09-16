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

import fitz

from pdfval.report.chapters import _TYPE_HELP, _range, _side_content
from pdfval.validators.chapter import CATEGORIES, category_of, table_anchors
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
_BOXES_BEFORE_LIST = 3         # more word boxes than this on a side: one box, changes listed

# Drawn orange. Everything else - missing or wrong content, a missing table or
# row, a/b/c instead of 1/2/3, a lost bullet, a missing picture or link - is
# red. Bold is red too, not orange: `confirm_bold` only ever keeps a
# "bold-missing"/"bold-added" candidate once it has measured the two sides'
# printed stroke weight and confirmed Staging's copy genuinely reads lighter
# or heavier, so by the time one reaches here it is exactly as real a defect
# as a wrong word - not a styling nuance to merely flag for review.
ORANGE_TYPES = {
    "list-indent", "table-merge", "table-columns", "table-cell-layout", "figure-alignment", "figure-size",
    "shading", "marker-size", "text-space", "icon-changed",
    "underline-missing", "underline-added", "line-spacing",
}

# An image or icon genuinely gone from one side - not resized, not moved, not
# rendering a little differently, but not there at all. Severity alone (4, the
# same tier as "figure oversized") would draw it in the ordinary, same-as-
# every-other-Images-finding blue; outright absence is content missing
# outright, same as a level-1 Content finding, and is drawn just as critical -
# thicker, opaque, and in red - so it never blends into a page full of routine
# size/alignment differences.
_ALWAYS_CRITICAL_TYPES = {"figure-missing", "figure-added", "icon-missing", "icon-added"}


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


def _marker_box(doc: fitz.Document, point: tuple, note: str) -> dict | None:
    """A small pin at roughly where the missing/extra content would sit on
    this side, carrying the SAME comment as the side that has it - so the
    issue is visible, and reads the same, whichever document is on screen."""
    page_index, y = point
    try:
        rect = doc[page_index].rect
    except Exception:
        return None
    x0 = rect.x0 + 36
    x1 = min(rect.x1 - 8, x0 + 90)
    y0 = min(max(rect.y0, y), max(rect.y0, rect.y1 - 14))
    return {"page": page_index + 1, "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y0 + 14, 2)],
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


def _order_anchors(out: list[dict]) -> list[dict]:
    """Sort a set of waypoints into Production's reading order and drop any
    `stage` side that would then run backwards - a heading or table Staging
    prints earlier than one already placed cannot be a waypoint, or scrolling
    forward in Production would jump Staging back."""
    out.sort(key=lambda h: (h["prod"]["page"], h["prod"]["y"]))
    last_stage = (0, 0.0)
    for h in out:
        if h["stage"] is not None:
            here = (h["stage"]["page"], h["stage"]["y"])
            if here <= last_stage:
                h["stage"] = None  # out of order - a Production-only waypoint instead of a wrong one
            else:
                last_stage = here
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
            # Many small boxes on one paragraph are noise: box the paragraph once
            # and list every change, one per line, in the comment instead.
            changes = [m["note"] for m in prod_boxes if m.get("note")] if prod_boxes else []
            changes += [m["note"] for m in stage_boxes if m.get("note") and not prod_boxes]
            if len(prod_boxes) > _BOXES_BEFORE_LIST or len(stage_boxes) > _BOXES_BEFORE_LIST:
                prod_boxes = _boxes(a) if a is not None else prod_boxes
                stage_boxes = _boxes(b) if b is not None else stage_boxes
            else:
                changes = []
            # A change made in several places is one issue, boxed everywhere.
            for rep_a, rep_b in repeats:
                prod_boxes += _boxes(rep_a)
                stage_boxes += _boxes(rep_b)
            # The issue's topic - the matched heading it sits under on both sides -
            # and where that topic starts in each document, so a click opens both
            # at the same topic even when one side has nothing to box.
            topic = (getattr(chapter, "topics", None) or {}).get(diff.get("section"))
            section = topic["title"] if topic else ""
            # A figure only one side prints still often has its caption text on
            # both - closer to the reader's actual place than the top of the
            # whole heading the figure falls under.
            exp_near, act_near = diff.get("exp_anchor"), diff.get("act_anchor")
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
                "minor": diff.get("type") in ORANGE_TYPES or diff.get("minor", False),
                # Level 1 is what the document actually says or lacks outright -
                # the one kind of finding a reader must not miss among the rest
                # that also fail the run, so it is marked to draw darker.
                "critical": (
                    (diff.get("severity") == 1 and diff.get("type") not in ORANGE_TYPES
                     and not diff.get("review_only"))
                    or diff.get("type") in _ALWAYS_CRITICAL_TYPES
                ),
            })
        chapter_rows.append({
            "id": f"ch{ci}", "title": chapter.title, "count": len(issues) - before,
            "prod_range": _range(chapter.exp_pages), "stage_range": _range(chapter.act_pages),
        })

    # Numbered in the order the nav lists them - by category, then as printed -
    # so issue #12 is the twelfth line in the nav and the "#12" on the page.
    issues.sort(key=lambda it: (_CATEGORY_KEYS.index(it["category"]), it["id"]))
    for index, item in enumerate(issues):
        item["id"], item["n"] = index, index + 1

    counts = {key: 0 for key in _CATEGORY_KEYS}
    for item in issues:
        counts[item["category"]] += 1
    anchors = _anchors(expected, actual, exp_entries, act_entries)
    if expected_path and actual_path:
        # Table waypoints on top of heading ones: a chapter that packs several
        # tables onto one page in Production and spreads them across several
        # in Staging (or excludes a whole section, like a per-country RoHS
        # declaration, from Content Validation altogether) has no other
        # landmark between its surrounding headings, and drifts out of step
        # without one.
        anchors = _order_anchors(anchors + table_anchors(expected, actual, expected_path, actual_path))
    return {
        "total": len(issues),
        "passed": not issues,
        "categories": [{**c, "count": counts[c["key"]]} for c in CATEGORIES],
        "groups": [{"key": c["key"], "label": c["label"], "help": c["help"],
                    "ids": [it["id"] for it in issues if it["category"] == c["key"]]} for c in CATEGORIES],
        "issues": issues,
        "chapters": chapter_rows,
        "anchors": anchors,
        "prod_pages": _render(expected, output_dir, "prod") if output_dir else [],
        "stage_pages": _render(actual, output_dir, "stage") if output_dir else [],
    }
