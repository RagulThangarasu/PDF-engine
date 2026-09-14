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
from pdfval.validators.chapter import CATEGORIES, category_of
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
# row, a/b/c instead of 1/2/3, a lost bullet, a missing picture or link - is red.
ORANGE_TYPES = {
    "list-indent", "table-merge", "table-columns", "figure-alignment", "figure-size",
    "bold-missing", "bold-added", "shading", "marker-size", "text-space", "icon-changed",
    "underline-missing", "underline-added",
}


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
    fixed = {
        "missing": ("Not in Staging", ""),
        "added": ("", "Not in Production"),
        "figure-missing": ("Picture missing in Staging", ""),
        "figure-added": ("", "Picture not in Production"),
        "figure-spots": ("", "Black spot in Staging's picture"),
        "bold-missing": ("Bold in Production", "Not bold in Staging"),
        "bold-added": ("Regular in Production", "Bold added in Staging"),
        "link-missing": ("Link in Production", "Link missing in Staging"),
        "link-added": ("No link in Production", "Extra link in Staging"),
        "link-broken": ("", "Link does not work in Staging"),
        "list-marker-missing": ("List item in Production", "Bullet / number missing in Staging"),
        "list-marker-added": ("Plain text in Production", "Bullet / number added in Staging"),
        "table-row-missing": ("Row missing in Staging", "This row is missing from the table"),
        "table-as-text": ("Table in Production", "Table missing in Staging — rows printed as plain text"),
        "table-fill-missing": ("Background in Production", "Background missing in Staging"),
        "icon-missing": ("Icon in Production", "Icon missing in Staging"),
        "icon-added": ("No icon in Production", "Icon added in Staging"),
        "icon-colour": ("Icon colour in Production", "Icon colour changed in Staging"),
        "table-row-added": ("Row not in Production's table", "Row added in Staging"),
    }
    if kind in fixed:
        return fixed[kind]
    if kind == "shading":
        return (("Plain page in Production", "Shaded box added in Staging") if "added" in title
                else ("Shaded box in Production", "Shading missing in Staging"))
    if kind == "link-target":
        # detail reads "Production: section: X · Staging: section: Y"
        sides = (diff.get("detail") or "").split(" · ")
        if len(sides) == 2:
            where = [re.sub(r"^\s*section\s*:\s*", "", s.split(":", 1)[-1]).strip() for s in sides]
            both = f"Production links to “{where[0]}”; Staging links to “{where[1]}”"
            return _clip(both, 120), _clip(both, 120)
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


def _anchors(expected: fitz.Document, actual: fitz.Document, exp_entries, act_entries) -> list[dict]:
    """The headings both documents have, with where each prints on each side -
    what the viewer scrolls the two documents together by. Kept in reading order
    on BOTH sides: a heading Staging moved elsewhere cannot be a waypoint, or
    scrolling forward in one document would jump the other one back."""
    out: list[dict] = []
    for m in match_toc_entries(exp_entries, act_entries):
        if m.expected_index is None or m.actual_index is None:
            continue
        e, a = exp_entries[m.expected_index], act_entries[m.actual_index]
        if not (0 <= e.page < expected.page_count and 0 <= a.page < actual.page_count):
            continue
        out.append({
            "title": e.title, "level": e.level,
            "prod": {"page": e.page + 1, "y": _y_share(expected, e.page, e.y)},
            "stage": {"page": a.page + 1, "y": _y_share(actual, a.page, a.y)},
        })
    out.sort(key=lambda h: (h["prod"]["page"], h["prod"]["y"]))
    kept: list[dict] = []
    for h in out:
        if kept and (h["stage"]["page"], h["stage"]["y"]) <= (kept[-1]["stage"]["page"], kept[-1]["stage"]["y"]):
            continue
        kept.append(h)
    return kept


def build_issue_report(chapters: list, expected: fitz.Document, actual: fitz.Document,
                       output_dir: str | None) -> dict:
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
            prod_anchor = (
                {"page": topic["prod"][0] + 1, "y": _y_share(expected, topic["prod"][0], topic["prod"][1])}
                if topic and 0 <= topic["prod"][0] < expected.page_count else None
            )
            stage_anchor = (
                {"page": topic["stage"][0] + 1, "y": _y_share(actual, topic["stage"][0], topic["stage"][1])}
                if topic and 0 <= topic["stage"][0] < actual.page_count else None
            )
            title, description = _title(diff["summary"]), _description(diff["summary"])
            comment_prod, comment_stage = _comments(diff, title, description)
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
                # kind of difference to confirm - indent, merge, alignment, style.
                "minor": diff.get("type") in ORANGE_TYPES,
                # Level 1 is what the document actually says or lacks outright -
                # the one kind of finding a reader must not miss among the rest
                # that also fail the run, so it is marked to draw darker.
                "critical": diff.get("severity") == 1 and diff.get("type") not in ORANGE_TYPES,
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
    return {
        "total": len(issues),
        "passed": not issues,
        "categories": [{**c, "count": counts[c["key"]]} for c in CATEGORIES],
        "groups": [{"key": c["key"], "label": c["label"], "help": c["help"],
                    "ids": [it["id"] for it in issues if it["category"] == c["key"]]} for c in CATEGORIES],
        "issues": issues,
        "chapters": chapter_rows,
        "anchors": _anchors(expected, actual, exp_entries, act_entries),
        "prod_pages": _render(expected, output_dir, "prod") if output_dir else [],
        "stage_pages": _render(actual, output_dir, "stage") if output_dir else [],
    }
