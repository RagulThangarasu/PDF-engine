"""The chapter browser: a whole L1 chapter, Production beside Staging.

`pdfval.validators.chapter` decides WHAT differs; this turns that into what a
reviewer looks at:

  * every page the chapter occupies in each document, rendered as it prints and
    boxed where it differs - however many pages that is, and whatever page
    numbers each document happens to use;
  * the chapter's whole content, element by element, left against right, in the
    order it is printed - not only the parts that differ, so a reader can see
    that the rest was actually compared;
  * every difference written out in full, numbered, with the same number on the
    box in both page renders.

The renders are JPEG, not PNG: these are pictures of pages, there are a couple
of hundred of them in a run, and a PNG copy of this same set costs ~26 MB
against ~8 MB for no visible gain.
"""
from __future__ import annotations

import os

import fitz

from pdfval.report import screenshots as _shots
from pdfval.validators.chapter import (
    KIND_FIGURE,
    KIND_LABEL,
    KIND_TABLE,
    FAILING_SEVERITY,
    SEVERITY_LABEL,
    Chapter,
    merge_elements,
)

PAGE_DPI = 110          # a full page, readable at the width the browser shows it
PAGE_JPEG_QUALITY = 78
CHAPTERS_SUBDIR = "chapters"
MAX_PAGES_PER_SIDE = 40  # a runaway chapter cannot fill the run directory

# How each difference is grouped in the "what differs" list. Same five
# categories the rest of the engine uses, so a reader moving between reports
# does not have to learn a second vocabulary.
_CATEGORY_OF = {
    "missing": "content", "added": "content", "text": "content", "moved": "content",
    "punctuation": "content", "capitalisation": "content", "word-order": "content",
    "table-shape": "table", "table-cell": "table", "table-extract": "table",
    "figure-content": "image", "figure-review": "image", "figure-size": "image",
    "figure-missing": "image", "figure-added": "image", "figure-elsewhere": "image",
    "figure-different": "image",
    "note-label": "note",
    "table-merge": "table", "table-header-fill": "table",
    "link-missing": "link", "link-added": "link", "link-target": "link", "link-broken": "link",
    "numbering": "numbering", "list-marker-missing": "numbering", "list-marker-added": "numbering",
    "marker-size": "numbering",
    "table-header-repeat": "table", "figure-label-missing": "image", "figure-alignment": "image",
    "bold-missing": "bold",
    "underline-missing": "formatting", "underline-added": "formatting",
    "emphasis": "formatting", "size": "formatting", "colour": "formatting",
}
_CATEGORY_META = {
    "content": {"label": "Content", "icon": "✎"},
    "table": {"label": "Table", "icon": "▦"},
    "image": {"label": "Image", "icon": "🖼"},
    "note": {"label": "Note", "icon": "❕"},
    "link": {"label": "Hyperlink", "icon": "🔗"},
    "numbering": {"label": "Lists", "icon": "1→a"},
    "bold": {"label": "Bold", "icon": "B"},
    "formatting": {"label": "Formatting", "icon": "Aa"},
}
_CATEGORY_ORDER = ["content", "numbering", "link", "table", "image", "note", "bold", "formatting"]

# What each kind of issue means, in the reader's words - the "Issue types"
# table at the top of the page explains each one once, instead of every issue
# explaining itself.
_TYPE_HELP = {
    "text": "The same passage is worded differently: words Production prints are missing from Staging, or Staging prints words Production does not.",
    "missing": "Content Production prints that has no counterpart anywhere in Staging's copy of the chapter.",
    "added": "Content Staging prints that Production does not have anywhere in the chapter.",
    "numbering": "The same list items are numbered or lettered differently (1., 2. against a., b.).",
    "list-marker-missing": "A list item in Production is plain text in Staging - its bullet or number is gone.",
    "list-marker-added": "Plain text in Production is a list item in Staging.",
    "marker-size": "The same list items use the same kind of marker on both sides, but it prints at a noticeably different size in Staging.",
    "note-label": "A NOTE / TIP / WARNING callout is marked as a different type.",
    "table-cell": "The text of a table cell differs.",
    "link-missing": "Text that is a link in Production is plain text in Staging.",
    "link-added": "Text that is plain in Production is a link in Staging.",
    "link-target": "The link text is the same, but the link goes somewhere else.",
    "link-broken": "The link does not work in Staging.",
    "table-shape": "The table has a different number of rows or columns.",
    "table-merge": "Table cells are merged or split differently.",
    "table-header-repeat": "A table that continues onto another page does not repeat its header row there.",
    "icon-missing": "An inline icon printed beside these words in Production is not printed in Staging.",
    "icon-added": "Staging prints an inline icon beside these words that Production does not.",
    "icon-colour": "The same inline icon is printed in a clearly different colour in Staging.",
    "table-as-text": "Production prints these rows as a table; Staging prints the same words as plain lines, with no table.",
    "table-row-missing": "A row of Production's table is not in Staging's table, and its words are not printed anywhere else in the section.",
    "table-row-added": "Staging's table has a row Production's does not.",
    "table-columns": "The table has a different number of columns.",
    "figure-missing": "A figure printed in Production is not printed anywhere in Staging.",
    "figure-added": "A figure printed in Staging is not in Production.",
    "figure-elsewhere": "The figure is printed under a different section in Staging.",
    "figure-different": "A different picture is printed in this place.",
    "figure-content": "A different picture is printed in this place.",
    "figure-size": "The same figure is printed noticeably larger in Staging.",
    "figure-alignment": "The figure is aligned differently (left / centre / right).",
    "figure-label-missing": "Label text printed on the Production figure is missing from the Staging one.",
    "bold-missing": "Words set in bold in Production are regular weight in Staging.",
    "bold-added": "Words set regular in Production are bold in Staging.",
    "underline-missing": "Words underlined in Production are printed plain in Staging.",
    "underline-added": "Words printed plain in Production are underlined in Staging.",
}

# Unchanged words kept either side of a difference in the issue table - enough
# to find the sentence, not so many that the red is lost in a paragraph.
_CONTEXT_WORDS = 8
_WHOLE_MAX_WORDS = 60


def _snippet(ops: list[dict], side: str) -> list[dict]:
    """One side's own text for an issue, with what differs flagged.

    Production keeps its unchanged words and what Staging is missing; Staging
    keeps its unchanged words and what it adds. Where only the OTHER side has
    words, a `gap` marks the spot. A long unchanged stretch is cut down to the
    few words either side of a change.
    """
    mine, theirs = ("del", "ins") if side == "prod" else ("ins", "del")
    parts: list[dict] = []
    for op in ops:
        if op["type"] == "equal":
            parts.append({"t": op["text"]})
        elif op["type"] == mine:
            parts.append({"t": op["text"], "hl": True})
        elif op["type"] == theirs:
            parts.append({"t": "", "gap": True})
    # A gap beside this side's own flagged words is the same change twice.
    parts = [
        p for i, p in enumerate(parts)
        if not p.get("gap") or not any(
            0 <= j < len(parts) and parts[j].get("hl") for j in (i - 1, i + 1)
        )
    ]
    n = _CONTEXT_WORDS
    for i, p in enumerate(parts):
        if p.get("hl") or p.get("gap"):
            continue
        words = p["t"].split()
        before, after = i > 0, i < len(parts) - 1
        if before and after and len(words) > 2 * n:
            p["t"] = " ".join(words[:n]) + " … " + " ".join(words[-n:])
        elif before and not after and len(words) > n:
            p["t"] = " ".join(words[:n]) + " …"
        elif after and not before and len(words) > n:
            p["t"] = "… " + " ".join(words[-n:])
    return parts


def _whole(el) -> dict:
    """An element one side prints and the other does not, flagged whole."""
    if el.kind == KIND_FIGURE:
        text = _element_label(el)
    elif el.kind == KIND_TABLE:
        text = f"{_element_label(el)}: {el.text}"
    else:
        text = el.text or _element_label(el)
    words = text.split()
    if len(words) > _WHOLE_MAX_WORDS:
        text = " ".join(words[:_WHOLE_MAX_WORDS]) + " …"
    return {"t": text, "hl": True}


def _side_content(diff: dict, a, b) -> tuple[list[dict], list[dict]]:
    """(production, staging) text for the issue table, the differing part of
    each flagged - or empty where the issue is not about words (a table's
    shape, a figure's size), which the description already says."""
    if diff.get("word_diff"):
        return _snippet(diff["word_diff"], "prod"), _snippet(diff["word_diff"], "stage")
    kind = diff["type"]
    if kind in ("missing", "figure-missing") and a is not None:
        return [_whole(a)], []
    if kind in ("added", "figure-added") and b is not None:
        return [], [_whole(b)]
    if kind == "link-missing" and a is not None:
        return [{"t": f"link on “{a.text[:120]}”", "hl": True}], []
    if kind == "link-added" and b is not None:
        return [], [{"t": f"link on “{b.text[:120]}”", "hl": True}]
    return [], []


def _element_label(el) -> str:
    """What to call an element in the content list."""
    if el is None:
        return ""
    if el.kind == KIND_FIGURE:
        return f"Figure {el.width:.0f}×{el.height:.0f}pt"
    if el.kind == KIND_TABLE:
        return f"Table {el.rows}×{el.cols}"
    return KIND_LABEL.get(el.kind, "Text")


def _render_side(
    doc: fitz.Document,
    pages: list[int],
    limits: dict[int, tuple[float, float]],
    boxes: dict[int, list[tuple]],
    out_dir: str,
    prefix: str,
) -> list[dict]:
    """Every page of the chapter on this side, rendered and boxed.

    `limits` clips the first and last page to the chapter's own band, so a
    chapter that starts half way down a page does not open with the end of the
    chapter before it. Pages in between are rendered whole - the chapter runs
    through them.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return []
    folder = os.path.join(out_dir, CHAPTERS_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    zoom = PAGE_DPI / 72.0
    shots: list[dict] = []
    for page_index in pages[:MAX_PAGES_PER_SIDE]:
        if not (0 <= page_index < doc.page_count):
            continue
        page = doc[page_index]
        rect = page.rect
        y0, y1 = limits.get(page_index, (rect.y0, rect.y1))
        clip = fitz.Rect(rect.x0, max(rect.y0, y0 - 4), rect.x1, min(rect.y1, y1 + 4))
        if clip.height < 8:
            continue
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception:
            continue
        draw = ImageDraw.Draw(img, "RGBA")
        for bbox, label, kind in boxes.get(page_index, []):
            rect_px = (
                (bbox[0] - clip.x0) * zoom - 2, (bbox[1] - clip.y0) * zoom - 2,
                (bbox[2] - clip.x0) * zoom + 2, (bbox[3] - clip.y0) * zoom + 2,
            )
            _shots.draw_highlight(draw, img, rect_px, kind, label)
        name = f"{prefix}_p{page_index + 1}.jpg"
        try:
            img.save(os.path.join(folder, name), format="JPEG", quality=PAGE_JPEG_QUALITY)
        except Exception:
            continue
        shots.append({
            "page": page_index + 1,
            "src": f"{CHAPTERS_SUBDIR}/{name}",
            "w": img.width,
            "h": img.height,
            # Where this image sits on the page, so an issue's close-up can be
            # cut out of it instead of rendering the page a second time.
            "clip_x0": clip.x0,
            "clip_y0": clip.y0,
            "zoom": zoom,
        })
    return shots


def _limits(chapter_span, pages: list[int], doc: fitz.Document) -> dict[int, tuple[float, float]]:
    """The y-band the chapter occupies on its first and last page."""
    (start_page, start_y), end = chapter_span
    out: dict[int, tuple[float, float]] = {}
    for page_index in pages:
        rect = doc[page_index].rect
        y0 = start_y - 6 if page_index == start_page else rect.y0
        y1 = end[1] if (end and page_index == end[0]) else rect.y1
        out[page_index] = (y0, y1)
    return out


def build_chapter_report(
    chapters: list[Chapter],
    spans: list,
    expected: fitz.Document,
    actual: fitz.Document,
    output_dir: str | None,
) -> list[dict]:
    """The data behind chapters.html - one entry per shared L1 chapter."""
    out: list[dict] = []
    for n, chapter in enumerate(chapters):
        span = spans[n] if n < len(spans) else None
        # Number the differences in reading order; the number is what ties a
        # line in the list to a box on the page, on both sides at once.
        exp_boxes: dict[int, list[tuple]] = {}
        act_boxes: dict[int, list[tuple]] = {}
        diffs: list[dict] = []
        for i, diff in enumerate(chapter.differences, start=1):
            label = str(i)
            a, b = diff.get("exp"), diff.get("act")
            # Red for a difference in what the chapter SAYS or shows; thin
            # orange for one in how it is set. Drawn alike, a heading printed
            # a point smaller looked exactly as wrong on the page as a word
            # that changed, and the one real typo in a chapter disappeared
            # among a dozen boxes round text that reads identically.
            level = diff.get("severity") or 1
            kind = _shots.KIND_DIFF if level <= FAILING_SEVERITY else _shots.KIND_CONTEXT
            # A change repeated across the chapter is one finding, but every
            # place it occurs is still boxed - under the same number.
            occurrences = [(a, b)] + list(diff.get("repeats") or [])
            for occ_a, occ_b in occurrences:
                for el, store in ((occ_a, exp_boxes), (occ_b, act_boxes)):
                    if el is None:
                        continue
                    for page_index, bbox in el.boxes:
                        store.setdefault(page_index, []).append((bbox, label, kind))
            category = _CATEGORY_OF.get(diff["type"], "content")
            prod_parts, stage_parts = _side_content(diff, a, b)
            topic = diff["summary"].split(" — ")[0].rstrip(". ")
            diffs.append({
                "n": i,
                "chapter_id": f"ch{n}",
                "chapter_title": chapter.title,
                # The issue's type in the validator's own words ("Text content
                # changed", "Figure oversized in Staging"). The key sorts the
                # page's issue-type table most serious first.
                "topic": topic,
                "topic_key": f"{level}-{topic}",
                "topic_help": _TYPE_HELP.get(diff["type"], ""),
                "prod_parts": prod_parts,
                "stage_parts": stage_parts,
                "type": diff["type"],
                "category": category,
                "label": _CATEGORY_META[category]["label"],
                "icon": _CATEGORY_META[category]["icon"],
                "element": KIND_LABEL.get(diff.get("kind"), "Content"),
                "summary": diff["summary"],
                "detail": diff.get("detail") or "",
                "word_diff": diff.get("word_diff") or [],
                "minor": level > FAILING_SEVERITY,
                "severity": level,
                "severity_label": SEVERITY_LABEL.get(level, "Content"),
                "_exp": a,
                "_act": b,
                "prod_page": (a.page + 1) if a is not None else None,
                "stage_page": (b.page + 1) if b is not None else None,
                "occurrences": len(occurrences),
            })

        # Which numbers land on which row, so the content list and the page
        # renders point at each other.
        # Each row names the finding numbers it carries. A repeated change is
        # one finding, so several rows can share a number.
        numbers_by_row: dict[int, list[int]] = {
            row_index: sorted({ref + 1 for ref in row.get("refs") or []})
            for row_index, row in enumerate(chapter.rows)
            if row.get("refs")
        }

        rows = []
        for row_index, row in enumerate(chapter.rows):
            exp_el = merge_elements(row["exp"])
            act_el = merge_elements(row["act"])
            # A layout row is content one document divides differently: say
            # WHERE the other document prints it, in the column that would
            # otherwise read "nothing here" - which is precisely the claim the
            # row is not making.
            container_side = container_note = ""
            if False:
                container_side = "stage" if row["exp"] else "prod"
                other = "Staging" if container_side == "stage" else "Production"
                holder = row.get("container")
                if holder is not None:
                    snippet = holder.text[:90] + ("…" if len(holder.text) > 90 else "")
                    container_note = (
                        f"same content, printed inside a {_element_label(holder).lower()} "
                        f"on p.{holder.page + 1} in {other}: “{snippet}”"
                    )
                else:
                    container_note = (
                        f"same words, divided into different blocks in {other}’s copy of this chapter"
                    )
            rows.append({
                "container_side": container_side,
                "container_note": container_note,
                # Content one document divides differently is a match: shown as
                # one, with no note - it is not an issue and needs no room.
                "status": "match" if row["status"] == "layout" else row["status"],
                "numbers": numbers_by_row.get(row_index, []),
                "kind": (exp_el or act_el).kind if (exp_el or act_el) else "text",
                "prod_label": _element_label(exp_el),
                "stage_label": _element_label(act_el),
                "prod_text": exp_el.text if exp_el else "",
                "stage_text": act_el.text if act_el else "",
                "prod_page": (exp_el.page + 1) if exp_el else None,
                "stage_page": (act_el.page + 1) if act_el else None,
                "summaries": [d["summary"] for d in row["differences"]],
                "word_diff": next(
                    (d["word_diff"] for d in row["differences"] if d.get("word_diff")), []
                ),
            })

        layout_rows = sum(1 for row in chapter.rows if row["status"] == "layout")
        counts: dict[str, int] = {}
        for item in diffs:
            counts[item["category"]] = counts.get(item["category"], 0) + 1

        entry = {
            "id": f"ch{n}",
            "title": chapter.title,
            "prod_pages": [p + 1 for p in chapter.exp_pages],
            "stage_pages": [p + 1 for p in chapter.act_pages],
            "prod_range": _range(chapter.exp_pages),
            "stage_range": _range(chapter.act_pages),
            "elements": {"prod": len(chapter.exp_elements), "stage": len(chapter.act_elements)},
            "diffs": diffs,

            "rows": rows,
            "layout_rows": layout_rows,
            "total": len(diffs),
            "real_total": sum(1 for d in diffs if d["severity"] <= FAILING_SEVERITY),
            "categories": [
                {**_CATEGORY_META[key], "key": key, "count": counts[key]}
                for key in _CATEGORY_ORDER if counts.get(key)
            ],
            "skipped": {
                "prod": list((chapter.skipped or {}).get("prod") or []),
                "stage": list((chapter.skipped or {}).get("stage") or []),
            },
            "only_tops": {
                "prod": list(chapter.exp_only_tops),
                "stage": list(chapter.act_only_tops),
            },
            "prod_shots": [],
            "stage_shots": [],
        }
        if output_dir and span is not None:
            entry["prod_shots"] = _render_side(
                expected, chapter.exp_pages, _limits(span.exp_span, chapter.exp_pages, expected),
                exp_boxes, output_dir, f"ch{n}_prod",
            )
            entry["stage_shots"] = _render_side(
                actual, chapter.act_pages, _limits(span.act_span, chapter.act_pages, actual),
                act_boxes, output_dir, f"ch{n}_stage",
            )
        # Each issue's own close-up on both sides, cut from the page renders,
        # and the issues grouped most serious first.
        groups: dict[int, list[dict]] = {}
        for d in diffs:
            kind = _shots.KIND_DIFF if d["severity"] <= FAILING_SEVERITY else _shots.KIND_CONTEXT
            d["prod_crop"] = _issue_shot(expected, d.pop("_exp"), output_dir, f"ch{n}_issue{d['n']}_prod", str(d["n"]), kind)
            d["stage_crop"] = _issue_shot(actual, d.pop("_act"), output_dir, f"ch{n}_issue{d['n']}_stage", str(d["n"]), kind)
            groups.setdefault(d["severity"], []).append(d)
        entry["severity_groups"] = [
            {"level": level, "label": SEVERITY_LABEL.get(level, ""), "fails": level <= FAILING_SEVERITY,
             "items": groups[level]}
            for level in sorted(groups)
        ]
        out.append(entry)
    return out


_ISSUE_SHOTS = [0]
MAX_ISSUE_SHOTS = 1500


def _issue_shot(doc: fitz.Document, el, output_dir: str | None, name: str, label: str, kind: str) -> str | None:
    """A screenshot of just this issue: the element's own region, rendered
    fresh, with only this issue's box and number on it. A slice of the shared
    page image carried every neighbouring issue's box as well - three numbers
    stacked on one drawing - and a reader could not tell which one was meant."""
    if el is None or not el.boxes or not output_dir or _ISSUE_SHOTS[0] >= MAX_ISSUE_SHOTS:
        return None
    page = el.boxes[0][0]
    boxes = [b for p, b in el.boxes if p == page]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    _ISSUE_SHOTS[0] += 1
    return _shots.capture_region(doc, page, output_dir, name, bbox, kind, label)


def _range(pages: list[int]) -> str:
    if not pages:
        return "—"
    lo, hi = pages[0] + 1, pages[-1] + 1
    return f"p.{lo}" if lo == hi else f"p.{lo}–{hi}"
