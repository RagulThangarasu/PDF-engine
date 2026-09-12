"""Neither column of a finding may be left blank.

A finding only one document has evidence for - text Staging dropped, a figure it
never rendered, a table row that is gone - used to show one screenshot beside an
empty column. That tells the reader nothing about the question they actually
have ("is it really gone, or did it just move?"), and it is the one case where
seeing the other document matters most.

Every finding names the section it sits in, and
`pdfval.validators.headings.resolve_entries` guarantees both documents carry
that section - so the empty column is filled with the other document's own view
of the same section, boxed in thin orange as CONTEXT rather than bold red as a
difference, and carrying the finding's own number so it reads as the partner of
the crop beside it.
"""
from __future__ import annotations

import difflib
import itertools
import math
import re
from collections import Counter

import fitz

from pdfval.report import screenshots

# How the counterpart spot is found, best first:
#
#   1. The text directly ABOVE the flagged region in the document that has it,
#      located again in the other document. A section band alone is a crop of
#      a whole page with a box round most of it - technically "the right
#      section", useless as an answer to "where would this have been?".
#   2. Failing that, the section band.
_NEIGHBOUR_LOOK_UP = 90.0   # points above the region to look for its lead-in line
_NEIGHBOUR_MIN_CHARS = 10   # shorter than this is not a distinctive lead-in
_NEIGHBOUR_MIN_RATIO = 0.72  # similarity for "this is the same line"
_NEIGHBOUR_BAND = 150.0     # minimum points below the lead-in to frame
_NEIGHBOUR_MAX_PAGES = 4    # a section runs on; the two documents paginate differently
# The frame is sized from the REGION it stands in for. A fixed 150pt band put a
# two-line strip of prose opposite a 400pt diagram and asked the reader to
# compare them - the counterpart has to cover a comparable area to be an answer.
_BAND_SOURCE_SCALE = 1.15
_BAND_MAX = 520.0
_FIGURE_MIN_SIDE = 24.0     # smaller than this is a rule or a bullet, not artwork
_MIN_GAP_BAND = 10.0        # a "where it belongs" box thinner than this is a hairline;
                            # below it the lead-in line comes back inside the box so
                            # there is something for the reader to recognise

# Matching the flagged region's own CONTENT in the other document, which is
# what the reader is really asking to see. A band measured down from a lead-in
# line lands wherever that line happens to fall; the table or sentence the
# finding is about can be half a page further on, and the two boxes then frame
# different things - a Staging table beside a Production diagram - which reads
# as a difference that was never reported.
_CONTENT_MIN_TOKENS = 4     # fewer real words than this identifies nothing
_CONTENT_MIN_SCORE = 0.45   # token overlap (F1) for "this says the same thing"
_CONTENT_LINE_SLACK = 4     # lines either side of the source's own line count
_CONTENT_PAD = 3.0          # points of padding around the matched lines


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _page_lines(doc: fitz.Document, page_index: int) -> list[tuple[float, float, float, float, str]]:
    """(x0, y0, x1, y1, text) for every text line on the page, in reading order."""
    out: list[tuple[float, float, float, float, str]] = []
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return out
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            text = "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
            if text:
                x0, y0, x1, y1 = (float(v) for v in ln["bbox"])
                out.append((x0, y0, x1, y1, text))
    out.sort(key=lambda ln: (ln[1], ln[0]))
    return out


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD_RE.findall(text or ""))


def _token_f1(a: Counter, b: Counter) -> float:
    """How much two pieces of text say the same thing, 0-1. A multiset F1 rather
    than a plain overlap, so a candidate that contains the wanted words plus a
    page of others does not score as a perfect match."""
    total = sum(a.values()) + sum(b.values())
    if not total:
        return 0.0
    return 2 * sum((a & b).values()) / total


def _region_text(doc: fitz.Document, page_index: int | None, bbox) -> str:
    """The text printed inside `bbox` - what the flagged region actually says."""
    if page_index is None or bbox is None or not (0 <= page_index < doc.page_count):
        return ""
    try:
        return doc[page_index].get_text("text", clip=fitz.Rect(*(float(v) for v in bbox)))
    except Exception:
        return ""


def _counterpart_content(
    doc: fitz.Document,
    start_page: int,
    end: tuple[int, float] | None,
    source_text: str,
    source_line_count: int,
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Where in this document's copy of the section the SAME content is printed.

    Scans the section's pages for the run of consecutive lines whose wording
    best matches the flagged region's, and returns that run's own box - so the
    orange counterpart box frames the same table, the same paragraph, the same
    row as the red one beside it. Matching content instead of position is what
    keeps a "table breaks the margin in Staging" finding from showing the
    Staging table beside whatever Production happens to print at that height.
    """
    want = _tokens(source_text)
    if sum(want.values()) < _CONTENT_MIN_TOKENS:
        return None
    lo = max(1, source_line_count - _CONTENT_LINE_SLACK)
    hi = max(lo, source_line_count + _CONTENT_LINE_SLACK)
    max_page = min(doc.page_count, start_page + _NEIGHBOUR_MAX_PAGES)
    if end is not None:
        max_page = min(max_page, end[0] + 1)

    best: tuple[int, int, int] | None = None
    best_score = 0.0
    per_page: dict[int, list] = {}
    for page_index in range(start_page, max_page):
        lines = [
            ln for ln in _page_lines(doc, page_index)
            if not (end and (page_index, ln[1]) >= end)
        ]
        per_page[page_index] = lines
        for i in range(len(lines)):
            acc: Counter = Counter()
            for j in range(i, min(len(lines), i + hi)):
                acc.update(_tokens(lines[j][4]))
                if j - i + 1 < lo:
                    continue
                score = _token_f1(acc, want)
                if score > best_score:
                    best_score, best = score, (page_index, i, j)
    if best is None or best_score < _CONTENT_MIN_SCORE:
        return None
    page_index, i, j = best
    window = per_page[page_index][i : j + 1]
    rect = doc[page_index].rect
    return page_index, (
        max(rect.x0, min(ln[0] for ln in window) - _CONTENT_PAD),
        max(rect.y0, min(ln[1] for ln in window) - _CONTENT_PAD),
        min(rect.x1, max(ln[2] for ln in window) + _CONTENT_PAD),
        min(rect.y1, max(ln[3] for ln in window) + _CONTENT_PAD),
    )


def _lead_in_line(doc: fitz.Document, page_index: int, bbox) -> str:
    """The last line of text directly above `bbox` - the thing a reader would
    use to say where the flagged region sat."""
    if not bbox:
        return ""
    top = float(bbox[1])
    best = ""
    for _, y0, _, y1, text in _page_lines(doc, page_index):
        if y1 <= top + 1 and y1 >= top - _NEIGHBOUR_LOOK_UP and len(text) >= _NEIGHBOUR_MIN_CHARS:
            best = text
    return best


def _figure_in_band(doc: fitz.Document, page_index: int, band: tuple) -> tuple | None:
    """The largest figure overlapping `band`, if any.

    A reader comparing a PICTURE wants the other document's picture, not the
    paragraph that happens to sit where the lead-in line landed. Framing the
    prose is why an "Image missing" finding showed a big Production diagram
    beside two lines of Staging text.
    """
    try:
        from pdfval.extractor import get_figures

        figures = get_figures(doc, page_index)
    except Exception:
        return None
    best, best_area = None, 0.0
    for fig in figures:
        x0, y0, x1, y1 = fig.bbox
        if x1 - x0 < _FIGURE_MIN_SIDE or y1 - y0 < _FIGURE_MIN_SIDE:
            continue
        if y1 <= band[1] or y0 >= band[3]:
            continue
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best, best_area = (x0, y0, x1, y1), area
    return best


def _section_figures(
    doc: fitz.Document, start_page: int, end: tuple[int, float] | None
) -> list[tuple[int, tuple]]:
    """Every figure in this document's copy of the section, in reading order."""
    try:
        from pdfval.extractor import get_figures
    except Exception:
        return []
    last = min(doc.page_count - 1, (end[0] if end else doc.page_count - 1), start_page + _NEIGHBOUR_MAX_PAGES)
    out: list[tuple[int, tuple]] = []
    for page_index in range(start_page, max(start_page, last) + 1):
        try:
            figures = get_figures(doc, page_index)
        except Exception:
            continue
        for fig in figures:
            x0, y0, x1, y1 = fig.bbox
            if x1 - x0 < _FIGURE_MIN_SIDE or y1 - y0 < _FIGURE_MIN_SIDE:
                continue
            if end and (page_index, y0) >= end:
                continue
            out.append((page_index, (x0, y0, x1, y1)))
    return out


def _counterpart_figure(
    doc: fitz.Document,
    start_page: int,
    end: tuple[int, float] | None,
    source_bbox,
    used: set,
) -> tuple[int, tuple] | None:
    """The figure in this document's copy of the section that best stands in for
    `source_bbox`.

    A figure finding is answered by the other document's FIGURE, so this looks
    for one directly instead of going through `_neighbour_anchor`'s lead-in
    text. Text anchoring is actively wrong here: the caption under a diagram in
    one document is printed above it in the other, so the band landed at the
    very end of the section and got clipped by the next heading to a two-line
    sliver. Matched on shape, and never handing the same figure to two findings
    in one section.
    """
    try:
        sx0, sy0, sx1, sy1 = (float(v) for v in source_bbox)
    except (TypeError, ValueError):
        return None
    src_area = max(1.0, (sx1 - sx0) * (sy1 - sy0))
    src_aspect = (sx1 - sx0) / max(1.0, sy1 - sy0)
    best, best_score = None, None
    for page_index, bbox in _section_figures(doc, start_page, end):
        if (page_index, tuple(round(v) for v in bbox)) in used:
            continue
        x0, y0, x1, y1 = bbox
        area = max(1.0, (x1 - x0) * (y1 - y0))
        aspect = (x1 - x0) / max(1.0, y1 - y0)
        score = abs(math.log(area / src_area)) + abs(math.log(max(0.05, aspect / max(0.05, src_aspect))))
        if best_score is None or score < best_score:
            best, best_score = (page_index, bbox), score
    if best is not None:
        used.add((best[0], tuple(round(v) for v in best[1])))
    return best


def _neighbour_anchor(
    doc: fitz.Document,
    start_page: int,
    lead_in: str,
    end: tuple[int, float] | None = None,
    source_height: float = 0.0,
) -> tuple[int, tuple[float, float, float, float]] | None:
    """`(page, bbox)` of where `lead_in` lands in this document - framed from
    that line down, which is where the missing region belongs.

    Searched across the section's whole page RANGE, not just the page its
    heading is on: the two documents paginate differently, and the line that
    sits one page after a heading in Production can be three pages after it
    here. Looking only at the heading's page is what made this fall back to
    boxing the entire section.

    `end`, when given, is the `(page, y)` of the NEXT heading in this document -
    a short, generic lead-in line (a numbered callout caption like "3. Power
    switch" is common across several near-identical panel diagrams in one
    manual) can otherwise match text that belongs to a different section
    entirely, landing the counterpart crop there instead. Candidates at or past
    that boundary are refused, and the returned band is clipped to it.
    """
    if not lead_in:
        return None
    want = _norm(lead_in)
    if len(want) < _NEIGHBOUR_MIN_CHARS:
        return None
    end_page = end[0] if end else None
    max_page = min(doc.page_count, start_page + _NEIGHBOUR_MAX_PAGES)
    if end_page is not None:
        max_page = min(max_page, end_page + 1)
    best, best_ratio = None, 0.0
    for page_index in range(start_page, max_page):
        for _, y0, _, y1, text in _page_lines(doc, page_index):
            if end and (page_index, y0) >= end:
                continue
            ratio = difflib.SequenceMatcher(None, want, _norm(text), autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, (page_index, y0, y1)
    if best is None or best_ratio < _NEIGHBOUR_MIN_RATIO:
        return None
    page_index, y0, y1 = best
    rect = doc[page_index].rect
    depth = min(_BAND_MAX, max(_NEIGHBOUR_BAND, source_height * _BAND_SOURCE_SCALE))
    band_y1 = min(rect.y1, y1 + depth)
    if end and end[0] == page_index:
        band_y1 = min(band_y1, end[1])
    # Box the space BELOW the lead-in, not the lead-in itself: this anchor is
    # only reached when the other document has nothing of its own here, and a
    # box drawn round the one line that IS there reads as "this text is the
    # defect". Only when that leaves nothing worth looking at does the line
    # come back inside the box.
    top = y1 + 2.0
    if band_y1 - top < _MIN_GAP_BAND:
        top = y0
    return page_index, (rect.x0, top, rect.x1, band_y1)


def _next_heading_bound(entries, title: str) -> tuple[int, float] | None:
    """`(page, y)` of the heading immediately after `title`, in true reading
    order - the far edge `_neighbour_anchor` must not cross."""
    from pdfval.validators.toc import normalize_title

    key = normalize_title(title or "")
    if not key or not entries:
        return None
    match = next((e for e in entries if normalize_title(e.title) == key), None)
    if match is None:
        return None
    ordered = sorted(entries, key=lambda e: (e.page, e.y))
    for i, e in enumerate(ordered):
        if e is match:
            return (ordered[i + 1].page, ordered[i + 1].y) if i + 1 < len(ordered) else None
    return None


# What the orange counterpart box is showing, per anchor used to place it.
_CAPTIONS = {
    "content": "{other} p.{page} — the same content here, shown for comparison; not itself a defect",
    "figure": "{other} p.{page} — the nearest figure here, shown for comparison; not itself a defect",
    "place": "{other} p.{page} — nothing of this sits here; the box marks where it belongs",
    "section": "{other} p.{page} — the same section (“{heading}”), the nearest the box could be placed",
}

# Detail keys that carry the flagged wording itself, in the order they are
# worth trying: what Production has, then what Staging has, then the text the
# finding quoted. Used when the finding has no region of its own to read.
_TEXT_KEYS = ("expected", "actual", "text")


def _finding_text(details: dict) -> str:
    """The flagged wording, straight off the finding, for findings that carry no
    bbox to read the page through (a content or callout difference)."""
    for key in _TEXT_KEYS:
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (list, tuple)):
            joined = " ".join(v for v in value if isinstance(v, str) and not v.startswith("... (+"))
            if joined.strip():
                return joined
    return ""


def fill_counterpart_screenshots(
    report, expected: fitz.Document, actual: fitz.Document, output_dir: str | None
) -> int:
    """Fill the empty side of every finding that has one screenshot, a heading,
    and a counterpart section in the other document. Returns how many were
    filled. Runs after every check, so it sees the findings exactly as the
    report will show them.
    """
    if not output_dir:
        return 0
    # Imported here rather than at module scope: this module is loaded from the
    # report package, and the resolver lives under `validators`.
    from pdfval.validators.headings import resolve_entries, section_band

    try:
        exp_entries, act_entries = resolve_entries(expected, actual)
    except Exception:
        return 0
    if not exp_entries or not act_entries:
        return 0

    counter = itertools.count(1)
    filled = 0
    # Two "Image missing" findings in one section must not both be answered
    # with the same Staging figure.
    figures_used: dict[tuple, set] = {}
    for check in getattr(report, "checks", []) or []:
        for issue in getattr(check, "issues", []) or []:
            details = issue.details or {}
            heading = details.get("heading")
            has_prod = bool(details.get("prod_screenshot"))
            has_stage = bool(details.get("stage_screenshot"))
            if not heading or has_prod == has_stage:
                continue
            side = "stage" if has_prod else "prod"
            doc = actual if side == "stage" else expected
            entries = act_entries if side == "stage" else exp_entries
            page, bbox = section_band(doc, entries, heading)
            if page is None:
                continue
            # Point at the SPOT, not at the section. The document that does have
            # the region knows what sits directly above it; finding that same
            # line over here frames the place the reader is actually asking
            # about, instead of boxing most of a page and calling it an answer.
            src_doc = expected if side == "stage" else actual
            src_page = issue.page if isinstance(issue.page, int) else None
            lead_in = (
                _lead_in_line(src_doc, src_page, details.get("bbox"))
                if src_page is not None and 0 <= src_page < src_doc.page_count
                else ""
            )
            # Frame an area the size of the thing being compared, not a fixed
            # strip: a 400pt diagram needs 400pt of the other document beside
            # it to mean anything.
            src_bbox = details.get("bbox")
            source_height = 0.0
            try:
                source_height = float(src_bbox[3]) - float(src_bbox[1])
            except (TypeError, IndexError, ValueError):
                pass
            bound = _next_heading_bound(entries, heading)
            precise = None
            how = "section"
            # A figure finding is answered by the other document's FIGURE, found
            # directly - never through the text around it.
            if check.name == "Image Validation" and src_bbox:
                precise = _counterpart_figure(
                    doc, page, bound, src_bbox, figures_used.setdefault((side, heading), set())
                )
                if precise:
                    how = "figure"
            if precise is None:
                # What the flagged region SAYS, found again over here. This is
                # the only anchor that guarantees the two boxes frame the same
                # thing: a table, a row, a paragraph is identified by its own
                # words, not by where it happens to sit on the page.
                source_text = _region_text(src_doc, src_page, src_bbox) or _finding_text(details)
                lines = [ln for ln in source_text.splitlines() if ln.strip()]
                precise = _counterpart_content(doc, page, bound, source_text, len(lines) or 1)
                if precise:
                    how = "content"
            if precise is None:
                precise = _neighbour_anchor(
                    doc, page, lead_in, end=bound, source_height=source_height
                )
                if precise:
                    how = "place"
                    if check.name == "Image Validation":
                        figure = _figure_in_band(doc, precise[0], precise[1])
                        if figure:
                            precise, how = (precise[0], figure), "figure"
            if precise:
                page, bbox = precise
            seq = next(counter)
            # The SAME number the finding's own crop carries - a counterpart
            # box numbered differently from the box it partners defeats the
            # whole point of numbering them.
            label = str(details.get("shot_label") or seq)
            path = screenshots.capture_region(
                doc, page, output_dir, f"counterpart_{seq}_{side}", bbox,
                screenshots.KIND_CONTEXT, label,
            )
            if not path:
                continue
            other = "Staging" if side == "stage" else "Production"
            details[f"{side}_screenshot"] = path
            # Say what the orange box IS. Without this the reader sees a box
            # round a paragraph opposite a box round a diagram and reads it as
            # a second, text-level defect - which is exactly what the box is
            # not: it is where to look in the other document.
            details[f"{side}_screenshot_caption"] = _CAPTIONS[how].format(
                other=other, page=page + 1, heading=heading
            )
            details[f"{side}_screenshot_kind"] = how
            # The old "there is no corresponding place to show" placeholder is
            # no longer true for this finding.
            details.pop(f"{side}_screenshot_note", None)
            issue.details = details
            filled += 1
    return filled
