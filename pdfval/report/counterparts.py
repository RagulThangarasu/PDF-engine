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


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _page_lines(doc: fitz.Document, page_index: int) -> list[tuple[float, float, str]]:
    """(y0, y1, text) for every text line on the page, in reading order."""
    out: list[tuple[float, float, str]] = []
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
                out.append((float(ln["bbox"][1]), float(ln["bbox"][3]), text))
    out.sort()
    return out


def _lead_in_line(doc: fitz.Document, page_index: int, bbox) -> str:
    """The last line of text directly above `bbox` - the thing a reader would
    use to say where the flagged region sat."""
    if not bbox:
        return ""
    top = float(bbox[1])
    best = ""
    for y0, y1, text in _page_lines(doc, page_index):
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
        for y0, y1, text in _page_lines(doc, page_index):
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
    return page_index, (rect.x0, y0, rect.x1, band_y1)


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
            # A figure finding is answered by the other document's FIGURE, found
            # directly - never through the text around it.
            if check.name == "Image Validation" and src_bbox:
                precise = _counterpart_figure(
                    doc, page, bound, src_bbox, figures_used.setdefault((side, heading), set())
                )
            if precise is None:
                precise = _neighbour_anchor(
                    doc, page, lead_in, end=bound, source_height=source_height
                )
                if precise and check.name == "Image Validation":
                    figure = _figure_in_band(doc, precise[0], precise[1])
                    if figure:
                        precise = (precise[0], figure)
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
            details[f"{side}_screenshot_caption"] = (
                f"{other} - p.{page + 1}, where this belongs" if precise
                else f"{other} - p.{page + 1}, the same section ({heading})"
            )
            # The old "there is no corresponding place to show" placeholder is
            # no longer true for this finding.
            details.pop(f"{side}_screenshot_note", None)
            issue.details = details
            filled += 1
    return filled
