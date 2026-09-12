"""Helpers for pulling structured data out of a PDF using PyMuPDF/pdfplumber."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF


@dataclass
class ImageInfo:
    xref: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    kind: str = "raster"  # "raster" (embedded image XObject) or "vector" (drawn figure)

    @property
    def display_width(self) -> float:
        """On-page width in points. This - not `width`, which is the source
        raster's pixel width - is what decides whether a figure is visually
        significant: a 60x60px logo stretched across half a page matters, and a
        512x512px icon drawn at 8pt does not.
        """
        return self.bbox[2] - self.bbox[0]

    @property
    def display_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def display_area(self) -> float:
        return self.display_width * self.display_height


def page_count(doc: fitz.Document) -> int:
    return doc.page_count


def get_page_text(doc: fitz.Document, page_index: int) -> str:
    return doc[page_index].get_text("text")


def _round_bbox(bbox: tuple[float, float, float, float], ndigits: int = 0) -> tuple:
    return tuple(round(v, ndigits) for v in bbox)


def get_images(doc: fitz.Document, page_index: int) -> list[ImageInfo]:
    """Every embedded raster image placed on the page.

    The same image object placed twice legitimately yields two entries, but a
    single placement can also be reported more than once (an image plus its
    soft-mask/stencil companion drawn at the identical spot), so identical
    xref+position pairs are collapsed.
    """
    page = doc[page_index]
    infos: list[ImageInfo] = []
    seen: set[tuple] = set()
    for info in page.get_image_info(xrefs=True):
        bbox = tuple(info.get("bbox", (0, 0, 0, 0)))
        key = (info.get("xref", -1), _round_bbox(bbox))
        if key in seen:
            continue
        seen.add(key)
        infos.append(
            ImageInfo(
                xref=info.get("xref", -1),
                bbox=bbox,
                width=info.get("width", 0),
                height=info.get("height", 0),
                kind="raster",
            )
        )
    return infos


# A vector figure is assembled from many small drawing primitives (the strokes
# and fills of an illustration). These thresholds decide when a cluster of them
# amounts to a real figure rather than a rule, an underline, a table border or a
# background tint.
_VECTOR_MIN_SIDE = 60.0  # points - a figure is at least this wide AND tall
_VECTOR_MIN_AREA = 6000.0  # points^2
_VECTOR_MIN_PRIMITIVES = 4  # a lone rectangle is a box/tint, not an illustration
_VECTOR_CLUSTER_GAP = 12.0  # points - primitives at most this far apart are one figure
_VECTOR_MAX_PAGE_FRACTION = 0.85  # a "figure" covering the whole page is a background


def _rects_touch(a: tuple, b: tuple, gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _union(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _cluster_rects(rects: list[tuple], gap: float) -> list[tuple[tuple, int]]:
    """Greedily merge overlapping/nearby rectangles into clusters, repeating
    until nothing merges further. Returns (bbox, primitive_count) per cluster.
    """
    clusters: list[list] = [[r, 1] for r in rects]
    merged = True
    while merged:
        merged = False
        out: list[list] = []
        for bbox, count in clusters:
            for other in out:
                if _rects_touch(bbox, other[0], gap):
                    other[0] = _union(other[0], bbox)
                    other[1] += count
                    merged = True
                    break
            else:
                out.append([bbox, count])
        clusters = out
    return [(tuple(bbox), count) for bbox, count in clusters]


def get_vector_figures(doc: fitz.Document, page_index: int) -> list[ImageInfo]:
    """Illustrations drawn with vector primitives rather than embedded as a
    raster image - on-screen-display mockups, panel/port diagrams, connection
    schematics. `get_image_info` cannot see these at all, so without this a
    manual whose figures are all vector art looks to Image Validation like a
    document containing no images.
    """
    page = doc[page_index]
    page_rect = page.rect
    page_area = max(1.0, page_rect.width * page_rect.height)

    rects: list[tuple] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        # Hairline rules, underlines and table borders are effectively 1-D.
        if r.width < 2 and r.height < 2:
            continue
        rects.append((r.x0, r.y0, r.x1, r.y1))

    if not rects:
        return []

    figures: list[ImageInfo] = []
    for bbox, count in _cluster_rects(rects, _VECTOR_CLUSTER_GAP):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if count < _VECTOR_MIN_PRIMITIVES:
            continue
        if w < _VECTOR_MIN_SIDE or h < _VECTOR_MIN_SIDE or w * h < _VECTOR_MIN_AREA:
            continue
        if (w * h) / page_area > _VECTOR_MAX_PAGE_FRACTION:
            continue
        figures.append(
            ImageInfo(xref=-1, bbox=bbox, width=int(w), height=int(h), kind="vector")
        )
    return figures


def get_figures(doc: fitz.Document, page_index: int) -> list[ImageInfo]:
    """Everything on the page that reads as a figure: embedded raster images
    plus vector-drawn illustrations.

    A vector cluster is dropped when it merely traces an already-detected
    raster image (so the same figure isn't counted twice), OR when it sits on
    top of a real data table - a ruled table is a grid of rectangles and lines,
    exactly what the vector-figure detector looks for, so without this a
    bordered "Commonly used names for FHD/4K" table gets picked up as an
    illustration and then compared, figure-to-figure, against whatever the
    other document has in that slot. `get_tables` has already filtered out
    UI-mockup screenshots, so what it returns here is genuine tabular content
    that is not a figure.
    """
    rasters = get_images(doc, page_index)
    table_bboxes = _page_table_bboxes(doc, page_index)
    vectors = []
    for fig in get_vector_figures(doc, page_index):
        if any(_overlap_fraction(fig.bbox, r.bbox) > 0.6 for r in rasters):
            continue
        # Covered by the tables on the page - one table, or (as with a stacked
        # pair of spec tables sharing a border) several that together fill the
        # cluster. A single-table test misses the stacked case: neither table
        # alone covers half the combined bbox, but between them they cover all
        # of it.
        if _covered_fraction(fig.bbox, table_bboxes) > 0.6:
            continue
        vectors.append(fig)
    return rasters + vectors


def _covered_fraction(fig_bbox: tuple, boxes: list[tuple]) -> float:
    """Fraction of `fig_bbox` covered by the union of `boxes`. Table bboxes on
    a page don't overlap each other, so summing the per-box intersections gives
    the union area exactly; the cap guards the general case."""
    fig_area = max(1e-6, (fig_bbox[2] - fig_bbox[0]) * (fig_bbox[3] - fig_bbox[1]))
    covered = 0.0
    for b in boxes:
        ix0, iy0 = max(fig_bbox[0], b[0]), max(fig_bbox[1], b[1])
        ix1, iy1 = min(fig_bbox[2], b[2]), min(fig_bbox[3], b[3])
        if ix1 > ix0 and iy1 > iy0:
            covered += (ix1 - ix0) * (iy1 - iy0)
    return min(1.0, covered / fig_area)


def _page_table_bboxes(doc: fitz.Document, page_index: int) -> list[tuple]:
    """Bounding boxes of the genuine data tables on this page, via the same
    pdfplumber pass `get_tables` uses. Best-effort: a doc not backed by a file
    on disk (no `.name`) yields nothing rather than an error.
    """
    path = getattr(doc, "name", "") or ""
    if not path or not os.path.isfile(path):
        return []
    try:
        return [t["bbox"] for t in get_tables(path, page_index, doc)]
    except Exception:
        return []


def _overlap_fraction(a: tuple, b: tuple) -> float:
    """Fraction of `a`'s area that lies inside `b`."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    a_area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ((ix1 - ix0) * (iy1 - iy0)) / a_area


# A single real table cell occasionally wraps to a few lines (a wrapped
# description, or a short bulleted sub-list), but this many embedded line
# breaks in ONE cell means several distinct, unrelated menu items got
# merged together - a strong sign pdfplumber mistook a vector-drawn UI
# mockup (e.g. an on-screen-display menu illustration made of rectangles/
# lines, not a raster image) for a real data table.
_MAX_CELL_NEWLINES = 5


def _has_invalid_cell_geometry(cells: list | None) -> bool:
    """When pdfplumber can't find a row/column boundary between two unrelated
    blocks of text, it merges them into one phantom cell whose rect is
    extrapolated off the page (a negative x0/y0) rather than bounded by any
    real ruling - a strong, unambiguous corruption signal since real page
    content is never positioned outside the page's own coordinate space.
    Whatever the region is, its shape (and any text merged into that phantom
    cell) can't be trusted for any purpose, table or excluded-region alike.
    """
    for row in cells or []:
        for cell in row or []:
            if cell and (cell[0] < 0 or cell[1] < 0):
                return True
    return False


def _column_count(rows: list) -> int:
    return max((len(row or []) for row in rows or []), default=0)


def is_single_column_region(rows: list) -> bool:
    """A detected "table" with one column is a bordered BOX, not tabular data -
    a Note/Important/Warning admonition, a tinted callout panel, an icon frame.

    This matters far more than it sounds. Whether a box like that is ruled with
    strokes pdfplumber's line detector can see is purely a property of the
    producer: on the pair this was built against, Production's admonition boxes
    came back as 36 single-column "tables" and Staging's identical ones as 2.
    Every one of those made Content Validation drop the box's prose from
    Production's side of the diff while keeping Staging's - so text printed
    verbatim in BOTH documents was reported as "not in Production". Tabular
    data has at least two columns; anything narrower is prose in a frame and is
    compared as prose.
    """
    return _column_count(rows) < 2


def _looks_like_real_table(rows: list, cells: list | None = None) -> bool:
    has_text = False
    for row in rows or []:
        for cell in row or []:
            if cell and cell.count("\n") > _MAX_CELL_NEWLINES:
                return False
            if cell and cell.strip():
                has_text = True
    # pdfplumber's grid/line detector sometimes finds a purely decorative
    # bordered box (an icon frame, a diagram outline) with no cell text at
    # all - that's not a data table either.
    if not has_text:
        return False
    if is_single_column_region(rows):
        return False
    return not _has_invalid_cell_geometry(cells)


def _strip_blank_edge_columns(rows: list, cells: list) -> tuple[list, list]:
    """A leading or trailing column with no text in ANY row is a pdfplumber
    column-boundary artifact (a stray rule detected as a column edge with
    nothing in it) rather than a real column - left in, it makes an
    otherwise identical table on the other side read as having a different
    column count.
    """
    if not rows or not rows[0]:
        return rows, cells
    n_cols = len(rows[0])
    if n_cols < 2:
        return rows, cells

    def col_blank(idx: int) -> bool:
        return all(not (row[idx] or "").strip() for row in rows if idx < len(row))

    start, end = 0, n_cols
    while end - start > 1 and col_blank(start):
        start += 1
    while end - start > 1 and col_blank(end - 1):
        end -= 1
    if start == 0 and end == n_cols:
        return rows, cells
    new_rows = [row[start:end] for row in rows]
    new_cells = [row[start:end] for row in cells] if cells else cells
    return new_rows, new_cells


class _PageTableCache:
    """pdfplumber has to parse the whole document to open it, so calling
    `pdfplumber.open` once per page - which the validators do, repeatedly, for
    both documents - dominated the run time on a manual of any size. One open
    per file, results memoized per page.
    """

    def __init__(self) -> None:
        self._docs: dict[str, Any] = {}
        self._pages: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def _doc(self, pdf_path: str):
        if pdf_path not in self._docs:
            import pdfplumber

            self._docs[pdf_path] = pdfplumber.open(pdf_path)
        return self._docs[pdf_path]

    def page_regions(self, pdf_path: str, page_index: int) -> list[dict[str, Any]]:
        key = (pdf_path, page_index)
        if key not in self._pages:
            pdf = self._doc(pdf_path)
            if page_index < 0 or page_index >= len(pdf.pages):
                self._pages[key] = []
            else:
                found = []
                try:
                    for t in pdf.pages[page_index].find_tables():
                        rows = t.extract()
                        cells = [list(row.cells) for row in t.rows]
                        rows, cells = _strip_blank_edge_columns(rows, cells)
                        found.append(
                            {
                                "bbox": t.bbox,
                                "rows": rows,
                                # Per-row cell rectangles (None where a cell is
                                # merged away), which is what the column- and
                                # cell-layout comparisons need - the extracted
                                # text alone says nothing about geometry.
                                "cells": cells,
                                "is_table": _looks_like_real_table(rows, cells),
                            }
                        )
                except Exception:
                    found = []
                self._pages[key] = found
        return self._pages[key]

    def close(self) -> None:
        for pdf in self._docs.values():
            try:
                pdf.close()
            except Exception:
                pass
        self._docs.clear()
        self._pages.clear()


_TABLE_CACHE = _PageTableCache()


def reset_table_cache() -> None:
    """Drop cached pdfplumber handles; call between validation runs so a
    long-lived process (the web server) doesn't hold every uploaded PDF open.
    """
    _TABLE_CACHE.close()
    _mockup_cache.clear()


# A dark panel covering this much of a detected "table" means the region is a
# rendered UI mockup - an on-screen-display menu illustration - not a data
# table. Real tables in these manuals sit on white or a light tint and are
# ruled with thin strokes; an OSD screenshot is a dark filled panel with text
# on top, which pdfplumber's line detector reads as a perfect grid.
_MOCKUP_MAX_LUMINANCE = 0.35
_MOCKUP_MIN_COVERAGE = 0.4

_mockup_cache: dict[tuple, bool] = {}


def _looks_like_ui_mockup(doc: "fitz.Document", page_index: int, bbox: tuple) -> bool:
    key = (id(doc), page_index, _round_bbox(bbox, 1))
    if key not in _mockup_cache:
        _mockup_cache[key] = _measure_dark_coverage(doc, page_index, bbox) >= _MOCKUP_MIN_COVERAGE
    return _mockup_cache[key]


def _measure_dark_coverage(doc: "fitz.Document", page_index: int, bbox: tuple) -> float:
    """The largest fraction of `bbox` covered by a single dark filled shape."""
    try:
        rect = fitz.Rect(*bbox)
        drawings = doc[page_index].get_drawings()
    except Exception:
        return 0.0
    area = max(1.0, rect.width * rect.height)
    best = 0.0
    for d in drawings:
        if d.get("type") not in ("f", "fs"):
            continue
        fill = d.get("fill")
        if not fill or len(fill) < 3:
            continue
        luminance = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        if luminance >= _MOCKUP_MAX_LUMINANCE:
            continue
        inter = fitz.Rect(d["rect"]) & rect
        if inter.is_empty:
            continue
        best = max(best, (inter.width * inter.height) / area)
    return best


def get_tables(
    pdf_path: str, page_index: int, doc: "fitz.Document | None" = None
) -> list[dict[str, Any]]:
    """Extract tables (with bboxes) for a single page using pdfplumber.

    When `doc` is supplied, regions that are really UI mockups are dropped.
    This matters because the two documents don't render their on-screen-display
    illustrations the same way: one draws them as vector panels, which
    pdfplumber happily reads as a 3-row table, while the other embeds them as
    raster images, which it cannot see at all. That asymmetry gave Production
    eleven tables Staging could never have, and every one of them became either
    a bogus "table is gone" or - worse - got matched against a real table and
    reported its entire contents as changed.
    """
    return [
        {"bbox": r["bbox"], "rows": r["rows"], "cells": r["cells"]}
        for r in _TABLE_CACHE.page_regions(pdf_path, page_index)
        if r["is_table"] and not (doc is not None and _looks_like_ui_mockup(doc, page_index, r["bbox"]))
    ]


def _tight_content_bbox(region: dict) -> tuple[float, float, float, float]:
    """The table's overall bbox from pdfplumber can extend past its last real
    row of content - a trailing blank grid row (no text in any cell) still
    counts towards the bbox, and if a caption/note sits directly under the
    table sharing that padding, its text falls inside the table's bbox and
    gets wrongly excluded from Content Validation as "table text". Use the
    union of only the cell rectangles from rows that actually have text in at
    least one cell, so a blank trailing row doesn't drag the bottom edge down
    past the real content.
    """
    x0 = y0 = x1 = y1 = None
    rows, cells = region.get("rows") or [], region.get("cells") or []
    for i, row_cells in enumerate(cells):
        text_row = rows[i] if i < len(rows) else []
        if not any((t or "").strip() for t in text_row):
            continue
        for cell in row_cells or []:
            if not cell:
                continue
            cx0, cy0, cx1, cy1 = cell
            x0 = cx0 if x0 is None else min(x0, cx0)
            y0 = cy0 if y0 is None else min(y0, cy0)
            x1 = cx1 if x1 is None else max(x1, cx1)
            y1 = cy1 if y1 is None else max(y1, cy1)
    if x0 is None:
        return region["bbox"]
    return (x0, y0, x1, y1)


def get_all_detected_regions(pdf_path: str, page_index: int) -> list[tuple[float, float, float, float]]:
    """All grid-like regions pdfplumber detects on a page, including ones
    `get_tables` rejects as mis-detected UI mockups. Content Validation uses
    this (not `get_tables`) to decide what to exclude from prose diffing - a
    vector-drawn menu mockup isn't real tabular data OR real flowing prose,
    so its scattered text shouldn't be sentence-diffed as content either.

    A region with invalid (off-page) cell geometry is excluded outright
    rather than tight-bboxed - it isn't a mockup either, it's pdfplumber
    merging unrelated prose into a phantom cell, and that prose still needs
    to be diffed as content, not swallowed by the phantom cell's bogus bbox.
    So is a single-column region: that is a bordered Note/Warning box whose
    text is ordinary prose, and only one of the two documents usually draws it
    with strokes pdfplumber can see (see `is_single_column_region`).
    """
    return [
        _tight_content_bbox(r)
        for r in _TABLE_CACHE.page_regions(pdf_path, page_index)
        if not _has_invalid_cell_geometry(r.get("cells"))
        and not is_single_column_region(r.get("rows"))
    ]
