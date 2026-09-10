"""Renders a real PDF page as a PNG with a red box around a specific region,
so a content issue can be visually verified against the actual page layout.
"""
from __future__ import annotations

import os

import fitz
from PIL import Image, ImageDraw

DPI = 150
PADDING = 4.0  # points of padding around the highlighted region
BOX_COLOR = (220, 20, 20)
BOX_WIDTH = 4  # px

CROP_DPI = 200  # close-ups are rendered finer than the full page
CROP_MARGIN = 24.0  # points of surrounding page kept around a cropped region

SCREENSHOTS_SUBDIR = "screenshots"

# When a figure's detected bbox covers this much of the page, it is almost
# certainly a mis-detection that swallowed surrounding prose/whitespace - a
# close-up zoomed to it is just the whole page. Before falling back to that,
# the region is re-tightened to the actual drawn content inside it.
_TIGHTEN_PAGE_FRACTION = 0.45
_TIGHTEN_PROBE_DPI = 72  # low-res probe is plenty to find where the ink is
_TIGHTEN_WHITE = 244  # a sample at least this bright counts as background
_TIGHTEN_MIN_SIDE = 24.0  # never tighten to something smaller than this (points)


def tighten_bbox(
    doc: fitz.Document, page_index: int, bbox: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Public wrapper: pull an over-large figure bbox back to its artwork.
    Used by callers that need to decide whether a region is worth capturing."""
    try:
        return _tighten_to_ink(doc[page_index], bbox)
    except Exception:
        return bbox


def _tighten_to_ink(
    page: fitz.Page, bbox: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Shrink an over-large figure `bbox` (PDF point space) to the region where
    the ARTWORK actually is - the embedded images and vector drawing inside it,
    with the surrounding body text discounted.

    A pixel probe alone can't do this: body text is ink too, so an illustration
    with a paragraph above and below it would "tighten" to the whole block. So
    the figure's real extent is taken from the page's own image and drawing
    rectangles that fall inside `bbox`, and the pixel probe is only a fallback.
    Returns the original bbox unchanged when it is already tight or nothing
    figure-like can be found.
    """
    page_rect = page.rect
    rect = fitz.Rect(*bbox) & page_rect
    if rect.is_empty or rect.width < _TIGHTEN_MIN_SIDE or rect.height < _TIGHTEN_MIN_SIDE:
        return bbox
    if (rect.width * rect.height) < _TIGHTEN_PAGE_FRACTION * page_rect.width * page_rect.height:
        return bbox

    art = _artwork_extent(page, rect)
    if art is not None and art.width >= _TIGHTEN_MIN_SIDE and art.height >= _TIGHTEN_MIN_SIDE:
        pad = 8.0
        out = fitz.Rect(art.x0 - pad, art.y0 - pad, art.x1 + pad, art.y1 + pad) & page_rect
        if (out.width * out.height) < 0.95 * rect.width * rect.height:
            return (out.x0, out.y0, out.x1, out.y1)
    return bbox


def _artwork_extent(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect | None:
    """Union of the image placements and vector-drawing rectangles that lie
    (mostly) inside `rect` - i.e. where the figure's artwork sits, ignoring the
    prose around it."""
    boxes: list[fitz.Rect] = []
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info.get("bbox", (0, 0, 0, 0)))
            if not r.is_empty and r.width > 2 and r.height > 2 and rect.intersects(r):
                boxes.append(r & rect)
    except Exception:
        pass
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r is None or r.is_empty:
                continue
            # A hairline is a rule or a table border, not artwork.
            if r.width < 3 and r.height < 3:
                continue
            if rect.intersects(r):
                boxes.append(fitz.Rect(r) & rect)
    except Exception:
        pass
    boxes = [b for b in boxes if not b.is_empty and b.width > 2 and b.height > 2]
    if not boxes:
        return None
    out = boxes[0]
    for b in boxes[1:]:
        out |= b
    return out


def _render(page: fitz.Page, dpi: int, clip: fitz.Rect | None = None) -> tuple[Image.Image, float]:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, clip=clip)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), zoom


def capture_page(
    doc: fitz.Document,
    page_index: int,
    output_dir: str,
    name: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> str | None:
    """Render `page_index` from `doc` to a PNG under `output_dir/screenshots/`,
    drawing a red box around `bbox` (in PDF point space) if given. Returns the
    path relative to `output_dir` (for use as an <img src>), or None on failure.
    """
    if page_index is None or not (0 <= page_index < doc.page_count):
        return None
    try:
        page = doc[page_index]
        img, zoom = _render(page, DPI)

        if bbox is not None:
            _draw_box(img, page, bbox, zoom, origin=(page.rect.x0, page.rect.y0))

        return _save(img, output_dir, name)
    except Exception:
        return None


def capture_region(
    doc: fitz.Document,
    page_index: int,
    output_dir: str,
    name: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> str | None:
    """Like `capture_page`, but zoomed in on `bbox` (plus a margin of
    surrounding page for context) instead of showing the whole page. A figure
    or table occupying a fifth of an A4 page is unreadable in a full-page
    thumbnail, which is what made image/table findings impossible to verify.
    Falls back to the full page when there is no bbox to zoom to.
    """
    if page_index is None or not (0 <= page_index < doc.page_count):
        return None
    if bbox is None:
        return capture_page(doc, page_index, output_dir, name, None)
    try:
        page = doc[page_index]
        page_rect = page.rect
        # An over-large figure bbox (a mis-detection that swallowed prose) makes
        # a "close-up" that is just the whole page. Pull it back to the content
        # actually drawn inside it first, so the reader sees the figure.
        bbox = _tighten_to_ink(page, bbox)
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(
            max(page_rect.x0, x0 - CROP_MARGIN),
            max(page_rect.y0, y0 - CROP_MARGIN),
            min(page_rect.x1, x1 + CROP_MARGIN),
            min(page_rect.y1, y1 + CROP_MARGIN),
        )
        if clip.is_empty or clip.width < 1 or clip.height < 1:
            return capture_page(doc, page_index, output_dir, name, bbox)

        img, zoom = _render(page, CROP_DPI, clip=clip)
        _draw_box(img, page, bbox, zoom, origin=(clip.x0, clip.y0))
        return _save(img, output_dir, name)
    except Exception:
        return capture_page(doc, page_index, output_dir, name, bbox)


def _draw_box(
    img: Image.Image,
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    zoom: float,
    origin: tuple[float, float],
) -> None:
    """Draw the red highlight. Pixel (0,0) of a render corresponds to the
    top-left of the *rendered area*, not to PDF point (0,0) - so the render
    origin has to be subtracted, or the box lands offset on any page whose
    MediaBox does not start at the origin, and on every cropped render.
    """
    ox, oy = origin
    x0, y0, x1, y1 = bbox
    rect = (
        max(0, (x0 - PADDING - ox) * zoom),
        max(0, (y0 - PADDING - oy) * zoom),
        min(img.width, (x1 + PADDING - ox) * zoom),
        min(img.height, (y1 + PADDING - oy) * zoom),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return
    ImageDraw.Draw(img).rectangle(rect, outline=BOX_COLOR, width=BOX_WIDTH)


def _save(img: Image.Image, output_dir: str, name: str) -> str:
    screenshots_dir = os.path.join(output_dir, SCREENSHOTS_SUBDIR)
    os.makedirs(screenshots_dir, exist_ok=True)
    filename = f"{name}.png"
    img.save(os.path.join(screenshots_dir, filename))
    return os.path.join(SCREENSHOTS_SUBDIR, filename)
