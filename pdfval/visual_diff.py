"""Pictures compared as they print, with OpenCV.

Two questions the figure checks cannot answer from detected boxes alone:

* Is this artwork printed on the other side at all? One document draws a
  diagram as several vector pieces where the other embeds it as one image,
  crops it differently, or prints it at another size - the detected boxes
  never line up, so a piece reads as "missing" and the whole as "oversized".
  `find_artwork` looks for the artwork itself on the other side's rendered
  page, at half to double its size, and confirms each candidate at a fixed
  resolution on grey levels and edges both, so a small shape that merely
  resembles something on the page is not taken for it.
* Where inside a matched figure do the two render differently?
  `differing_regions` aligns the two renders and boxes the spots that differ -
  on grey level (shape, edges, brightness) AND, separately, on hue, so a
  figure recoloured but otherwise pixel-for-pixel identical - a brand colour
  swapped, a warning icon's red turned orange - is still caught. A grey-level
  diff alone is blind to that: luminance does not change when only the colour
  does.
"""
from __future__ import annotations

import fitz

RENDER_EDGE = 320            # px on the long edge both figures are compared at
DIFF_LEVEL = 60              # grey-level difference (0-255) that counts at a pixel
MIN_REGION_SHARE = 0.004     # a region smaller than this share of the figure is anti-aliasing noise
MAX_CHANGED_SHARE = 0.35     # more than this differs: another picture or crop, not local spots
MAX_REGIONS = 6
# A pair can match perfectly in shape, edges and brightness and still be a
# different colour outright - a brand blue recoloured green, a warning icon's
# red swapped for orange - which a grey-level diff is blind to by definition.
# Compared separately, in colour, at the same registration the grey pass
# already solved: OpenCV hue runs 0-179 for 0-360 degrees, so 18 is a 36-degree
# swing - plainly a different colour, not the same hue rendered a shade off by
# two documents' different colour profiles. Saturation is compared too, and
# only the lower of the pair's two values decides whether a pixel counts:
# near-grey/white/black in EITHER render carries no reliable hue in the first
# place, on either side, and is never flagged as a colour change there.
COLOUR_HUE_LEVEL = 18
COLOUR_MIN_CHROMA = 40

_SOURCE_ZOOM = 4.0           # the artwork being looked for is rendered this sharp
_SEARCH_PX = 160             # ... and searched for at about this many pixels on its long edge
_ZOOMS = (0.3, 0.45, 0.65, 1.0, 1.5, 2.2)  # page renders are cached at these zooms only
_SCALES = 25                 # sizes tried between half and double
_MIN_TEMPLATE_PX = 16
_COARSE_FLOOR = 0.35
_CANDIDATES = 5
_VERIFY_PX = 96
_VERIFY_GREY = 0.65          # calibrated on a real manual pair: same artwork scored >= 0.72 grey and
_VERIFY_EDGES = 0.5          # >= 0.56 edges; look-alikes never both (at most 0.74 grey with 0.15 edges)
_PAGE_CACHE: dict[tuple, object] = {}
_PAGE_CACHE_MAX = 48


def available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def reset_cache() -> None:
    _PAGE_CACHE.clear()


def _area(r: "fitz.Rect") -> float:
    """A rect's area without `Rect.get_area()` - not on every PyMuPDF release
    a bare `PyMuPDF>=X` floor in requirements.txt still installs; `width`/
    `height` are on every version there has ever been."""
    return r.width * r.height


def _gray(doc: fitz.Document, page_index: int, clip, zoom: float):
    import numpy as np

    try:
        page = doc[page_index]
        rect = (fitz.Rect(clip) & page.rect) if clip is not None else page.rect
        if rect.is_empty or rect.width < 1 or rect.height < 1:
            return None
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False, colorspace=fitz.csGRAY)
    except Exception:
        return None
    if pix.width < 2 or pix.height < 2:
        return None
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width].copy()


def _page_gray(doc: fitz.Document, page_index: int, zoom: float):
    import cv2

    key = (id(doc), page_index, zoom)
    if key not in _PAGE_CACHE:
        if len(_PAGE_CACHE) >= _PAGE_CACHE_MAX:
            _PAGE_CACHE.clear()
        img = _gray(doc, page_index, None, zoom)
        _PAGE_CACHE[key] = None if img is None else cv2.GaussianBlur(img, (3, 3), 0)
    return _PAGE_CACHE[key]


def _ink(img):
    import numpy as np

    ys, xs = np.where(img < 235)
    if len(xs) == 0:
        return None
    return img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _edges(img):
    import cv2
    import numpy as np

    e = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 60, 160)
    return cv2.GaussianBlur(cv2.dilate(e, np.ones((3, 3), np.uint8)), (5, 5), 0)


def _ncc(big, small) -> tuple[float, tuple[int, int]]:
    import cv2

    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED))
    return float(score), loc


def _verified(doc: fitz.Document, page_index: int, rect: fitz.Rect, art) -> tuple[float, float]:
    """Grey-level and edge agreement of `art` with what `doc` prints at `rect`,
    both at a fixed resolution, with a few percent of size and place slack."""
    import cv2

    pad_w, pad_h = rect.width * 0.08, rect.height * 0.08
    zoom = _VERIFY_PX / max(rect.width, rect.height)
    region = _gray(doc, page_index, fitz.Rect(rect.x0 - pad_w, rect.y0 - pad_h, rect.x1 + pad_w, rect.y1 + pad_h), zoom)
    if region is None:
        return 0.0, 0.0
    region_grey, region_edges = cv2.GaussianBlur(region, (5, 5), 0), _edges(region)
    best = (0.0, 0.0)
    for slack in (0.94, 1.0, 1.06):
        tw, th = int(rect.width * zoom * slack), int(rect.height * zoom * slack)
        if min(tw, th) < 8 or tw > region.shape[1] or th > region.shape[0]:
            continue
        t = cv2.resize(art, (tw, th), interpolation=cv2.INTER_AREA)
        grey = _ncc(region_grey, cv2.GaussianBlur(t, (5, 5), 0))[0]
        edges = _ncc(region_edges, _edges(t))[0]
        if grey + edges > sum(best):
            best = (grey, edges)
    return best


def find_artwork(src_doc: fitz.Document, src_page: int, src_rect: tuple,
                 dst_doc: fitz.Document, regions: list[tuple[int, fitz.Rect]],
                 accept=None, min_side: float = 0.0) -> tuple[int, tuple] | None:
    """(page, bbox) where the artwork printed at `src_rect` is printed inside
    `regions` ([(page, rect)] of `dst_doc`) - drawn or embedded, cropped
    differently, at half to double its size, or as one piece of a bigger
    picture - or None. `accept(page, rect)` can veto a place (another topic);
    artwork whose inked extent is under `min_side` points is not looked for."""
    if not available() or not regions:
        return None
    import cv2
    import numpy as np

    src = _gray(src_doc, src_page, src_rect, _SOURCE_ZOOM)
    art = None if src is None else _ink(src)
    if art is None or min(art.shape) < 8 or min(art.shape) / _SOURCE_ZOOM < min_side or art.std() < 10:
        return None
    wanted = _SEARCH_PX / (max(art.shape) / _SOURCE_ZOOM)
    zoom = min(_ZOOMS, key=lambda z: abs(z - wanted))

    candidates: list[tuple[float, int, fitz.Rect]] = []
    for page_index, rect in regions:
        page = _page_gray(dst_doc, page_index, zoom)
        if page is None:
            continue
        origin = dst_doc[page_index].rect
        clip = fitz.Rect(rect) & origin
        x0, y0 = int((clip.x0 - origin.x0) * zoom), int((clip.y0 - origin.y0) * zoom)
        x1, y1 = int((clip.x1 - origin.x0) * zoom), int((clip.y1 - origin.y0) * zoom)
        crop = page[max(0, y0): y1, max(0, x0): x1]
        for scale in np.geomspace(0.5, 2.0, _SCALES):
            w = int(art.shape[1] * scale * zoom / _SOURCE_ZOOM)
            h = int(art.shape[0] * scale * zoom / _SOURCE_ZOOM)
            if min(w, h) < _MIN_TEMPLATE_PX or w > crop.shape[1] or h > crop.shape[0]:
                continue
            template = cv2.GaussianBlur(cv2.resize(art, (w, h), interpolation=cv2.INTER_AREA), (3, 3), 0)
            if template.std() < 10:
                continue
            score, (lx, ly) = _ncc(crop, template)
            if score >= _COARSE_FLOOR:
                px, py = origin.x0 + (max(0, x0) + lx) / zoom, origin.y0 + (max(0, y0) + ly) / zoom
                candidates.append((score, page_index, fitz.Rect(px, py, px + w / zoom, py + h / zoom)))

    tried: list[tuple[int, fitz.Rect]] = []
    for _, page_index, rect in sorted(candidates, key=lambda c: -c[0]):
        if any(p == page_index and _area(r & rect) >= 0.5 * _area(rect) for p, r in tried):
            continue
        tried.append((page_index, rect))
        if len(tried) > _CANDIDATES:
            break
        if accept is not None and not accept(page_index, rect):
            continue
        grey, edges = _verified(dst_doc, page_index, rect, art)
        if grey >= _VERIFY_GREY and edges >= _VERIFY_EDGES:
            return page_index, (rect.x0, rect.y0, rect.x1, rect.y1)
    return None


def _render(doc: fitz.Document, page_index: int, bbox: tuple):
    import numpy as np

    try:
        page = doc[page_index]
        rect = fitz.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width < 8 or rect.height < 8:
            return None
        zoom = RENDER_EDGE / max(rect.width, rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False, colorspace=fitz.csGRAY)
    except Exception:
        return None
    if pix.width < 8 or pix.height < 8:
        return None
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
    return arr.copy(), rect


def _render_color(doc: fitz.Document, page_index: int, bbox: tuple):
    import numpy as np

    try:
        page = doc[page_index]
        rect = fitz.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width < 8 or rect.height < 8:
            return None
        zoom = RENDER_EDGE / max(rect.width, rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    except Exception:
        return None
    if pix.width < 8 or pix.height < 8:
        return None
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return arr[:, : pix.width * pix.n].reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def _colour_mask(colour_a, colour_b, shift, size):
    """Where the two match in structure but not in hue - see COLOUR_HUE_LEVEL -
    at the SAME alignment the grey pass already solved. None of this ever
    fires on its own: a figure whose colour rendering could not be read (a
    vector drawing with no fill, say) simply contributes nothing, and the
    ordinary grey-level pass still runs regardless."""
    import cv2
    import numpy as np

    if colour_a is None or colour_b is None:
        return None
    colour_b = cv2.resize(colour_b, size, interpolation=cv2.INTER_AREA)
    colour_b = cv2.warpAffine(colour_b, shift, size, borderMode=cv2.BORDER_REPLICATE)
    hsv_a = cv2.cvtColor(cv2.GaussianBlur(colour_a, (5, 5), 0), cv2.COLOR_RGB2HSV)
    hsv_b = cv2.cvtColor(cv2.GaussianBlur(colour_b, (5, 5), 0), cv2.COLOR_RGB2HSV)
    chroma = np.minimum(hsv_a[:, :, 1], hsv_b[:, :, 1])
    hue_gap = np.abs(hsv_a[:, :, 0].astype(np.int16) - hsv_b[:, :, 0].astype(np.int16))
    hue_gap = np.minimum(hue_gap, 180 - hue_gap)
    return np.where((chroma >= COLOUR_MIN_CHROMA) & (hue_gap >= COLOUR_HUE_LEVEL), 255, 0).astype(np.uint8)


def differing_regions(expected: fitz.Document, exp_box: tuple[int, tuple],
                      actual: fitz.Document, act_box: tuple[int, tuple]) -> dict | None:
    """{"exp_marks": [(page, bbox)], "act_marks": [(page, bbox)]} for every
    compact region that renders differently, in each document's own page
    coordinates; None when nothing local differs or it cannot be measured."""
    if not available():
        return None
    import cv2
    import numpy as np

    a = _render(expected, *exp_box)
    b = _render(actual, *act_box)
    if a is None or b is None:
        return None
    (img_a, rect_a), (img_b, rect_b) = a, b
    h, w = img_a.shape
    img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)

    fa = cv2.GaussianBlur(img_a, (5, 5), 0).astype(np.float32)
    fb = cv2.GaussianBlur(img_b, (5, 5), 0).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(fa, fb)
    if abs(dx) > w * 0.1 or abs(dy) > h * 0.1:
        return None
    shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
    fb = cv2.warpAffine(fb, shift, (w, h), borderMode=cv2.BORDER_REPLICATE)

    diff = cv2.absdiff(fa, fb).astype(np.uint8)
    _, mask = cv2.threshold(diff, DIFF_LEVEL, 255, cv2.THRESH_BINARY)
    colour_mask = _colour_mask(_render_color(expected, *exp_box), _render_color(actual, *act_box), shift, (w, h))
    if colour_mask is not None:
        mask = cv2.bitwise_or(mask, colour_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if cv2.countNonZero(mask) > MAX_CHANGED_SHARE * w * h:
        return None
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_REGION_SHARE * w * h]
    if not regions:
        return None
    regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)[:MAX_REGIONS]

    def to_page(rect: fitz.Rect, x: int, y: int, rw: int, rh: int) -> tuple:
        sx, sy = rect.width / w, rect.height / h
        return (rect.x0 + x * sx, rect.y0 + y * sy, rect.x0 + (x + rw) * sx, rect.y0 + (y + rh) * sy)

    return {
        "exp_marks": [(exp_box[0], to_page(rect_a, *r)) for r in regions],
        "act_marks": [(act_box[0], to_page(rect_b, *r)) for r in regions],
    }
