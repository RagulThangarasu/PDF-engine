"""Perceptual fingerprints for a figure, and the similarity measure built on
them.

This is what lets Image Validation say "this is the same picture" rather than
only "this section has the same NUMBER of pictures". A figure is fingerprinted
from its RENDERED APPEARANCE - the page region rasterised as the reader sees it
- not from the embedded image stream, because the two documents rarely embed
the same bytes for the same illustration (different exporter, different
compression, raster on one side and vector on the other), while what they put
on the page looks the same.

Two independent signals are kept, and a difference is only reported when they
agree:

* `ncc`   - normalised cross-correlation of a 32x32 grey thumbnail. Each
            thumbnail is mean-subtracted and divided by its own standard
            deviation first, so a figure that was merely re-exported lighter,
            darker or with more contrast still correlates at ~1.0.
* `dhash` - a 64-bit difference hash (each bit says "is this pixel brighter
            than the one to its right"). It ignores absolute intensity
            completely and captures structure/edges, which is the signal that
            survives rescaling.

Requiring BOTH to say "different" is the whole reason this can be trusted where
a plain pixel diff could not: a re-rasterised illustration moves one of them a
little and never both a lot.
"""
from __future__ import annotations

from dataclasses import dataclass

import fitz
from PIL import Image

THUMB = 32  # edge of the grey thumbnail used for correlation
DHASH_EDGE = 8  # dhash is computed on a (DHASH_EDGE+1) x DHASH_EDGE grid
DHASH_BITS = DHASH_EDGE * DHASH_EDGE

# Rendering: enough pixels that the thumbnail isn't upscaled from mush, but not
# so many that fingerprinting a manual's worth of figures becomes the slow part.
RENDER_MIN_PX = 96
RENDER_MAX_ZOOM = 4.0

# A region flatter than this carries no structure to compare (a blank box, a
# solid tint). Correlating noise against noise produces arbitrary scores, so
# such a figure is fingerprinted as "featureless" and never drives a
# content-difference finding.
FLAT_STDDEV = 2.0


@dataclass
class Fingerprint:
    """A figure's comparable appearance. `usable` is False when the region
    could not be rendered or carries no structure worth comparing."""

    dhash: int
    thumb: list[float]  # THUMB*THUMB, mean-subtracted and unit-variance
    aspect: float  # rendered width / height on the page
    width: float  # rendered size in points
    height: float
    stddev: float
    usable: bool

    @property
    def featureless(self) -> bool:
        return not self.usable or self.stddev < FLAT_STDDEV


_UNUSABLE = Fingerprint(0, [], 1.0, 0.0, 0.0, 0.0, False)


def fingerprint(doc: fitz.Document, page_index: int, bbox: tuple) -> Fingerprint:
    """Rasterise the figure's page region and reduce it to comparable form."""
    try:
        page = doc[page_index]
        rect = fitz.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width < 1 or rect.height < 1:
            return _UNUSABLE
        zoom = min(
            RENDER_MAX_ZOOM,
            max(1.0, RENDER_MIN_PX / max(1.0, min(rect.width, rect.height))),
        )
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False, colorspace=fitz.csGRAY
        )
        if pix.width < 2 or pix.height < 2:
            return _UNUSABLE
        img = _to_image(pix)
    except Exception:
        return _UNUSABLE

    thumb_raw = _resample(img, THUMB, THUMB)
    stddev = _stddev(thumb_raw)
    dhash_raw = _resample(img, DHASH_EDGE + 1, DHASH_EDGE)
    return Fingerprint(
        dhash=_dhash(dhash_raw, DHASH_EDGE + 1, DHASH_EDGE),
        thumb=_normalize(thumb_raw, stddev),
        aspect=(rect.width / rect.height) if rect.height else 1.0,
        width=rect.width,
        height=rect.height,
        stddev=stddev,
        usable=True,
    )


def _to_image(pix: fitz.Pixmap) -> Image.Image:
    """The pixmap as a PIL greyscale image. `pix.samples` is normally tightly
    packed, but a pixmap can carry a per-row stride wider than the row itself,
    which would shear the image if handed straight to PIL - so that case is
    unpacked row by row.
    """
    data = pix.samples
    expected = pix.width * pix.height * pix.n
    if pix.n == 1 and len(data) == expected:
        return Image.frombytes("L", (pix.width, pix.height), data)
    stride, n = pix.stride, pix.n
    rows = bytearray()
    for y in range(pix.height):
        row = data[y * stride : y * stride + pix.width * n]
        rows.extend(row[::n] if n > 1 else row)
    return Image.frombytes("L", (pix.width, pix.height), bytes(rows))


def _resample(img: Image.Image, out_w: int, out_h: int) -> list[float]:
    """Box-average the render down to out_w x out_h.

    Averaging (rather than nearest-neighbour sampling) is what makes the
    fingerprint stable across two renderings of the same artwork at different
    raster resolutions - a point sample lands on a different part of a stroke
    each time, an area average does not.
    """
    return [float(v) for v in img.resize((out_w, out_h), Image.BOX).getdata()]


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _normalize(values: list[float], stddev: float) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    scale = stddev if stddev > 1e-6 else 1.0
    return [(v - mean) / scale for v in values]


def _dhash(values: list[float], w: int, h: int) -> int:
    bits = 0
    for y in range(h):
        row = y * w
        for x in range(w - 1):
            bits <<= 1
            if values[row + x] > values[row + x + 1]:
                bits |= 1
    return bits


def hash_similarity(a: Fingerprint, b: Fingerprint) -> float:
    """1.0 when every structural bit agrees, 0.0 when none do."""
    if not a.usable or not b.usable:
        return 0.0
    differing = bin(a.dhash ^ b.dhash).count("1")
    return 1.0 - differing / DHASH_BITS


def correlation(a: Fingerprint, b: Fingerprint) -> float:
    """Normalised cross-correlation of the two grey thumbnails, in [-1, 1]."""
    if not a.usable or not b.usable or len(a.thumb) != len(b.thumb) or not a.thumb:
        return 0.0
    total = sum(x * y for x, y in zip(a.thumb, b.thumb))
    return max(-1.0, min(1.0, total / len(a.thumb)))


# --- second opinion: matching two regions whose detected extents disagree ----
#
# The thumbnail fingerprint above assumes both sides detected the SAME extent
# of the same artwork. Often they do not: one document's exporter draws an
# illustration as one object and the other draws it in pieces, so figure
# detection hands back a tight crop on one side and a crop that also swallows
# the leader lines, the caption, or the neighbouring sub-figure on the other.
# Squashing two differently-cropped views of one picture into the same square
# stretches them differently, and they stop correlating.
#
# So when the cheap comparison is unsure, both regions are re-rendered at ONE
# COMMON ABSOLUTE SCALE (points per pixel, not fitted to a box) and the smaller
# is slid over the larger to find its best position. The same artwork then
# correlates strongly wherever inside the other crop it happens to sit.
SLIDE_PPP = 0.22  # pixels per PDF point
SLIDE_MAX_PX = 110  # cap per side, so the search stays cheap
SLIDE_SCALES = (1.0, 0.9, 1.1, 0.8, 1.25)  # relative scales tried
SLIDE_STEPS = 24  # positions sampled along the shorter search axis


def render_gray(doc: fitz.Document, page_index: int, bbox: tuple):
    """The region as a float32 numpy array at SLIDE_PPP, or None."""
    try:
        import numpy as np

        page = doc[page_index]
        rect = fitz.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width < 1 or rect.height < 1:
            return None
        zoom = min(SLIDE_PPP, SLIDE_MAX_PX / max(1.0, max(rect.width, rect.height)))
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False, colorspace=fitz.csGRAY
        )
        if pix.width < 4 or pix.height < 4:
            return None
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.stride)[:, : pix.width]
        return arr.astype(np.float32)
    except Exception:
        return None


def region_match(a, b) -> float | None:
    """Best normalised correlation of the smaller region placed anywhere inside
    the larger, over a few relative scales. None when it cannot be computed
    (no numpy, a region too small, or a featureless one).
    """
    if a is None or b is None:
        return None
    try:
        import numpy as np
        from PIL import Image as _Image
    except Exception:
        return None

    best = None
    for scale in SLIDE_SCALES:
        if scale == 1.0:
            scaled = a
        else:
            h, w = a.shape
            scaled = np.asarray(
                _Image.fromarray(a.astype(np.uint8)).resize(
                    (max(4, int(w * scale)), max(4, int(h * scale))), _Image.BOX
                ),
                dtype=np.float32,
            )
        for small, large in ((scaled, b), (b, scaled)):
            value = _best_offset_ncc(small, large)
            if value is not None and (best is None or value > best):
                best = value
    return best


def _best_offset_ncc(small, large) -> float | None:
    import numpy as np

    sh, sw = small.shape
    bh, bw = large.shape
    if sh > bh or sw > bw or sh < 4 or sw < 4:
        return None
    s = small - small.mean()
    s_norm = float(np.sqrt((s * s).sum()))
    if s_norm < 1e-3:
        return None  # featureless - would correlate with anything
    step = max(1, min(bh - sh, bw - sw) // SLIDE_STEPS + 1)
    best = None
    for oy in range(0, bh - sh + 1, step):
        for ox in range(0, bw - sw + 1, step):
            window = large[oy : oy + sh, ox : ox + sw]
            window = window - window.mean()
            w_norm = float(np.sqrt((window * window).sum()))
            if w_norm < 1e-3:
                continue
            value = float((s * window).sum()) / (s_norm * w_norm)
            if best is None or value > best:
                best = value
    return best


def similarity(a: Fingerprint, b: Fingerprint) -> float:
    """One score for matching figures to each other, blending both signals.

    Structure (dhash) is weighted a little higher than correlation: it is the
    signal that holds up when the same illustration is drawn at a different
    size, which is the common case when a figure moves between pages.
    """
    return 0.55 * hash_similarity(a, b) + 0.45 * max(0.0, correlation(a, b))


# --- sharpness ------------------------------------------------------------
#
# A figure that survived the rebuild as a soft, low-resolution copy is a real
# defect and nothing else here catches it: the fingerprint correlates at ~1.0
# (it is the same picture), the size check sees the same extent, and the broken
# -image check only fires on artwork that is blank or fails to decode.
#
# Measured as edge energy normalised by the region's own contrast, at a FIXED
# pixel size. Both normalisations matter: without the contrast term a pale
# diagram reads as blurry, and without the common pixel size the side that
# happens to be placed larger on the page always looks sharper.
# Rendered at a fixed DPI, NOT a fixed pixel size: blur in a PDF is almost
# always a low-resolution raster scaled up to fill the same box, and
# downsampling both sides to a common small size destroys exactly the
# high-frequency difference being looked for (a 3px Gaussian moved this metric
# by 7% at 320px, and by 60% at 150 DPI).
SHARP_DPI = 150
SHARP_MAX_PX = 1400  # cap, so a full-page figure stays cheap to measure


def sharpness(doc: "fitz.Document", page_index: int, bbox: tuple) -> float | None:
    """Edge energy of the region, normalised by its contrast, or None when it
    cannot be measured (no numpy, or a region too small or featureless)."""
    try:
        import numpy as np

        page = doc[page_index]
        rect = fitz.Rect(*bbox) & page.rect
        if rect.is_empty or rect.width < 8 or rect.height < 8:
            return None
        zoom = min(SHARP_DPI / 72.0, SHARP_MAX_PX / max(1.0, max(rect.width, rect.height)))
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False, colorspace=fitz.csGRAY
        )
        if pix.width < 16 or pix.height < 16:
            return None
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.stride)[:, : pix.width].astype(np.float32)
        contrast = float(arr.std())
        if contrast < 3.0:
            return None  # a flat region has no edges to be sharp or soft
        gy, gx = np.gradient(arr)
        edge = float(np.abs(gx).mean() + np.abs(gy).mean())
        return edge / contrast
    except Exception:
        return None
