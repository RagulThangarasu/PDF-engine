"""A picture whose stream will not display is reported whatever its size -
an icon-sized PNG included, which the figure-sized floor used to skip."""
import io

import fitz
import pytest

from pdfval.validators import chapter as ch
from pdfval.validators.image import _broken_reason


def _page_with(sizes):
    """One page carrying an image at each (width, height) in points."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 30, 30)).save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for w, h in sizes:
        page.insert_image(fitz.Rect(72, y, 72 + w, y + h), stream=buf.getvalue())
        y += h + 20
    return fitz.open("pdf", doc.tobytes())


def test_icon_sized_image_is_checked_for_a_broken_stream(monkeypatch):
    doc = _page_with([(6, 6), (40, 40)])
    seen = []

    def fake_reason(d, page, img, stream_only=False):
        seen.append((round(img.bbox[2] - img.bbox[0]), stream_only))
        return "embedded image stream is empty"

    monkeypatch.setattr("pdfval.validators.image._broken_reason", fake_reason)
    out = ch.broken_image_changes(doc, [0], lambda p, y: "t1")

    # Both sizes reach the check - the icon asked only about its stream, the
    # figure asked the render questions too.
    assert (6, True) in seen and (40, False) in seen
    assert len(out) == 2
    assert any(d["summary"].startswith("Icon broken in Staging") for d in out)
    assert any(d["summary"].startswith("Image broken in Staging") for d in out)
    assert all(d["type"] == "figure-broken" for d in out)


def test_a_hairline_image_is_still_skipped(monkeypatch):
    doc = _page_with([(2, 2)])
    monkeypatch.setattr("pdfval.validators.image._broken_reason",
                        lambda *a, **k: "embedded image stream is empty")
    assert ch.broken_image_changes(doc, [0], lambda p, y: "t1") == []


def test_stream_only_skips_the_render_heuristics():
    """A small icon that paints as one flat colour is not "broken" - only an
    unreadable stream is, at that size."""
    doc = _page_with([(6, 6)])
    info = doc[0].get_image_info(xrefs=True)[0]
    from types import SimpleNamespace
    img = SimpleNamespace(kind="raster", xref=int(info["xref"]), bbox=tuple(info["bbox"]))
    assert _broken_reason(doc, 0, img, stream_only=True) is None
