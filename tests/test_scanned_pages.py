"""A page that is only a scanned image must still have its text compared (via
OCR) under the right heading, and be reported as scanned."""
import fitz
import pytest

from pdfval import ocr
from pdfval.validators.chapter import compare_chapters
from pdfval.validators.page import validate_pages

pytestmark = pytest.mark.skipif(not ocr.available(), reason="Tesseract not installed")

_TOC = [[1, "1 Introduction", 1], [1, "2 Setup", 2]]


def _build(word: str) -> fitz.Document:
    doc = fitz.open()
    for title, body in (
        ("1 Introduction", "This printer supports wireless printing from any device on the network."),
        ("2 Setup", f"Connect the power cable and press the {word} button for 3 seconds. "
                    "The light turns green when the printer is ready to use."),
    ):
        page = doc.new_page()
        page.insert_text((72, 90), title, fontsize=18, fontname="hebo")
        page.insert_textbox(fitz.Rect(72, 120, 520, 300), body, fontsize=12)
    doc.set_toc(_TOC)
    return doc


def _scan_page_two(src: fitz.Document) -> fitz.Document:
    out = fitz.open()
    out.insert_pdf(src, from_page=0, to_page=0)
    page = out.new_page(width=src[1].rect.width, height=src[1].rect.height)
    page.insert_image(page.rect, pixmap=src[1].get_pixmap(dpi=200))
    out.set_toc(_TOC)
    return fitz.open("pdf", out.tobytes())


def test_scanned_page_text_is_compared_via_ocr():
    ocr.reset_ocr_cache()
    prod, stage = _build("Power"), _scan_page_two(_build("Start"))

    assert ocr.scanned_pages(stage) == [1] and ocr.scanned_pages(prod) == []
    page_msgs = [i.message for i in validate_pages(prod, stage).issues]
    assert any("Scanned page 2 in Staging" in m for m in page_msgs)

    chapters = {c.title: c for c in compare_chapters(prod, stage)}
    assert not chapters["1 Introduction"].differences
    setup = chapters["2 Setup"].differences
    assert len(setup) == 1
    assert "Start" in setup[0]["summary"] and "read by OCR" in setup[0]["summary"]


def _lines(*texts, page=0, ocr_=False):
    return [{"text": t, "page": page, "bbox": (72, 100 + 20 * i, 400, 112 + 20 * i), "ocr": ocr_}
            for i, t in enumerate(texts)]


def test_word_audit_counts_every_word_of_a_topic():
    from pdfval.validators.chapter import Chapter, word_audit_changes

    at = lambda page, y: "t0"
    chapter = Chapter(title="Setup", exp_pages=[0], act_pages=[0], exp_elements=[], act_elements=[])
    prod = _lines("Press the Power button for 3 seconds.", "See the con-", "tents list.", "Focus ring")
    stage = _lines("Press the Power", "button for 5 seconds. See the contents list.", ocr_=True)

    diffs = word_audit_changes(chapter, prod, stage, at, at)
    by = {d["type"]: d for d in diffs}
    # Wrapping and a line-end hyphen are not differences; the changed number,
    # and a lone label no paragraph carries, are.
    assert "“3”" in by["missing"]["summary"] and "“focus”" in by["missing"]["summary"]
    assert "con" not in by["missing"]["summary"] and "“press”" not in by["missing"]["summary"]
    assert "“5”" in by["added"]["summary"] and by["added"].get("ocr_sides") == ["Staging"]

    # A word another finding already names is not reported twice.
    chapter.differences = [{"section": "t0", "summary": "", "gone": ["3", "focus", "ring"], "extra": ["5"]}]
    assert word_audit_changes(chapter, prod, stage, at, at) == []
