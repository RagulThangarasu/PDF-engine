"""User rules: a lost paragraph space is reported; "on page N" dropped in
Staging is expected but one Staging still prints is reported; one web address
written two ways is one link target."""
import fitz

from pdfval.validators.chapter import (
    KIND_TEXT, Element, _normalise, _same_address, page_ref_in_staging_changes, paragraph_gap_changes,
)


def _el(text, page, bbox, size=10.0):
    return Element(kind=KIND_TEXT, text=text, key=_normalise(text), boxes=[(page, bbox)], size=size)


def test_paragraph_space_lost_in_staging():
    first = _el("Connect the power cable.", 0, (72, 100, 400, 112))
    second = _el("Then press the Power button.", 0, (72, 130, 400, 142))  # 18pt gap above
    joined = _el("Connect the power cable. Then press the Power button.", 0, (72, 100, 400, 124))
    found = paragraph_gap_changes([first, second], [joined])
    assert len(found) == 1 and "Paragraph space missing" in found[0]["summary"]
    # continued on the next page: not a paragraph space
    second_next_page = _el("Then press the Power button.", 1, (72, 60, 400, 72))
    assert paragraph_gap_changes([first, second_next_page], [joined]) == []


def test_page_reference_only_reported_when_staging_prints_it(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), 'See "Choosing a location" on page 26.', fontsize=10)
    kept = _el('See "Choosing a location" on page 26.', 0, (70, 90, 400, 104))
    dropped = _el('See "Choosing a location".', 0, (70, 90, 400, 104))
    found = page_ref_in_staging_changes([kept], doc)
    assert len(found) == 1 and "on page 26" in found[0]["summary"]
    assert page_ref_in_staging_changes([dropped], doc) == []


def test_same_web_address_is_one_target():
    assert _same_address("http://support.benq.com.", "https://Support.BenQ.com/")
    assert not _same_address("http://support.benq.com", "http://benq.eu")
