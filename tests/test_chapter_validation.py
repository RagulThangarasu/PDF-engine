"""Behaviour tests for Chapter Validation.

Run with:  python -m pytest tests            (or)  python tests/test_chapter_validation.py

A chapter is an L1 heading read through to the next one, on both documents at
once. The check is only useful if it reports what a reviewer would call a
difference and nothing else, so these fixtures pin down both halves:

* the same words, set in a narrower column, carried over a page break, and
  printed on more pages than Production uses - no difference;
* a table of contents, a Q&A section and a page footer - never compared;
* a typo fixed and a paragraph removed - each reported, and nothing else.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from pdfval.extractor import reset_table_cache  # noqa: E402
from pdfval.ocr import reset_ocr_cache  # noqa: E402
from pdfval.validators.chapter import compare_chapters, reset_furniture_cache  # noqa: E402
from pdfval.validators.headings import reset_heading_cache  # noqa: E402
from pdfval.validators.toc import reset_paragraph_cache  # noqa: E402
from pdfval.visual_diff import reset_cache as reset_visual_cache  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        from tests import make_fixtures

        make_fixtures.main()
    return path


def _chapters(prod: str, stage: str):
    prod_path, stage_path = _fixture(prod), _fixture(stage)
    expected = fitz.open(prod_path)
    actual = fitz.open(stage_path)
    try:
        return compare_chapters(expected, actual, prod_path, stage_path)
    finally:
        expected.close()
        actual.close()
        # Every cache `cli.run` clears after a comparison. Several are keyed by
        # id(doc), and a later test's freshly opened document can be handed the
        # same id - and with it this test's paragraphs and heading lines, which
        # is how a content-validation test elsewhere in the suite failed only
        # when these tests ran first.
        reset_table_cache()
        reset_paragraph_cache()
        reset_heading_cache()
        reset_furniture_cache()
        reset_ocr_cache()
        reset_visual_cache()


def _real(chapter) -> list[dict]:
    return [d for d in chapter.differences if not d.get("minor")]


def test_chapters_are_the_shared_l1_headings():
    chapters = _chapters("chapters_prod.pdf", "chapters_stage_reflowed.pdf")
    assert [c.title for c in chapters] == ["Getting started", "Care and cleaning"]
    # The contents page is not part of any chapter on either side.
    assert 0 not in chapters[0].exp_pages and 0 not in chapters[0].act_pages


def test_reflow_and_page_break_are_not_differences():
    chapters = _chapters("chapters_prod.pdf", "chapters_stage_reflowed.pdf")
    real = [(c.title, d["type"], d["summary"]) for c in chapters for d in _real(c)]
    assert real == []


def test_paragraph_cut_by_a_page_break_is_one_element():
    chapters = _chapters("chapters_prod.pdf", "chapters_stage_reflowed.pdf")
    carried = [e for e in chapters[0].act_elements if "remembers the last source" in e.text]
    assert len(carried) == 1
    assert "next time it is powered on" in carried[0].text
    assert len(carried[0].pages) == 2  # boxed on both pages it is printed across


def test_contents_qa_and_footer_are_not_compared():
    chapters = _chapters("chapters_prod.pdf", "chapters_stage_reflowed.pdf")
    texts = [e.text for c in chapters for e in c.exp_elements + c.act_elements]
    assert not any("Why is there no" in t for t in texts)  # the Q&A section
    assert not any(t.startswith("User guide") for t in texts)  # the footer
    assert not any("....." in t for t in texts)  # the contents listing
    care = chapters[1]
    assert "Q&A" in care.skipped["prod"] and "Q&A" in care.skipped["stage"]


def test_typo_and_removed_paragraph_are_reported():
    chapters = {c.title: c for c in _chapters("chapters_prod.pdf", "chapters_stage_edited.pdf")}

    [typo] = _real(chapters["Getting started"])
    assert typo["type"] == "text"
    changed = {(op["type"], op["text"]) for op in typo["word_diff"] if op["type"] != "equal"}
    assert changed == {("del", "Nevigate"), ("ins", "Navigate")}

    [gone] = _real(chapters["Care and cleaning"])
    assert gone["type"] == "missing"
    assert "soft, dry cloth" in gone["exp"].text


# --- list numbering ---------------------------------------------------------
# List markers are left out of the wording comparison (a marker drawn beside
# its step here and inside it there is not a wording change), so a renumbered
# procedure needs its own check - and must still be caught.


def test_list_renumbered_to_letters_is_reported():
    [chapter] = _chapters("list_prod.pdf", "list_stage_lettered.pdf")
    [change] = [d for d in _real(chapter) if d["type"] == "numbering"]
    assert "is step 1 in Production but item a in Staging" in change["summary"]
    assert "(and 4 more like it)" in change["summary"]


def test_list_renumbered_to_parenthesised_letters_is_reported():
    # "(a)", "(b)", "(c)" - not just "a.", "a)" - is a marker style its own
    # right, and must be read as a complete marker rather than as a stray
    # leading "(" in front of one, or every list set this way goes unreported.
    [chapter] = _chapters("list_prod.pdf", "list_stage_parenlettered.pdf")
    [change] = [d for d in _real(chapter) if d["type"] == "numbering"]
    assert "is step 1 in Production but item a in Staging" in change["summary"]
    assert "(and 4 more like it)" in change["summary"]


def test_list_renumbered_to_letters_with_a_space_before_the_period_is_reported():
    # "a ." - a space printed before the marker's own period - is still a
    # complete letter marker, not a stray "a" with unrelated punctuation
    # tacked on somewhere after it.
    [chapter] = _chapters("list_prod.pdf", "list_stage_spacedletters.pdf")
    [change] = [d for d in _real(chapter) if d["type"] == "numbering"]
    assert "is step 1 in Production but item a in Staging" in change["summary"]
    assert "(and 4 more like it)" in change["summary"]


def test_list_reduced_to_bullets_is_reported():
    [chapter] = _chapters("list_prod.pdf", "list_stage_bulleted.pdf")
    [change] = [d for d in _real(chapter) if d["type"] == "numbering"]
    assert "is step 1 in Production but a bulleted item in Staging" in change["summary"]


def test_same_numbering_drawn_as_detached_markers_is_not_reported():
    [chapter] = _chapters("list_prod.pdf", "list_stage_detached.pdf")
    assert [d for d in chapter.differences if d["type"] == "numbering"] == []


# --- inline icons ------------------------------------------------------------
# An icon beside a word in running text ("Settings [gear icon]") is its own
# check: the word-level text diff never sees it (it is a picture, not a
# character), and the whole-figure check never sees it either (too small to
# be a figure of its own) - so a missing or recoloured inline icon has no
# other check that would catch it.


def test_inline_icon_removed_is_reported_as_missing():
    [chapter] = _chapters("icons_prod.pdf", "icons_stage_missing.pdf")
    [change] = [d for d in _real(chapter) if d["type"] == "icon-missing"]
    assert "the icon beside “Settings”" in change["summary"]


def test_inline_icon_recoloured_is_reported():
    [chapter] = _chapters("icons_prod.pdf", "icons_stage_colour.pdf")
    [change] = [d for d in _real(chapter) if d["type"] == "icon-colour"]
    assert "grey in Production and red in Staging" in change["summary"]


def test_identical_icon_is_not_reported():
    [chapter] = _chapters("icons_prod.pdf", "icons_prod.pdf")
    assert [d for d in chapter.differences if d["type"].startswith("icon-")] == []


# --- a wrapper heading only one side has is reported, not silently swallowed --

def _copyright_disclaimer_expected() -> fitz.Document:
    doc = fitz.open()
    cover = doc.new_page(width=612, height=792)
    cover.insert_text((72, 300), "Monitor XL2540", fontsize=28)
    p0 = doc.new_page(width=612, height=792)
    p0.insert_text((72, 60), "Copyright", fontsize=16)
    p0.insert_textbox(fitz.Rect(72, 90, 540, 200),
                      "Copyright (c) 2025 by BenQ Corporation. All rights reserved.", fontsize=10)
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 60), "Disclaimer", fontsize=16)
    p1.insert_textbox(fitz.Rect(72, 90, 540, 200),
                      "BenQ Corporation makes no representations or warranties of any kind.", fontsize=10)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 60), "Getting Started", fontsize=16)
    p2.insert_textbox(fitz.Rect(72, 90, 540, 200),
                      "Unpack the monitor and place it on a stable, flat surface near an outlet.", fontsize=10)
    doc.set_toc([[1, "Copyright", 2], [1, "Disclaimer", 3], [1, "Getting Started", 4]])
    return doc


def _copyright_disclaimer_actual() -> fitz.Document:
    doc = fitz.open()
    cover = doc.new_page(width=612, height=792)
    cover.insert_text((72, 300), "Monitor XL2540", fontsize=28)
    p0 = doc.new_page(width=612, height=792)
    p0.insert_text((72, 40), "Copyright & Disclaimer", fontsize=18)
    p0.insert_text((72, 70), "Copyright", fontsize=14)
    p0.insert_textbox(fitz.Rect(72, 90, 540, 140),
                      "Copyright (c) 2025 by BenQ Corporation. All rights reserved.", fontsize=10)
    p0.insert_text((72, 160), "Disclaimer", fontsize=14)
    p0.insert_textbox(fitz.Rect(72, 180, 540, 260),
                      "BenQ Corporation makes no representations or warranties of any kind.", fontsize=10)
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 60), "Getting Started", fontsize=16)
    p1.insert_textbox(fitz.Rect(72, 90, 540, 200),
                      "Unpack the monitor and place it on a stable, flat surface near an outlet.", fontsize=10)
    doc.set_toc([[1, "Copyright & Disclaimer", 2], [2, "Copyright", 2], [2, "Disclaimer", 2],
                 [1, "Getting Started", 3]])
    return doc


def test_wrapper_heading_only_in_staging_is_reported():
    # Staging wraps "Copyright" and "Disclaimer" (Production's own top-level
    # headings, same content) inside one new "Copyright & Disclaimer" heading.
    # That wrapper sits right before its own first child heading - which is
    # also the first heading Production and Staging share - so
    # `_mark_front_matter` marks it `.excluded` for an unrelated reason (front
    # matter position), not because it is a TOC/Q&A/RoHS title. It must still
    # be reported as its own section-added finding, not silently dropped.
    # A cover page (page one) precedes all of this on both sides: chapter_pairs
    # never sweeps page one itself into any chapter's span (see its own "cover
    # page proper" comment), so this front matter is placed on pages two and
    # three, same as it would be in a real manual, to test what it is actually
    # meant to test rather than the unrelated cover-page exclusion.
    expected, actual = _copyright_disclaimer_expected(), _copyright_disclaimer_actual()
    try:
        chapters = compare_chapters(expected, actual)
    finally:
        expected.close()
        actual.close()
        reset_table_cache()
        reset_paragraph_cache()
        reset_heading_cache()
        reset_furniture_cache()
        reset_ocr_cache()
        reset_visual_cache()
    matches = [d for ch in chapters for d in ch.differences
               if d["type"] == "section-added" and "Copyright & Disclaimer" in d["summary"]]
    assert len(matches) == 1
    assert "not printed anywhere in Production" in matches[0]["summary"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
