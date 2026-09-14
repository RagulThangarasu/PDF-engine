"""Behaviour tests for the checks and the page boxes behind pdf.html.

Each test builds the smallest pair of PDFs that shows the behaviour, in memory,
so what is being pinned is visible in the test itself:

  * black spots on Staging's copy of a picture are found and boxed where they
    are, and a clean copy reports nothing;
  * "bold missing" is decided by how dark the text prints, not by the font -
    text that prints just as dark is not an issue;
  * a list's indent is compared within the list, so a document set on another
    page template is not reported, but a flattened sub-item is;
  * a reworded sentence is boxed on the words that changed, not the paragraph;
  * every difference type is filed under one of the six categories.
"""
from __future__ import annotations

import fitz

from pdfval import imagefp
from pdfval.report import issues as I
from pdfval.validators import chapter as C


# --- black spots ------------------------------------------------------------

def _drawing(spot: tuple[float, float] | None = None) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(fitz.Rect(50, 50, 350, 300), color=(0, 0, 0), width=2)
    page.draw_line((60, 110), (340, 110), color=(0, 0, 0), width=1)
    page.draw_circle((200, 200), 45, color=(0.2, 0.2, 0.2), width=1.5)
    page.draw_line((80, 270), (320, 150), color=(0.3, 0.3, 0.3), width=1)
    if spot is not None:
        page.draw_circle(spot, 6, color=(0, 0, 0), fill=(0, 0, 0))
    return doc


def _figure(doc: fitz.Document) -> C.Element:
    bbox = (45.0, 45.0, 355.0, 305.0)
    return C.Element(kind=C.KIND_FIGURE, boxes=[(0, bbox)], width=310, height=260,
                     fp=imagefp.fingerprint(doc, 0, bbox))


def test_black_spot_on_staging_picture_is_found_where_it_is():
    prod, stage = _drawing(), _drawing(spot=(120, 250))
    diffs = C.black_spots(_figure(prod), _figure(stage), "the figure", prod, stage)
    assert [d["type"] for d in diffs] == ["figure-spots"]
    page, (x0, y0, x1, y1) = diffs[0]["marks"][0]
    assert page == 0 and x0 <= 120 <= x1 and y0 <= 250 <= y1


def test_identical_pictures_have_no_black_spots():
    prod, stage = _drawing(), _drawing()
    assert C.black_spots(_figure(prod), _figure(stage), "the figure", prod, stage) == []


# --- bold, judged by how it prints -----------------------------------------

def _line(fontname: str) -> tuple[fitz.Document, C.Element]:
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), "Connection settings", fontname=fontname, fontsize=14)
    text = "Connection settings"
    element = C.Element(kind=C.KIND_TEXT, text=text, key=C._normalise(text), boxes=[(0, (45.0, 85.0, 250.0, 105.0))])
    return doc, element


def _bold_candidate() -> dict:
    return {"type": "bold-missing", "kind": C.KIND_TEXT, "phrases": ["connection settings"],
            "summary": "Bold missing in Staging — “connection settings” is bold in Production and regular in Staging.",
            "detail": ""}


def test_bold_that_prints_regular_in_staging_is_reported():
    prod_doc, prod_el = _line("hebo")
    stage_doc, stage_el = _line("helv")
    kept = C.confirm_bold([_bold_candidate()], prod_el, stage_el, prod_doc, stage_doc)
    assert [d["type"] for d in kept] == ["bold-missing"]
    assert kept[0]["marks"] and kept[0]["exp_marks"]


def test_text_that_prints_just_as_dark_is_not_a_bold_issue():
    prod_doc, prod_el = _line("hebo")
    stage_doc, stage_el = _line("hebo")
    assert C.confirm_bold([_bold_candidate()], prod_el, stage_el, prod_doc, stage_doc) == []


# --- list indent, within the list --------------------------------------------

def _list(first_x: float, second_x: float) -> tuple[fitz.Document, C.Element]:
    doc = fitz.open()
    page = doc.new_page(width=500, height=300)
    for (x, y, marker, words) in ((first_x, 100, "1.", "Open the settings menu"),
                                  (second_x, 120, "2.", "Select the display option")):
        page.insert_text((x, y), marker, fontname="helv", fontsize=11)
        page.insert_text((x + 15, y), words, fontname="helv", fontsize=11)
    text = "1. Open the settings menu 2. Select the display option"
    element = C.Element(kind=C.KIND_TEXT, text=text, key=C._normalise(text), boxes=[(0, (30.0, 85.0, 480.0, 125.0))])
    return doc, element


def test_list_set_on_another_page_template_is_not_an_indent_issue():
    prod_doc, prod_el = _list(80, 80)
    stage_doc, stage_el = _list(50, 50)  # the whole list 30pt further left
    assert C.list_indent_changes([prod_el], [stage_el], prod_doc, stage_doc) == []


def test_sub_item_indented_differently_within_its_list_is_reported():
    prod_doc, prod_el = _list(80, 80)
    stage_doc, stage_el = _list(80, 100)  # the second item pushed 20pt deeper
    diffs = C.list_indent_changes([prod_el], [stage_el], prod_doc, stage_doc)
    assert [d["type"] for d in diffs] == ["list-indent"]
    assert "level 2" in diffs[0]["summary"] and "level 1" in diffs[0]["summary"]


# --- only the changed words are boxed -------------------------------------------

def _sentence(text: str) -> tuple[fitz.Document, C.Element]:
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((50, 100), text, fontname="helv", fontsize=12)
    element = C.Element(kind=C.KIND_TEXT, text=text, key=C._normalise(text), boxes=[(0, (45.0, 85.0, 480.0, 105.0))])
    return doc, element


def test_reworded_sentence_is_boxed_on_the_changed_word_only():
    prod_text, stage_text = "Set up auncher. For details, see the guide.", "Set up launcher. For details, see the guide."
    prod_doc, prod_el = _sentence(prod_text)
    stage_doc, stage_el = _sentence(stage_text)
    diff = {"type": "text", "word_diff": C._word_diff(prod_text, stage_text)}
    prod, stage = I._word_marks(diff, prod_el, stage_el, prod_doc, stage_doc)
    word_p = prod_doc[0].search_for("auncher.")[0]
    word_s = stage_doc[0].search_for("launcher.")[0]
    assert len(prod) == 1 and len(stage) == 1
    assert abs(prod[0]["bbox"][0] - word_p.x0) < 1 and abs(prod[0]["bbox"][2] - word_p.x1) < 1
    assert abs(stage[0]["bbox"][0] - word_s.x0) < 1 and abs(stage[0]["bbox"][2] - word_s.x1) < 1
    assert "auncher" in prod[0]["note"] and "launcher" in stage[0]["note"]


# --- categories ----------------------------------------------------------------

def test_every_difference_is_filed_under_one_category():
    assert C.category_of({"type": "text"}) == "content"
    assert C.category_of({"type": "missing", "kind": C.KIND_TEXT}) == "content"
    assert C.category_of({"type": "missing", "kind": C.KIND_TABLE}) == "content"
    assert C.category_of({"type": "table-cell"}) == "content"
    assert C.category_of({"type": "table-merge"}) == "tables"
    assert C.category_of({"type": "added", "kind": C.KIND_FIGURE}) == "images"
    assert C.category_of({"type": "figure-spots"}) == "images"
    assert C.category_of({"type": "figure-alignment"}) == "images"
    assert C.category_of({"type": "link-target"}) == "links"
    assert C.category_of({"type": "numbering"}) == "lists"
    assert C.category_of({"type": "list-indent"}) == "lists"
    assert C.category_of({"type": "bold-missing"}) == "bold"
    assert C.category_of({"type": "table-header-repeat"}) == "content"
    assert C.category_of({"type": "shading"}) == "formatting"
    assert {c["key"] for c in C.CATEGORIES} == {"content", "images", "links", "lists", "bold", "tables", "formatting"}


# --- tables, row by row --------------------------------------------------------

from collections import Counter  # noqa: E402


def _table(rows, spans=None):
    cells = tuple(tuple(C._normalise(c) for c in row) for row in rows)
    text = " | ".join(" ".join(c for c in row if c) for row in cells)
    return C.Element(
        kind=C.KIND_TABLE, text=text, key=text, boxes=[(0, (50.0, 100.0, 400.0, 100.0 + 20 * len(rows)))],
        rows=len(rows), cols=max(len(r) for r in rows), cells=cells, spans=tuple(spans or [0] * len(rows)),
        row_boxes=tuple((0, (50.0, 100.0 + 20 * i, 400.0, 120.0 + 20 * i)) for i in range(len(rows))),
    )


_SPEC = [("Model", "Size"), ("ST4304", "43 inch"), ("ST5504", "55 inch"), ("ST6504", "65 inch")]


def test_table_row_missing_in_staging_is_reported_on_that_row():
    prod, stage = _table(_SPEC), _table([_SPEC[0], _SPEC[1], _SPEC[3]])
    diffs = C.table_row_changes(prod, stage, Counter(), Counter())
    assert [d["type"] for d in diffs] == ["table-row-missing"]
    assert "st5504" in diffs[0]["summary"]
    assert diffs[0]["exp"].boxes == [(0, (50.0, 140.0, 400.0, 160.0))]


def test_row_printed_outside_the_other_table_is_not_missing():
    prod, stage = _table(_SPEC), _table([_SPEC[0], _SPEC[1], _SPEC[3]])
    table_units = sum((C._text_units(" ".join(r)) for r in stage.cells), Counter())
    staging_section = table_units + C._text_units("st5504 55 inch")
    assert C.table_row_changes(prod, stage, Counter(), staging_section) == []


def test_cells_merged_in_a_row_are_reported():
    prod = _table([("Model", "43", "55"), ("Speaker", "2 W", "2 W")])
    stage = _table([("Model", "43", "55"), ("Speaker", "2 W 2 W", "")], spans=[0, 1])
    diffs = C.table_row_changes(prod, stage, Counter(), Counter())
    assert [d["type"] for d in diffs] == ["table-merge"]
    assert "merged" in diffs[0]["summary"] and "row 2" in diffs[0]["summary"]
