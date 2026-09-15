"""Behaviour tests for the checks and the page boxes behind pdf.html.

Each test builds the smallest pair of PDFs that shows the behaviour, in memory,
so what is being pinned is visible in the test itself:

  * "bold missing" is decided by how dark the text prints, not by the font -
    text that prints just as dark is not an issue;
  * a list's indent is compared within the list, so a document set on another
    page template is not reported, but a flattened sub-item is;
  * a reworded sentence is boxed on the words that changed, not the paragraph;
  * every difference type is filed under one of the six categories.
"""
from __future__ import annotations

import fitz

from pdfval.report import issues as I
from pdfval.validators import chapter as C


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
    assert C.category_of({"type": "figure-different"}) == "images"
    assert C.category_of({"type": "figure-alignment"}) == "images"
    assert C.category_of({"type": "link-target"}) == "links"
    assert C.category_of({"type": "numbering"}) == "lists"
    assert C.category_of({"type": "list-indent"}) == "lists"
    assert C.category_of({"type": "bold-missing"}) == "bold"
    assert C.category_of({"type": "table-header-repeat"}) == "content"
    assert C.category_of({"type": "shading"}) == "formatting"
    assert {c["key"] for c in C.CATEGORIES} == {"content", "images", "links", "lists", "bold", "tables", "formatting"}


# --- visual differences inside a matched figure (OpenCV) ------------------------

from pdfval import visual_diff  # noqa: E402

_FIG_BOX = (0, (45.0, 45.0, 355.0, 305.0))


def _drawing(spot: tuple[float, float] | None = None) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(fitz.Rect(50, 50, 350, 300), color=(0, 0, 0), width=2)
    page.draw_line((60, 110), (340, 110), color=(0, 0, 0), width=1)
    page.draw_circle((200, 200), 45, color=(0.2, 0.2, 0.2), width=1.5)
    if spot is not None:
        page.draw_circle(spot, 10, color=(0, 0, 0), fill=(0, 0, 0))
    return doc


def test_spot_inside_a_matched_figure_is_boxed_where_it_is():
    found = visual_diff.differing_regions(_drawing(), _FIG_BOX, _drawing(spot=(120, 250)), _FIG_BOX)
    assert found is not None
    page, (x0, y0, x1, y1) = found["act_marks"][0]
    assert page == 0 and x0 <= 120 <= x1 and y0 <= 250 <= y1


def test_identical_figures_have_no_visual_differences():
    assert visual_diff.differing_regions(_drawing(), _FIG_BOX, _drawing(), _FIG_BOX) is None


def _diagram() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.draw_rect(fitz.Rect(40, 60, 120, 110), color=(0, 0, 0), width=2)    # a screen
    page.draw_line((30, 118), (130, 118), color=(0, 0, 0), width=3)         # its base
    page.draw_rect(fitz.Rect(250, 40, 300, 150), color=(0, 0, 0), width=2)  # a tower
    page.draw_circle((275, 70), 8, color=(0, 0, 0), width=1.5)
    page.draw_line((130, 90), (250, 90), color=(0, 0, 0), width=1)          # the cable
    return doc


def test_piece_of_a_drawing_is_found_inside_the_other_sides_picture():
    # Production draws the diagram in pieces; Staging embeds it whole, 1.2x larger.
    visual_diff.reset_cache()
    drawn = _diagram()
    pix = drawn[0].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(20, 30, 320, 160))
    embedded = fitz.open()
    page = embedded.new_page(width=500, height=400)
    page.insert_image(fitz.Rect(60, 100, 60 + 300 * 1.2, 100 + 130 * 1.2), pixmap=pix)
    hit = visual_diff.find_artwork(drawn, 0, (245, 35, 305, 155), embedded, [(0, page.rect)])
    assert hit is not None
    _, (x0, y0, _, _) = hit
    assert abs(x0 - (60 + (249 - 20) * 1.2)) < 12 and abs(y0 - (100 + (39 - 30) * 1.2)) < 12


def test_artwork_not_printed_on_the_other_side_is_not_found():
    visual_diff.reset_cache()
    other = fitz.open()
    page = other.new_page(width=400, height=300)
    page.draw_circle((200, 150), 60, color=(0, 0, 0), width=2)
    page.draw_line((140, 150), (260, 150), color=(0, 0, 0), width=2)
    assert visual_diff.find_artwork(_diagram(), 0, (245, 35, 305, 155), other, [(0, page.rect)]) is None


class _DetectedFigure:
    def __init__(self, bbox: tuple, kind: str = "vector"):
        self.bbox, self.kind = bbox, kind


def test_table_header_bar_with_text_on_it_is_not_a_figure():
    rows = [{"bbox": (60.0, 62.0, 120.0, 80.0)}]
    assert C._is_fill_band(_DetectedFigure((50, 57, 540, 85)), rows)
    assert not C._is_fill_band(_DetectedFigure((50, 57, 540, 85), kind="image"), rows)
    assert not C._is_fill_band(_DetectedFigure((100, 100, 300, 250)), rows)


def test_page_reference_is_not_a_list_number():
    el = C.Element(kind=C.KIND_TEXT, text="Follow the steps in Connect the puck to the monitor. on page 23. "
                                          "Hotkey Puck G2 is designed for this monitor only.")
    assert [item[1] for item in C._list_items_in([el])] == []


def test_phrase_found_mid_sentence_is_not_read_as_an_unmarked_item():
    C._CHAR_CACHE.clear()
    doc = fitz.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((50, 100), "The arm works for clamp and grommet mounting to suit your need.",
                     fontname="helv", fontsize=11)
    assert C._marker_before(doc, [0], "grommet mounting") is None


def test_callout_label_in_a_table_cell_is_not_content():
    assert C._cell_units("NOTE: The options vary by input.") == C._cell_units("The options vary by input.")
    assert C._cell_units("Take note of the value.") != C._cell_units("Take of the value.")


# --- line spacing, only against its own neighbours -------------------------

def _paragraph(page, y0: float, size: float, dy: float, lines: int = 3, x: float = 50.0) -> tuple:
    for n in range(lines):
        page.insert_text((x, y0 + n * dy), f"Line {n} of body text here that reads fine", fontsize=size)
    bbox = (x - 2.0, y0 - size, x + 400.0, y0 + (lines - 1) * dy + 2.0)
    return bbox


def _leading_doc(body_dy: float, callout_dy: float, size: float = 10.0) -> tuple[fitz.Document, C.Element, C.Element]:
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    paras = [_paragraph(page, y, size, body_dy) for y in (60, 140, 220)]
    texts = [C.Element(kind=C.KIND_TEXT, text="body", key="body", boxes=[(0, b)], section="t0") for b in paras]
    note_bbox = _paragraph(page, 320, size, callout_dy)
    note = C.Element(kind=C.KIND_NOTE, text="note", key="note", boxes=[(0, note_bbox)], section="t0")
    return doc, note, texts


def test_line_spacing_ignores_a_document_wide_template_change():
    # Both body and callout are looser in "Staging" - one consistent template,
    # not a defect in this one callout.
    prod_doc, prod_note, prod_texts = _leading_doc(body_dy=14.0, callout_dy=14.0)
    stage_doc, stage_note, stage_texts = _leading_doc(body_dy=20.0, callout_dy=20.0)
    diffs = C.line_spacing_changes([prod_note], [stage_note], prod_doc, stage_doc, prod_texts, stage_texts)
    assert diffs == []


def test_line_spacing_catches_a_callout_out_of_step_with_its_own_page():
    # Staging's body text matches Production's own rhythm; only its callout
    # is set looser than the rest of that same page - a real, local change.
    prod_doc, prod_note, prod_texts = _leading_doc(body_dy=14.0, callout_dy=14.0)
    stage_doc, stage_note, stage_texts = _leading_doc(body_dy=14.0, callout_dy=22.0)
    diffs = C.line_spacing_changes([prod_note], [stage_note], prod_doc, stage_doc, prod_texts, stage_texts)
    assert len(diffs) == 1 and diffs[0]["type"] == "line-spacing" and "looser" in diffs[0]["summary"]


def test_visual_difference_is_review_only():
    diff = {"type": "figure-visual", "review_only": True}
    assert C.severity_of(diff) > C.FAILING_SEVERITY
    assert C.category_of(diff) == "images"


# --- whole-book plain text scan -------------------------------------------------
#
# One chapter's own topic-by-topic pass never looks past its own chapter, so
# content lost between chapters - or never reported by anything upstream -
# needs one more pass over the whole book, plain text only, after every
# chapter is done. `plain_text_scan` is that pass.

def _plain(text: str, page: int = 0, section: str = "t0") -> C.Element:
    return C.Element(kind=C.KIND_TEXT, text=text, key=C._normalise(text),
                     boxes=[(page, (50.0, 50.0, 300.0, 70.0))], section=section)


def test_plain_text_scan_finds_content_missing_from_the_whole_book():
    ch1 = C.Chapter(
        title="Getting started", exp_pages=[0], act_pages=[0],
        exp_elements=[_plain("Turn the device on before connecting any cable to it safely")],
        act_elements=[_plain("Turn the device on before connecting any cable to it safely")],
    )
    ch2 = C.Chapter(
        title="Care and cleaning", exp_pages=[1], act_pages=[1],
        exp_elements=[_plain("Clean the lens only with a soft dry microfibre cloth")],
        act_elements=[],
    )
    diffs = C.plain_text_scan([ch1, ch2], fitz.open(), fitz.open())
    assert any(d["type"] == "missing" and "lens" in d["summary"] for _, d in diffs)


def test_plain_text_scan_ignores_content_printed_in_another_chapter():
    # Genuinely present in Staging - just under a different chapter than
    # Production's copy. Not missing, so not reported.
    ch1 = C.Chapter(
        title="Getting started", exp_pages=[0], act_pages=[0],
        exp_elements=[_plain("Clean the lens only with a soft dry microfibre cloth")],
        act_elements=[],
    )
    ch2 = C.Chapter(
        title="Care and cleaning", exp_pages=[1], act_pages=[1],
        exp_elements=[],
        act_elements=[_plain("Clean the lens only with a soft dry microfibre cloth")],
    )
    assert C.plain_text_scan([ch1, ch2], fitz.open(), fitz.open()) == []


def test_plain_text_scan_does_not_repeat_a_difference_already_reported():
    el = _plain("Clean the lens only with a soft dry microfibre cloth")
    ch = C.Chapter(title="Care and cleaning", exp_pages=[1], act_pages=[1], exp_elements=[el], act_elements=[])
    ch.differences.append({"type": "missing", "kind": C.KIND_TEXT, "section": "t0",
                           "summary": "Text missing in Staging", "exp": el, "act": None})
    assert C.plain_text_scan([ch], fitz.open(), fitz.open()) == []


def _page_with_text(text: str) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 100), text, fontname="helv", fontsize=12)
    return doc


def test_plain_text_scan_finds_a_genuine_extra_word_in_staging():
    # Extra content: genuinely nowhere in Production, on the page or off it.
    prod_doc = _page_with_text("Turn the device on before connecting any cable")
    stage_doc = _page_with_text("Turn the device on before connecting any cable extra unrelated bonus wording here")
    el = C.Element(kind=C.KIND_TEXT,
                   text="Turn the device on before connecting any cable extra unrelated bonus wording here",
                   key=C._normalise("x"), boxes=[(0, (45.0, 85.0, 380.0, 105.0))])
    ch = C.Chapter(title="Getting started", exp_pages=[0], act_pages=[0],
                   exp_elements=[_plain("Turn the device on before connecting any cable")],
                   act_elements=[el])
    diffs = C.plain_text_scan([ch], prod_doc, stage_doc)
    assert any(d["type"] == "added" and "bonus" in d["summary"] for _, d in diffs)


def test_plain_text_scan_ignores_an_extra_word_that_is_only_classified_differently():
    # "Extra" in Staging's plain-text stream only because Production's own
    # copy is inside a table cell there (excluded from the plain-text stream)
    # - genuinely on Production's page, so not a real difference. The table
    # cell text is never fed into `plain_text_scan` (tables are excluded from
    # both sides' streams); what matters is that it is still on the PAGE.
    prod_doc = _page_with_text("Warranty card included in the box")
    stage_doc = _page_with_text("Warranty card included in the box")
    table_el = C.Element(kind=C.KIND_TABLE, text="Warranty card included in the box",
                         key=C._normalise("x"), boxes=[(0, (45.0, 85.0, 380.0, 105.0))])
    text_el = C.Element(kind=C.KIND_TEXT, text="Warranty card included in the box",
                        key=C._normalise("x"), boxes=[(0, (45.0, 85.0, 380.0, 105.0))])
    ch = C.Chapter(title="Getting started", exp_pages=[0], act_pages=[0],
                   exp_elements=[table_el], act_elements=[text_el])
    assert C.plain_text_scan([ch], prod_doc, stage_doc) == []


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
