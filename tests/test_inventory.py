"""The completeness sweep measures each page and compares the two inventories.

These check the measuring, which is the part that must be right: everything
downstream - the sweep's own gaps and the AI pass that reasons over them - is
only as good as the facts, and a fact that counts a rendering choice as a gap
produces exactly the false findings this must not make.
"""
import io

import fitz
import pytest

from pdfval import ai_review
from pdfval.validators import inventory


def _page(shaded=False, link=False, picture=False, icon=False, words=12):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(60, 60, 500, 160), " ".join(["word"] * words), fontsize=11)
    if shaded:
        page.draw_rect(fitz.Rect(60, 200, 480, 280), color=None, fill=(0.85, 0.9, 0.95))
    if link:
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(60, 300, 200, 315),
                          "uri": "https://example.com/support"})
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (10, 40, 90)).save(buf, format="PNG")
    if picture:
        page.insert_image(fitz.Rect(60, 340, 260, 520), stream=buf.getvalue())
    if icon:
        page.insert_image(fitz.Rect(60, 560, 74, 574), stream=buf.getvalue())
    return fitz.open("pdf", doc.tobytes())


def test_a_shaded_panel_is_measured_with_its_colour():
    plain = inventory.page_inventory(_page(), 0)
    noted = inventory.page_inventory(_page(shaded=True), 0)
    assert plain["panels"] == []
    assert len(noted["panels"]) == 1
    assert inventory._hex(noted["panels"][0]).startswith("#")


def test_a_hyperlink_is_measured_with_its_target():
    assert inventory.page_inventory(_page(), 0)["links"] == []
    assert inventory.page_inventory(_page(link=True), 0)["links"] == ["https://example.com/support"]


def test_a_small_icon_is_counted_not_listed_as_a_picture():
    """A row of icons groups differently on the two sides, so icons are a count
    and only real artwork is listed size by size."""
    # The artwork finder caches by id(doc), and CPython reuses ids for
    # short-lived documents - so a cached result from an earlier test can be
    # served for this one. Cleared here; see the note on the cache itself.
    from pdfval.validators.chapter import _PAGE_ART_CACHE

    _PAGE_ART_CACHE.clear()
    facts = inventory.page_inventory(_page(picture=True, icon=True), 0)
    assert all(min(size) >= inventory.PICTURE_MIN_SIDE for size in facts["images"])
    assert facts["icons"] >= 1


def test_text_is_measured_and_carried_for_the_model():
    facts = inventory.page_inventory(_page(words=20), 0)
    assert facts["words"] >= 20 and facts["lines"] >= 1
    assert "word" in facts["text"]


def test_the_fact_block_names_every_measured_kind():
    block = ai_review._facts_block(inventory.page_inventory(_page(shaded=True, link=True), 0))
    for label in ("text:", "pictures", "small icons", "shaded panels", "tables", "hyperlinks"):
        assert label in block


def test_a_missing_document_degrades_to_captions_alone():
    assert ai_review._facts_for(None, 0) is None


def test_a_panel_only_one_side_prints_is_a_gap():
    a = inventory.page_inventory(_page(shaded=True), 0)
    b = inventory.page_inventory(_page(), 0)
    gaps = inventory.compare_pages(a, b)
    assert any("shaded panel" in g and "not in Staging" in g for g in gaps)
    assert inventory.compare_pages(a, a) == []


def test_the_same_page_compared_with_itself_has_no_gaps():
    doc = _page(shaded=True, link=True, picture=True, icon=True)
    inv = inventory.page_inventory(doc, 0)
    assert inventory.compare_pages(inv, inv) == []


def test_a_typeface_only_one_document_uses_is_reported_once():
    import fitz
    a = fitz.open(); pa = a.new_page()
    pa.insert_text((60, 80), "hello", fontname="helv", fontsize=12)
    b = fitz.open(); pb = b.new_page()
    pb.insert_text((60, 80), "hello", fontname="tiro", fontsize=12)
    found = inventory.font_changes(fitz.open("pdf", a.tobytes()), fitz.open("pdf", b.tobytes()))
    assert found and all(f["type"] == "font-set" for f in found)
    assert any("Production" in f["title"] for f in found)


def test_identical_documents_have_no_font_gap():
    import fitz
    doc = _page()
    assert inventory.font_changes(doc, doc) == []


def test_headings_set_smaller_in_staging_are_reported_once():
    """Four headings, all set 18pt in one document and 12pt in the other: one
    finding for the document, not one per heading."""
    import fitz

    def book(size):
        doc = fitz.open()
        titles = ["Getting started", "Connections", "Using the menu", "Maintenance"]
        for n, t in enumerate(titles):
            page = doc.new_page()
            page.insert_text((60, 90), t, fontsize=size, fontname="hebo")
            page.insert_textbox(fitz.Rect(60, 120, 500, 300), "body text here " * 10, fontsize=11)
        doc.set_toc([[1, t, i + 1] for i, t in enumerate(titles)])
        return fitz.open("pdf", doc.tobytes())

    found = inventory.heading_style_changes(book(18), book(12))
    assert len(found) == 1 and found[0]["type"] == "heading-style"
    assert "18.0pt" in found[0]["description"] and "12.0pt" in found[0]["description"]


def test_headings_set_the_same_way_are_not_reported():
    import fitz

    def book():
        doc = fitz.open()
        titles = ["Getting started", "Connections", "Using the menu", "Maintenance"]
        for t in titles:
            page = doc.new_page()
            page.insert_text((60, 90), t, fontsize=16, fontname="hebo")
        doc.set_toc([[1, t, i + 1] for i, t in enumerate(titles)])
        return fitz.open("pdf", doc.tobytes())

    assert inventory.heading_style_changes(book(), book()) == []


def _text_page(*lines):
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for ln in lines:
        page.insert_text((60, y), ln, fontsize=11)
        y += 18
    return fitz.open("pdf", doc.tobytes())


def test_a_word_only_one_page_prints_is_a_gap():
    a = inventory.page_inventory(_text_page("the copyright year is 2026"), 0)
    b = inventory.page_inventory(_text_page("the copyright year is 2024"), 0)
    gaps = inventory.compare_pages(a, b)
    assert any("2026" in g and "Production" in g for g in gaps)
    assert any("2024" in g and "Staging" in g for g in gaps)


def test_a_word_carried_onto_the_neighbouring_page_is_not_missing():
    """The two documents break pages in different places; the tail of one page
    is the head of the other's next one, and that is not a word lost."""
    a = inventory.page_inventory(_text_page("shutting down the projector safely"), 0)
    b = inventory.page_inventory(_text_page("shutting down the"), 0)
    b["neighbours"] = [inventory._page_words({"full_text": "projector safely"})]
    assert inventory.compare_pages(a, b) == []


def test_a_real_url_on_one_page_only_is_a_gap():
    a = inventory.page_inventory(_page(link=True), 0)
    b = inventory.page_inventory(_page(), 0)
    assert any("example.com/support" in g for g in inventory.compare_pages(a, b))


def test_internal_links_are_not_counted_against_each_other():
    """One document's contents page carries dozens of internal links and the
    other's none - that is how they were built, not a gap."""
    import fitz
    doc = fitz.open(); page = doc.new_page()
    page.insert_text((60, 80), "contents", fontsize=11)
    for n in range(4):
        page.insert_link({"kind": fitz.LINK_GOTO, "from": fitz.Rect(60, 100 + n * 20, 200, 115 + n * 20),
                          "page": 0})
    linked = inventory.page_inventory(fitz.open("pdf", doc.tobytes()), 0)
    plain = inventory.page_inventory(_text_page("contents"), 0)
    assert not any("internal" in g for g in inventory.compare_pages(linked, plain))


def _list_page(gap, marker="2."):
    """One numbered item whose text starts `gap` points after its marker."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 100), "1.", fontsize=11)
    page.insert_text((80, 100), "SOURCE", fontsize=11)
    page.insert_text((60, 130), marker, fontsize=11)
    page.insert_text((72 + gap, 130), "POWER", fontsize=11)
    return fitz.open("pdf", doc.tobytes())


def test_a_wider_gap_after_a_marker_is_reported_with_both_measurements():
    found = inventory.marker_spacing_changes(_list_page(4), _list_page(40), [(1, 1)])
    assert len(found) == 1
    f = found[0]
    assert f["type"] == "marker-spacing" and "2." in f["title"]
    assert f["stage_boxes"] and f["prod_boxes"]          # boxed on both sides
    assert "wider" in f["description"]


def test_the_same_spacing_on_both_sides_is_not_reported():
    assert inventory.marker_spacing_changes(_list_page(4), _list_page(6), [(1, 1)]) == []


def test_a_number_in_a_table_cell_is_not_read_as_a_list_marker():
    """An appendix timing table is a grid of bare numbers; measured as markers
    every column read as a spacing change."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(40, 80, 520, 200))
    for n, x in enumerate((60, 200, 340)):
        page.draw_line(fitz.Point(x + 120, 80), fitz.Point(x + 120, 200))
        page.insert_text((x, 110), str(24 + n), fontsize=10)
        page.insert_text((x + 60, 110), "60.0", fontsize=10)
    page.draw_line(fitz.Point(40, 140), fitz.Point(520, 140))
    built = fitz.open("pdf", doc.tobytes())
    gaps = inventory.marker_gaps(built, 0)
    assert all(not k.isdigit() or float(v[0]) < inventory.MARKER_GAP_MAX_REACH
               for k, v in gaps.items())
