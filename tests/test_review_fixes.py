"""Regressions from reviewing the GP520 Production/Staging run."""
import fitz

from pdfval.validators import chapter as C
from pdfval.validators.chapter import KIND_TABLE, KIND_TEXT, Element


def _pdf(tmp_path, name, draw):
    doc = fitz.open()
    page = doc.new_page(width=420, height=595)
    draw(page)
    path = tmp_path / name
    doc.save(path)
    return fitz.open(path)


def test_space_from_a_line_wrap_is_not_a_space_change():
    # Production wraps "Support." / "BenQ.com" over two lines.
    wraps = tuple(C._wrap_pairs(["visit Support.", "BenQ.com for more"]))
    assert not C.mark_changes("visit Support. BenQ.com for more", "visit Support.BenQ.com for more", wraps, ())
    # Not wrapped there: a real space change.
    assert C.mark_changes("optical, chemical,manual or", "optical, chemical, manual or")


def test_repeat_removed_in_staging_is_not_settled_by_its_other_copy():
    own = " go to power energy power energy auto "
    assert not C._elsewhere_counted("power energy", own, " go to power energy auto ")
    assert C._elsewhere_counted("power energy", own, " power energy x power energy auto ")


def test_table_cross_reference_as_section_link_is_not_new_wording():
    x = "quick access to setting menu ( )."
    y = 'quick access to setting menu (see "quick access to setting menu").'
    assert C._cell_units(C._bare_xref(x)) == C._cell_units(C._bare_xref(y))
    assert C._bare_xref("hide. note: noise") == C._bare_xref("hide. note noise")


def test_row_number_in_its_own_cell_counts_as_the_marker():
    t = Element(kind=KIND_TABLE, cells=(("no.", "descriptions"), ("4.", "hdmi 1 input port see wired")))
    assert C._row_carries_marker(t, "hdmi 1 input port see wired projection", "4.")
    assert not C._row_carries_marker(t, "hdmi 1 input port see wired projection", "5.")


def test_italic_lost_in_staging_is_reported():
    a = Element(kind=KIND_TEXT, text="Press the Home key", key="press the home key", italic_words=("home",))
    b = Element(kind=KIND_TEXT, text="Press the Home key", key="press the home key")
    kinds = {d["type"] for d in C._compare_merged(a, b, 1.0)}
    assert "italic-missing" in kinds
    assert "italic-added" in {d["type"] for d in C._compare_merged(b, a, 1.0)}


def test_italic_span_detected_by_font_name():
    assert C._span_italic({"text": "word", "flags": 0, "font": "Roboto-Italic"})
    assert not C._span_italic({"text": "word", "flags": 0, "font": "Roboto-Regular"})


def test_raised_trademark_reads_as_the_symbol():
    spans = [{"text": ". Google TV", "size": 11.0, "origin": (0, 268.7)},
             {"text": "TM", "size": 6.4, "origin": (0, 263.4)},
             {"text": " brings together", "size": 11.0, "origin": (0, 268.7)}]
    assert "".join(s["text"] for s in C._raised_marks(spans)) == ". Google TV™ brings together"


def test_number_in_a_table_row_at_the_page_foot_is_not_a_page_number(tmp_path):
    def draw(page):
        page.insert_text((40, 560), "150", fontsize=9)
        page.insert_text((100, 560), "3810", fontsize=9)
        page.insert_text((200, 585), "28", fontsize=9)
    doc = _pdf(tmp_path, "foot.pdf", draw)
    by_text = {w[4]: tuple(w[:4]) for w in doc[0].get_text("words")}
    assert not C._alone_on_row(doc, 0, by_text["150"])
    assert C._alone_on_row(doc, 0, by_text["28"])


def test_table_word_crossing_a_column_rule_is_kept_whole(tmp_path):
    import pdfplumber

    from pdfval.extractor import _rows_by_whole_words

    def draw(page):
        for y in (100, 120, 140):
            page.draw_line((30, y), (300, y))
        for x in (30, 70, 300):
            page.draw_line((x, 100), (x, 140))
        page.insert_text((34, 114), "4.", fontsize=10)
        page.insert_text((60, 114), "OK", fontsize=10)
        page.insert_text((34, 134), "6.", fontsize=10)
        page.insert_text((60, 134), "Digital zoom", fontsize=10)
    _pdf(tmp_path, "cells.pdf", draw)
    with pdfplumber.open(tmp_path / "cells.pdf") as pdf:
        page = pdf.pages[0]
        t = page.find_tables()[0]
        rows = _rows_by_whole_words(page, t, [list(r.cells) for r in t.rows])
    assert rows == [["4.", "OK"], ["6.", "Digital zoom"]]


def test_line_wrap_hyphen_is_not_a_wording_change():
    ops = C._word_diff("the web man- agement interface via LAN.", "the web management interface via LAN")
    assert [o["text"] for o in ops if o["type"] != "equal"] == ["LAN.", "LAN"]


def test_dash_opening_a_line_is_a_list_marker_not_punctuation():
    wraps = tuple(C._wrap_pairs(["disabled when", "- a wireless device; or", "- you press"]))
    assert not C.mark_changes("disabled when - a wireless device; or - you press",
                              "disabled when a wireless device; or you press", wraps, ())
    assert C.mark_changes("1500 m - 2000 m", "1500 m 2000 m")


def test_cross_reference_split_across_lines_is_taken_out():
    text = 'auto cinema mode (see "optimizing image \nquality by auto cinema mode") | 19. netflix'
    assert "optimizing" not in C._xref_free(text)
    assert "netflix" in C._xref_free(text)


def test_repeated_sentence_is_not_absorbed_into_its_first_copy():
    whole = [Element(kind=KIND_TEXT, key="the product supports wifi the following is an example")]
    taken = [Element(kind=KIND_TEXT, key="the product supports wifi"),
             Element(kind=KIND_TEXT, key="the following is an example")]
    repeat = Element(kind=KIND_TEXT, key="the following is an example")
    assert C._unused_share(repeat, whole, taken) < C._ABSORB_MIN
    assert C._unused_share(repeat, whole, taken[:1]) >= C._ABSORB_MIN


def test_word_hyphenated_inside_a_table_cell_is_one_word():
    from pdfval.validators import camelot_tables as ct
    assert ct._norm("The receiv-\ner is off. Power sup-\nply.") == "the receiver is off power supply"


def test_value_cell_beside_a_merged_label_does_not_repeat_the_row_below():
    from pdfval.extractor import _drop_repeated_below
    rows = [["Temperature range", "Operating: 0°C\nStorage: -10°C"], [None, "Storage: -10°C"]]
    assert _drop_repeated_below(rows) == [["Temperature range", "Operating: 0°C"], [None, "Storage: -10°C"]]


def test_underline_lost_in_staging_is_never_reported():
    assert "underline-missing" in C._NEVER_REPORTED
    assert "underline-added" not in C._NEVER_REPORTED


def test_later_bullet_of_an_icon_note_belongs_to_that_note(monkeypatch):
    first = Element(kind=KIND_TEXT, text="• one", key="one", boxes=[(0, (84, 268, 434, 281))])
    second = Element(kind=KIND_TEXT, text="• two", key="two", boxes=[(0, (84, 280, 462, 293))])
    third = Element(kind=KIND_TEXT, text="• three", key="three", boxes=[(0, (84, 291, 527, 305))])
    far = Element(kind=KIND_TEXT, text="• far", key="far", boxes=[(0, (84, 400, 527, 414))])
    monkeypatch.setattr(C, "_has_note_icon", lambda doc, el: el is first)
    topic = [first, second, third, far]
    assert C._in_icon_note(None, topic, third)   # its note's icon is beside the first bullet
    assert not C._in_icon_note(None, topic, far)  # a paragraph further down is not in the note


def test_table_cell_number_is_not_read_as_a_diagram_callout(monkeypatch):
    monkeypatch.setattr(C, "_inside_table", lambda doc, page, bbox: True)
    line = {"text": "50", "bbox": (49, 612, 60, 624), "page": 0}
    assert not C._is_artwork_callout(object(), line)


def test_table_continued_without_its_head_is_reported(tmp_path):
    from pdfval.validators.chapter import table_header_repeats

    def grid(page, rows, top=30):
        for r, cells in enumerate(rows):
            y = top + r * 30
            page.draw_line((30, y), (400, y))
            page.draw_line((30, y + 30), (400, y + 30))
            for x in (30, 200, 400):
                page.draw_line((x, y), (x, y + 30))
            for c, text in enumerate(cells):
                page.insert_text((36 + c * 170, y + 20), text, fontsize=9)

    doc = fitz.open()
    first = doc.new_page(width=420, height=180)   # table runs to the foot of the page
    grid(first, [["Item", "Value"], ["Cable", "USB Type C"], ["LED", "Red"], ["Weight", "111g"]])
    second = doc.new_page(width=420, height=180)  # ... and continues with no head row
    grid(second, [["Power", "2.1W"], ["Band", "5GHz"], ["Jack", "DC 5V"]], top=15)
    path = tmp_path / "continued.pdf"
    doc.save(path)
    out = table_header_repeats(fitz.open(path), str(path), [0, 1])
    assert [d["type"] for d in out] == ["table-header-repeat"]
    assert "head row is not printed again" in out[0]["summary"]
