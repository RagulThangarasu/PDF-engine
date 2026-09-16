"""Build the small, deliberate PDF pairs the checks are tested against.

Each fixture isolates ONE behaviour so a test failure says what broke:

  tables_prod.pdf / tables_stage_ok.pdf     - long table, header repeated on
                                              every continuation page
  tables_stage_noheader.pdf                 - same table, continuation page
                                              does not reprint the header
  images_prod.pdf / images_stage_ok.pdf     - same figures, one pushed onto the
                                              next page inside its own section
  images_stage_swapped.pdf                  - one figure replaced by a
                                              different picture
  images_stage_missing.pdf                  - one figure removed outright
  list_prod.pdf / list_stage_detached.pdf   - one numbered procedure, with Staging
                                              drawing each marker as its own text
                                              object beside the step
  list_stage_lettered.pdf                   - that procedure relettered a,b,c
  list_stage_parenlettered.pdf              - that procedure relettered (a),(b),(c)
  list_stage_spacedletters.pdf              - that procedure relettered a ., b ., c .
  list_stage_bulleted.pdf                   - that procedure reduced to bullets
  chapters_prod.pdf                         - a contents page, two L1 chapters, a Q&A
                                              section and a footer on every page
  chapters_stage_reflowed.pdf               - the same words, narrower column, one
                                              paragraph cut by a page break, a
                                              reworded Q&A: nothing to report
  chapters_stage_edited.pdf                 - Production's layout, typo fixed, one
                                              paragraph removed
"""
from __future__ import annotations

import os

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

PAGE_W, PAGE_H = 612.0, 792.0
HEADER_ROW = ["Item", "Function", "Range"]
BODY_ROWS = [
    [
        f"Setting {i}",
        f"Adjusts behaviour {i} of the monitor and takes effect immediately once changed here",
        f"{i} - {i + 9}",
    ]
    for i in range(1, 82)
]

# Kept inside the body-text column (the lede paragraph below reaches ~550pt) so
# the table itself never trips the margin-overflow check.
COL_X = [72.0, 190.0, 370.0, 520.0]
ROW_H = 22.0
# Enough rows per page that the table runs into the page's bottom band, which
# is what makes it read as a table CONTINUING rather than as one that happens
# to end there - the same signal the validator uses.
ROWS_PER_PAGE = 27


def _draw_table_rows(page, rows: list[list[str]], top: float, right_x: float | None = None) -> float:
    """Draw ruled rows starting at `top`; returns the y the table ended at.
    `right_x` overrides the table's right edge (used to push it past the margin)."""
    cols = list(COL_X)
    if right_x is not None:
        cols[-1] = right_x
    y = top
    for row in rows:
        page.draw_line(pymupdf.Point(cols[0], y), pymupdf.Point(cols[-1], y), width=0.6)
        for c, text in enumerate(row):
            page.insert_text((cols[c] + 4, y + 15), text, fontsize=9)
        y += ROW_H
    page.draw_line(pymupdf.Point(cols[0], y), pymupdf.Point(cols[-1], y), width=0.6)
    for x in cols:
        page.draw_line(pymupdf.Point(x, top), pymupdf.Point(x, y), width=0.6)
    return y


def _table_document(repeat_header: bool, overflow_right: bool = False) -> pymupdf.Document:
    """A table long enough to need three pages, with a heading bookmarked so
    the validators' section anchoring has something to work with.
    `overflow_right` pushes the table's right edge past the page margin."""
    doc = pymupdf.open()
    per_page = ROWS_PER_PAGE
    chunks = [BODY_ROWS[i : i + per_page] for i in range(0, len(BODY_ROWS), per_page)]
    for _ in chunks:
        doc.new_page(width=PAGE_W, height=PAGE_H)
    # A page of ordinary body prose ahead of the table, so the margin check has
    # a real full-width text column to measure the page margin from (a document
    # that is nothing but a table is degenerate for that check).
    lede = (
        "This reference lists every setting available on the monitor together with "
        "a short description of what it does and the range of values it accepts. "
        "The table continues over several pages; the header row is repeated at the "
        "top of each page so every column stays labelled wherever you are reading."
    )
    running_header = (
        "Monitor user guide  -  settings reference section  -  this header repeats on every page"
    )
    for index, chunk in enumerate(chunks):
        page = doc[index]
        top = 90.0
        # A running header spanning the body column, on every page - real
        # manuals have one, and it is what lets the margin check establish the
        # page's content column independently of the table itself.
        page.insert_textbox(pymupdf.Rect(COL_X[0], 40, COL_X[-1], 60), running_header, fontsize=9)
        if index == 0:
            page.insert_text((COL_X[0], 74), "Settings reference", fontsize=14)
            page.insert_textbox(pymupdf.Rect(COL_X[0], 82, COL_X[-1], 150), lede, fontsize=9)
            top = 150.0
        rows = chunk
        if index == 0 or repeat_header:
            rows = [HEADER_ROW] + chunk
        # PAGE_W is 612; the body column ends at COL_X[-1]=520. Pushing the
        # table's right rule to 600 puts it ~76pt past the established margin.
        _draw_table_rows(page, rows, top, right_x=600.0 if overflow_right else None)
    doc.set_toc([[1, "Settings reference", 1]])
    return doc


def _draw_monitor(page, rect: pymupdf.Rect) -> None:
    """A distinctive line drawing: screen, stand, base, and a bezel button."""
    screen = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.62)
    page.draw_rect(screen, color=(0, 0, 0), width=1.4)
    page.draw_rect(
        pymupdf.Rect(screen.x0 + 8, screen.y0 + 8, screen.x1 - 8, screen.y1 - 8),
        color=(0.35, 0.35, 0.35),
        width=0.7,
    )
    neck_x = (rect.x0 + rect.x1) / 2
    page.draw_line(pymupdf.Point(neck_x - 9, screen.y1), pymupdf.Point(neck_x - 9, rect.y1 - 12), width=1.2)
    page.draw_line(pymupdf.Point(neck_x + 9, screen.y1), pymupdf.Point(neck_x + 9, rect.y1 - 12), width=1.2)
    page.draw_rect(
        pymupdf.Rect(rect.x0 + 20, rect.y1 - 12, rect.x1 - 20, rect.y1), color=(0, 0, 0), width=1.2
    )
    page.draw_circle(pymupdf.Point(screen.x1 - 20, screen.y1 - 18), 4, color=(0, 0, 0), width=1.0)


def _draw_cable(page, rect: pymupdf.Rect) -> None:
    """A clearly different picture: a plug, a looping lead, and a socket.

    Drawn from a good handful of separate strokes on purpose - a figure has to
    be more than a few primitives before the extractor treats a cluster of
    vector marks as an illustration rather than as a rule or a box.
    """
    plug = pymupdf.Rect(rect.x0, rect.y0 + rect.height * 0.35, rect.x0 + 46, rect.y0 + rect.height * 0.65)
    page.draw_rect(plug, color=(0, 0, 0), width=1.4)
    page.draw_rect(
        pymupdf.Rect(plug.x0 + 6, plug.y0 + 6, plug.x0 + 18, plug.y1 - 6), color=(0, 0, 0), width=1.0
    )
    page.draw_rect(
        pymupdf.Rect(plug.x0 + 24, plug.y0 + 6, plug.x0 + 36, plug.y1 - 6), color=(0, 0, 0), width=1.0
    )
    mid_y = rect.y0 + rect.height * 0.5
    page.draw_bezier(
        pymupdf.Point(plug.x1, mid_y),
        pymupdf.Point(rect.x0 + rect.width * 0.5, rect.y0),
        pymupdf.Point(rect.x0 + rect.width * 0.6, rect.y1),
        pymupdf.Point(rect.x1 - 26, mid_y),
        color=(0, 0, 0),
        width=1.6,
    )
    page.draw_circle(pymupdf.Point(rect.x1 - 16, mid_y), 14, color=(0, 0, 0), width=1.4)
    page.draw_circle(pymupdf.Point(rect.x1 - 16, mid_y), 6, color=(0, 0, 0), width=1.0)
    page.draw_line(
        pymupdf.Point(rect.x0 + 10, rect.y1 - 6), pymupdf.Point(rect.x1 - 10, rect.y1 - 6), width=1.0
    )


FIGURE_RECT_A = pymupdf.Rect(70, 120, 280, 300)
FIGURE_RECT_B = pymupdf.Rect(320, 120, 530, 300)
FIGURE_RECT_LOW = pymupdf.Rect(70, 560, 280, 740)


def _image_document(variant: str) -> pymupdf.Document:
    """Two pages under one heading, carrying three figures.

    variant "prod"    - all three figures on page 1
            "ok"      - the third figure pushed onto page 2, same section
            "swapped" - the third figure replaced by a different picture
            "missing" - the third figure removed
            "wider"   - the third figure is the same drawing, ~40% wider
    """
    doc = pymupdf.open()
    doc.new_page(width=PAGE_W, height=PAGE_H)
    doc.new_page(width=PAGE_W, height=PAGE_H)
    # Pages are fetched AFTER both exist: adding a page invalidates page handles
    # taken before it.
    page1, page2 = doc[0], doc[1]
    page1.insert_text((60, 60), "Assembly", fontsize=16)
    _draw_monitor(page1, FIGURE_RECT_A)
    _draw_cable(page1, FIGURE_RECT_B)
    page2.insert_text((60, 60), "continued", fontsize=9)

    if variant == "prod":
        _draw_monitor(page1, FIGURE_RECT_LOW)
    elif variant == "ok":
        _draw_monitor(page2, FIGURE_RECT_A)  # same picture, next page, same section
    elif variant == "swapped":
        _draw_cable(page2, FIGURE_RECT_A)  # a different picture in its place
    elif variant == "wider":
        wide = pymupdf.Rect(
            FIGURE_RECT_LOW.x0, FIGURE_RECT_LOW.y0, FIGURE_RECT_LOW.x0 + FIGURE_RECT_LOW.width * 1.4, FIGURE_RECT_LOW.y1
        )
        _draw_monitor(page1, wide)  # same drawing, ~40% wider
    # "missing": nothing drawn at all

    doc.set_toc([[1, "Assembly", 1]])
    return doc


_PARA_A = (
    "The 5-way controller is located below the lower part of the front bezel. "
    "While sitting in front of the monitor, move the controller to the directions "
    "instructed by the on-screen icons for menu navigation and operations."
)
_PARA_B = (
    "Use a clean part of the cloth to wipe dry the screen completely. If it is "
    "still not clean, continue with another clean part of the cloth to avoid "
    "spreading the grease around."
)
_PARA_C = "Repeat this step until the screen is clean."
_PARA_D = "Available menu options may vary depending on the input sources and settings."


def _wrapped(page, text: str, x: float, y: float, width: float) -> None:
    page.insert_textbox(pymupdf.Rect(x, y, x + width, y + 400), text, fontsize=10)


_PARA_E = (
    "Keep the shipping carton and packing materials for future use, in case you "
    "need to transport the monitor a long distance."
)


def _content_document(variant: str) -> pymupdf.Document:
    """One heading, two pages. Every fixture contains the SAME sentences - the
    only thing that changes is which page each lands on, and (in "glued")
    whether a space survives after a full stop.

    variant "prod"   - paragraphs B+C together at the top of page 1
            "ok"      - paragraph C reflowed onto page 2 (still same section)
            "glued"   - page 1 keeps "...completely.If" with the space dropped
            "edited"  - paragraph C is gone (a Missing), paragraph E is new (an Added)
    """
    doc = pymupdf.open()
    doc.new_page(width=PAGE_W, height=PAGE_H)
    doc.new_page(width=PAGE_W, height=PAGE_H)
    page1, page2 = doc[0], doc[1]
    page1.insert_text((60, 60), "Cleaning the screen", fontsize=16)
    _wrapped(page1, _PARA_A, 60, 90, 480)

    if variant == "glued":
        _wrapped(page1, _PARA_B.replace("completely. If", "completely.If") + " " + _PARA_C, 60, 200, 480)
        _wrapped(page2, _PARA_D, 60, 90, 480)
    elif variant == "ok":
        _wrapped(page1, _PARA_B, 60, 200, 480)
        _wrapped(page2, _PARA_C + " " + _PARA_D, 60, 90, 480)
    elif variant == "edited":
        # PARA_C is gone with nothing in its place (a pure Missing); PARA_E is
        # new, sitting after PARA_D where it pairs with nothing (a pure Added).
        _wrapped(page1, _PARA_B, 60, 200, 480)
        _wrapped(page2, _PARA_D + " " + _PARA_E, 60, 90, 480)
    else:  # prod
        _wrapped(page1, _PARA_B + " " + _PARA_C, 60, 200, 480)
        _wrapped(page2, _PARA_D, 60, 90, 480)

    doc.set_toc([[1, "Cleaning the screen", 1]])
    return doc


_STEPS = [
    "Disable Set time automatically.",
    "Select Date.",
    "Specify the date, and then select OK.",
    "Select Time.",
    "Specify the time, and then select OK.",
]
_LIST_LEAD = (
    "Perform one or more of the following actions to set the time the display "
    "shows while it is idle."
)


def _list_document(variant: str) -> pymupdf.Document:
    """One heading over a five-step procedure, typeset the two ways real
    documents typeset one.

    variant "prod"     - "1." ... "5." inline, at the start of each step's own
                         line (how Production sets a procedure)
            "detached" - the SAME 1-5 numbering, but each marker drawn as its
                         own text object to the left of the step (how Staging
                         sets one) - same content, nothing to report
            "lettered" - detached markers relettered a. ... e.
            "parenlettered" - detached markers relettered (a) ... (e), the
                         fully-parenthesised style some documents use for
                         sub-items instead of a trailing period or bracket
            "spacedletters" - detached markers relettered a . ... e ., a space
                         printed before the period some documents leave in
            "bulleted" - detached markers replaced by bullets
    """
    doc = pymupdf.open()
    doc.new_page(width=PAGE_W, height=PAGE_H)
    page = doc[0]
    page.insert_text((60, 60), "Setting the display time", fontsize=16)
    _wrapped(page, _LIST_LEAD, 60, 84, 480)
    markers = {
        "prod": [f"{i}." for i in range(1, 6)],
        "detached": [f"{i}." for i in range(1, 6)],
        "lettered": ["a.", "b.", "c.", "d.", "e."],
        "parenlettered": ["(a)", "(b)", "(c)", "(d)", "(e)"],
        "spacedletters": ["a .", "b .", "c .", "d .", "e ."],
        "bulleted": ["\u2022"] * 5,
    }[variant]
    y = 150.0
    for marker, step in zip(markers, _STEPS):
        if variant == "prod":
            page.insert_text((100, y), f"{marker}  {step}", fontsize=10)
        else:
            page.insert_text((100, y), marker, fontsize=10)
            page.insert_text((125, y), step, fontsize=10)
        y += 18.0
    doc.set_toc([[1, "Setting the display time", 1]])
    return doc


_GUIDE_FOOTER = "User guide"
_P_INTRO = (
    "The display supports a wide range of input sources and can be mounted in either "
    "orientation. Before installing it, check that the mounting surface can carry at "
    "least four times the weight of the display."
)
# One sentence, split where Staging's page break falls: the first half ends
# mid-sentence and the second half opens in lower case.
_P_INPUT_A = (
    "To change the input source, open the System Settings menu, select Input, and "
    "choose the source you want to use. The display remembers the last source"
)
_P_INPUT_B = "that was selected and returns to it the next time it is powered on."
_P_STEP_TYPO = "4. Nevigate to the bottom of the page and choose the desired action."
_P_STEP_FIXED = "4. Navigate to the bottom of the page and choose the desired action."
_P_CARE = (
    "Clean the screen with a soft, dry cloth. Never spray liquid directly onto the "
    "display surface, and never use cleaning products that contain alcohol."
)
_QA_TEXT = {
    "prod": "Why is there no picture? Check that the cable is connected. See page 3.",
    "reflowed": "Why is there no image on the screen? Check the HDMI cable. See page 4.",
}


def _chapter_document(variant: str) -> pymupdf.Document:
    """A small manual for Chapter Validation: a printed contents page, then
    two L1 chapters, the second ending in a Q&A section.

    variant "prod"     - 3 pages, 470pt text column
            "reflowed" - 4 pages, 360pt column (every paragraph wraps
                         differently), the input paragraph cut by a page break,
                         and the Q&A reworded - all of which must go unreported
            "edited"   - Production's layout with the typo fixed and the
                         cleaning paragraph removed - both must be reported
    """
    doc = pymupdf.open()
    pages = 4 if variant == "reflowed" else 3
    for _ in range(pages):
        doc.new_page(width=PAGE_W, height=PAGE_H)
    width = 360 if variant == "reflowed" else 470

    contents = doc[0]
    contents.insert_text((72, 80), "Table of contents", fontsize=18)
    for n, (title, page_no) in enumerate(
        (("Getting started", 2), ("Changing the input", 2), ("Care and cleaning", pages))
    ):
        contents.insert_text((72, 120 + 20 * n), f"{title} {'.' * 40} {page_no}", fontsize=10)

    first = doc[1]
    first.insert_text((72, 70), "Getting started", fontsize=18)
    first.insert_textbox(pymupdf.Rect(72, 90, 72 + width, 200), _P_INTRO, fontsize=10)
    first.insert_text((72, 240), "Changing the input", fontsize=14)
    if variant == "reflowed":
        first.insert_textbox(pymupdf.Rect(72, 690, 72 + width, 760), _P_INPUT_A, fontsize=10)
        carry = doc[2]
        carry.insert_textbox(pymupdf.Rect(72, 64, 72 + width, 110), _P_INPUT_B, fontsize=10)
        carry.insert_text((72, 150), _P_STEP_TYPO, fontsize=10)
        care = doc[3]
    else:
        first.insert_textbox(
            pymupdf.Rect(72, 255, 72 + width, 360), _P_INPUT_A + " " + _P_INPUT_B, fontsize=10
        )
        first.insert_text((72, 400), _P_STEP_FIXED if variant == "edited" else _P_STEP_TYPO, fontsize=10)
        care = doc[2]

    care.insert_text((72, 70), "Care and cleaning", fontsize=18)
    if variant != "edited":
        care.insert_textbox(pymupdf.Rect(72, 90, 72 + width, 200), _P_CARE, fontsize=10)
    care.insert_text((72, 240), "Q&A", fontsize=14)
    care.insert_textbox(
        pymupdf.Rect(72, 255, 72 + width, 320),
        _QA_TEXT["reflowed" if variant == "reflowed" else "prod"],
        fontsize=10,
    )

    for index in range(pages):
        doc[index].insert_text((72, 772), f"{_GUIDE_FOOTER}   {index + 1}", fontsize=8)

    care_page = pages
    doc.set_toc([
        [1, "Table of contents", 1],
        [1, "Getting started", 2],
        [2, "Changing the input", 2],
        [1, "Care and cleaning", care_page],
        [2, "Q&A", care_page],
    ])
    return doc


def _draw_gear_icon(page, rect: pymupdf.Rect, color=(0, 0, 0)) -> None:
    """A small inline icon - a ring, a hub, and a spoke - three primitives so
    the vector-figure detector reads it as artwork, not a stray rule. The
    spoke is drawn as a thin filled rect, not a zero-width line: a bare line
    has no area in one dimension and the vector detector reads that as a
    rule/underline, not artwork, and drops it - two primitives short of a
    figure at all."""
    page.draw_rect(rect, color=color, width=1.0)
    centre = pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    page.draw_circle(centre, rect.width * 0.3, color=color, width=1.0)
    page.draw_rect(
        pymupdf.Rect(centre.x - 1, rect.y0 + 2, centre.x + 1, rect.y1 - 2), color=color, fill=color, width=0
    )


_ICON_RECT = pymupdf.Rect(188, 94, 212, 118)


def _icon_document(variant: str) -> pymupdf.Document:
    """One page, one heading, and a line of text reading "Press the Settings
    [[icon]] option to continue." (Courier, so word widths are exact and the
    icon can be placed precisely beside "Settings" without overlapping it).

    variant "prod"    - the icon is drawn
            "missing" - the icon is not drawn at all
            "colour"  - the same icon, drawn in a different colour
    """
    doc = pymupdf.open()
    doc.new_page(width=PAGE_W, height=PAGE_H)
    page = doc[0]
    page.insert_text((72, 60), "Icons", fontsize=16)
    page.insert_text((72, 110), "Press the Settings", fontsize=10, fontname="courier")
    page.insert_text((220, 110), "option to continue.", fontsize=10, fontname="courier")
    if variant == "prod":
        _draw_gear_icon(page, _ICON_RECT)
    elif variant == "colour":
        _draw_gear_icon(page, _ICON_RECT, color=(0.7, 0, 0))
    # "missing": nothing drawn at all

    doc.set_toc([[1, "Icons", 1]])
    return doc


def main() -> None:
    os.makedirs(FIXTURES, exist_ok=True)
    written = []
    for name, doc in (
        ("tables_prod.pdf", _table_document(repeat_header=True)),
        ("tables_stage_ok.pdf", _table_document(repeat_header=True)),
        ("tables_stage_noheader.pdf", _table_document(repeat_header=False)),
        ("tables_stage_overflow.pdf", _table_document(repeat_header=True, overflow_right=True)),
        ("images_prod.pdf", _image_document("prod")),
        ("images_stage_ok.pdf", _image_document("ok")),
        ("images_stage_swapped.pdf", _image_document("swapped")),
        ("images_stage_missing.pdf", _image_document("missing")),
        ("images_stage_wider.pdf", _image_document("wider")),
        ("content_prod.pdf", _content_document("prod")),
        ("content_stage_reflowed.pdf", _content_document("ok")),
        ("content_stage_glued.pdf", _content_document("glued")),
        ("content_stage_edited.pdf", _content_document("edited")),
        ("list_prod.pdf", _list_document("prod")),
        ("list_stage_detached.pdf", _list_document("detached")),
        ("list_stage_lettered.pdf", _list_document("lettered")),
        ("list_stage_parenlettered.pdf", _list_document("parenlettered")),
        ("list_stage_spacedletters.pdf", _list_document("spacedletters")),
        ("list_stage_bulleted.pdf", _list_document("bulleted")),
        ("chapters_prod.pdf", _chapter_document("prod")),
        ("chapters_stage_reflowed.pdf", _chapter_document("reflowed")),
        ("chapters_stage_edited.pdf", _chapter_document("edited")),
        ("icons_prod.pdf", _icon_document("prod")),
        ("icons_stage_missing.pdf", _icon_document("missing")),
        ("icons_stage_colour.pdf", _icon_document("colour")),
    ):
        path = os.path.join(FIXTURES, name)
        doc.save(path)
        doc.close()
        written.append(path)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
