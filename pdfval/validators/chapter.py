"""8. Chapter validation: one whole L1 section, compared as it is printed.

Every other check in this engine asks a narrow question - is this sentence
still there, is this table the same shape, is this figure the same picture -
and each answers it over its own slice of the document. This one asks the
question a reviewer actually asks:

    Take the chapter that starts at this top-level (L1) heading, read it
    through to the next L1 heading, and tell me everything that is different
    between the two documents - the prose, the figures, the tables, the notes,
    in the order they are printed.

So a chapter is the unit, not a sentence or a page. The two documents paginate
differently (the pair this was built against runs 61 pages against 71), so the
chapter is collected as a CONTINUOUS run of elements across however many pages
it occupies on each side, and page boundaries are then forgotten: nothing here
compares "page 12 with page 12", and nothing is reported for content that
merely fell onto a different page.

WHAT IS COMPARED

Each side's chapter is reduced to an ordered list of `Element`s - heading,
paragraph, note, table, figure - each carrying what it says, how it is set
(size, weight, colour), and where it sits so it can be boxed on a page render.
The two lists are aligned by content, and every pair is compared for:

  * wording            - the text changed
  * emphasis / size    - the same words set bold, italic, or at another size
  * colour             - the same words in another colour
  * figures            - gone, replaced, or rendered at another size
  * tables             - another shape, or a cell whose content changed
  * notes              - a NOTE/TIP/WARNING that lost its label or changed type
  * order              - an element that moved relative to its neighbours

and anything on one side with no counterpart is reported as missing/added.

WHAT IS DELIBERATELY NOT REPORTED

Two documents that say exactly the same thing still differ in ways no reviewer
wants listed, and the check is worthless if it drowns them:

  * a paragraph that WRAPS differently - three lines here, four there. Lines
    are merged into paragraphs before anything is compared, so the wrap is
    invisible to the comparison.
  * a paragraph that CONTINUES onto the next page. A paragraph broken by a page
    break is stitched back together (see `_stitch_across_pages`), so it is one
    element on both sides however each document paginates it.
  * running headers, footers and page numbers - page furniture, one-sided in
    nearly every chapter because the two documents paginate differently.
  * printed page cross-references ("see page 42") - the number is expected to
    differ between documents of different lengths.
"""
from __future__ import annotations

from pdfval.docid import doc_key

import difflib
import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import fitz
from diff_match_patch import diff_match_patch

from pdfval import i18n, imagefp, visual_diff
from pdfval.extractor import _cluster_rects, get_figures, get_tables, get_vector_figures
from pdfval.models import CheckResult, Issue
from pdfval.validators.alignment import classify_marker
from pdfval.validators.headings import body_style, resolve_entries
from pdfval.validators.links import USABLE_SCHEMES, _named_destination_resolves, _uri_authority
from pdfval.validators.table import _document_margins, _headers_match, _stitch_continued_tables
from pdfval.validators.toc import (
    TocEntry,
    heading_at,
    _looks_like_qa_index_page,
    is_excluded_heading,
    is_qa_faq_heading,
    looks_like_toc_listing,
    match_toc_entries,
    normalize_title,
)

# --- what counts as an element --------------------------------------------

KIND_HEADING = "heading"
KIND_TEXT = "text"
KIND_NOTE = "note"
KIND_TABLE = "table"
KIND_FIGURE = "figure"

KIND_LABEL = {
    KIND_HEADING: "Heading",
    KIND_TEXT: "Text",
    KIND_NOTE: "Note",
    KIND_TABLE: "Table",
    KIND_FIGURE: "Figure",
}

# Switches to narrow the run: True reports tables only, or nothing at all.
# Both off is the full comparison - content, lists, images, links, tables.
_ONLY_TABLE_CHECKS = False
_VALIDATION_DISABLED = False

_BOLD_FLAG = 1 << 4
_ITALIC_FLAG = 1 << 1

# Page furniture - the running header, the footer, the page number. Not chapter
# content, and one-sided in nearly every chapter because the two documents
# paginate differently, so it has to come out before anything is compared.
#
# Recognised from the DOCUMENT, not from a fixed margin: the two PDFs here are
# 915pt and 842pt tall and put their page number at 8% and 2% from the bottom,
# so any band wide enough to catch Production's also eats its last line of real
# body text. What actually identifies furniture is that it REPEATS - the same
# text, or at least the same short line in the same place, page after page.
_FURNITURE_BAND = 0.15     # fraction of page height, top and bottom, to consider
_FURNITURE_MIN_PAGES = 3   # the same line this many pages over is furniture
_FURNITURE_BAND_SHARE = 0.5  # ... and so is a band filled on this share of pages
_FURNITURE_BANDS = 40      # y-quantisation: a fortieth of the page (~2.5%)
_FURNITURE_MAX_SIZE = 1.15  # furniture is body-sized; a chapter title is not
# The "same band on most pages" rule only reaches this far in from the edge.
# Staging's body text starts 7% down its pages, so a 15% band caught the first
# short line of body text on nearly every page as a "running header" - which is
# how "on the home page and launch File Manager" disappeared from a paragraph.
# Repeated TEXT is still recognised across the full 15%.
_FURNITURE_EDGE = 0.07
_CHROME_MAX_CHARS = 90

# Merging lines into paragraphs. A gap this much larger than the line's own
# height starts a new paragraph; anything closer is the same paragraph wrapping.
_PARAGRAPH_GAP = 0.75
_PARAGRAPH_SIZE_STEP = 1.0  # pt of size change that starts a new paragraph - a
                            # 13.5pt heading over 12pt body text is two elements
# How far apart two lines' x-ranges may sit and still count as the same column.
# A table row's own label, sorted in between two lines of a much wider adjacent
# cell purely because its y sits between them, has an x-range nowhere near
# either - "Update" (a row label at x0=147) landing inside "Make sure to save
# all [Update] important data before proceeding" (a callout at x0=273-584) is
# exactly this: two different columns, never two lines of one paragraph.
_PARAGRAPH_COLUMN_SLACK = 4.0
_PARAGRAPH_WIDE_COLUMN = 120.0  # pt: a line this wide belongs to a text column, not a label column
# A hanging indent shifts a wrapped line's start a little right of its bullet -
# tens of points, never hundreds. Two side-by-side columns of prose (Staging's
# safety symbols beside "THIS EQUIPMENT MUST BE GROUNDED") can sit close enough
# that their edges fall inside `_PARAGRAPH_COLUMN_SLACK` too, purely because the
# gutter between them is narrow - without also checking how far the START of
# the line jumped, that read as one paragraph, splicing one column's sentence
# into the middle of the other's.
_PARAGRAPH_X0_JUMP_MAX = 60.0
# A figure smaller than this on either side is a rule, a bullet or an icon.
_FIGURE_MIN_SIDE = 24.0

# Two elements pair up when their text is at least this alike. High, because a
# wrong pairing reports two unrelated paragraphs as "changed".
_PAIR_MIN_RATIO = 0.62
# A containment match is good evidence but weaker than the words matching
# outright, so it is discounted before being compared against the same floor.
_CONTAINMENT_WEIGHT = 0.85
# ... and a figure, which has no text, pairs on how it looks.
_FIGURE_PAIR_MIN = 0.55

# How much of a difference is worth reporting. Two faithful re-exports of one
# source disagree slightly about almost everything measurable, so each of these
# is a floor below which the finding would be noise, not news.
_EMPHASIS_STEP = 0.25      # fraction of the element's characters
_EMPHASIS_MIN_CHARS = 15   # below this an emphasis fraction means nothing
_SIZE_STEP = 0.15          # relative font-size change ...
_SIZE_MIN_PT = 1.0         # ... and at least this much of one, absolutely
_COLOUR_STEP = 48          # per-channel distance (0-255) that reads as another
                           # colour - a near-black is not "a colour change"
_FIGURE_SIZE_STEP = 0.20   # relative width/height change ...
_FIGURE_SIZE_MIN_PT = 12.0  # ... and at least this many points of it
_TABLE_EMPTY_ASYMMETRY = 0.3  # share of cells blank on one side only
_FIGURE_SAME = 0.80        # similarity at or above this is the same picture
_FIGURE_DIFFERENT = 0.45   # below this it is a different picture

MAX_ISSUES_PER_CHAPTER = 200  # a runaway chapter cannot swamp the report


@dataclass
class Element:
    """One printed thing inside a chapter, on one side."""

    kind: str
    text: str = ""                 # what it says, as printed
    key: str = ""                  # what it says, normalised for comparison
    boxes: list = field(default_factory=list)   # [(page, bbox)] - may span pages
    order: int = 0
    size: float = 0.0              # dominant font size, points
    bold: float = 0.0              # fraction of characters set bold
    italic: float = 0.0
    color: int = 0                 # dominant text colour, 0xRRGGBB
    label: str = ""                # a note's own label (NOTE / TIP / WARNING); a link's target
    detail: str = ""               # a link's reason for not working
    rows: int = 0                  # table shape
    cols: int = 0
    cells: tuple = ()              # table cell text, normalised, row by row
    width: float = 0.0             # figure size on the page, points
    height: float = 0.0
    fp: object | None = None       # figure appearance (imagefp.Fingerprint), trimmed to its ink
    raw_bbox: tuple | None = None  # a figure's box as detected, before trimming
    raw_fp: object | None = None   # ... and its appearance over that box
    header_fill: int | None = None # table header row background, 0xRRGGBB; None = none
    spans: tuple = ()              # merged cells per table row (cells a merge swallowed)
    bold_words: tuple = ()         # the words of this element that are set bold, in order
    underline_words: tuple = ()    # the words of this element drawn with an underline rule, in order
    italic_words: tuple = ()       # the words of this element set italic, in order
    wraps: tuple = ()              # (last word, first word) around each line/page break joined into the text
    grid: tuple = ()               # table cell columns per row: ((x0, x1) | None, ...)
    row_boxes: tuple = ()          # table rows: ((page, bbox), ...), one per row of `cells`
    section: str = ""              # the nearest heading above it that BOTH documents have (normalised)
    icon_column: bool = False      # a table printing its icons in a column of their own
    drawn: bool = False            # a figure drawn as vector shapes rather than an embedded image
    headers: tuple = ()            # a table's header cells, as printed
    ocr: bool = False              # text read by OCR off a scanned page (no text layer there)

    @property
    def page(self) -> int:
        return self.boxes[0][0] if self.boxes else 0

    @property
    def bbox(self) -> tuple:
        return self.boxes[0][1] if self.boxes else (0.0, 0.0, 0.0, 0.0)

    @property
    def pages(self) -> list[int]:
        return sorted({p for p, _ in self.boxes})


@dataclass
class Chapter:
    """One L1 heading's worth of document, on both sides."""

    title: str
    exp_pages: list[int]
    act_pages: list[int]
    exp_elements: list[Element]
    act_elements: list[Element]
    pairs: list = field(default_factory=list)     # [(exp elements, act elements, note, container)]
    # Every group in reading order, matched or not - the whole chapter, both
    # sides, which is what the browser lays out left against right.
    rows: list = field(default_factory=list)
    differences: list = field(default_factory=list)
    # L1 headings only one side treats as a chapter of its own - Production
    # bookmarks "Settings", Staging keeps the same material under the chapter
    # before it. Reported, but not allowed to split the comparison.
    exp_only_tops: list = field(default_factory=list)
    act_only_tops: list = field(default_factory=list)
    # Table-of-contents and Q&A content left out of this chapter, per side.
    skipped: dict = field(default_factory=dict)
    # Every topic both documents have: id -> {title, prod: (page, y), stage: (page, y)}.
    topics: dict = field(default_factory=dict)


# --- text normalisation ----------------------------------------------------

_WS_RE = re.compile(r"\s+")
# Control characters some producers embed in the text layer - Production opens
# every safety bullet with a BEL (\x07) glued to its first word, so "\x07To"
# never equalled Staging's "To" and read as "Staging adds “To”". Not printed, not content.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOFT_HYPHEN_RE = re.compile(r"(\w)[-‐­]\s+(\w)")
_PAGE_REF_RE = re.compile(
    r"\s*\b(?:on|see|refer to)?\s*page\s*\d{1,4}\b", re.IGNORECASE
)
# A list marker ("1.", "b)", "iv.", "•"). Taken out of the comparison key: the
# two documents draw markers differently - inside the item's text in one, as a
# separate text object beside it in the other - so a legend reads "1. Speakers
# 2. Power input" on one side and "1. Speakers Power input" on the other while
# printing the same thing. Marker STYLE is compared by Alignment Validation;
# here only the words count.
# An inline reference to a numbered callout on the facing diagram - "press the
# release button (1), detach the stand (2 and 3)". One export draws those
# numbers as circled vector glyphs and the other sets them as text, so the very
# same sentence reads "release button ( ), detach" here and "release button ( 1
# ), detach" there. Taken out of the comparison on BOTH sides: the numbers are
# printed either way, and which glyph draws them is the figure check's
# business, not the wording's. Only bare digits joined by "and"/"," qualify -
# and the empty pair a dropped glyph leaves behind.
_INLINE_CALLOUT_RE = re.compile(r"\((?:[\s,]|\d{1,2}|and\b)*\)", re.IGNORECASE)

_LIST_MARKER_TOKEN_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\((?:\d{1,2}|[a-z]|[ivx]{2,4})\)|(?:\d{1,2}|[a-z]|[ivx]{2,4}) ?[.)])(?=\s|$)"
    # The same marker with NO space after it: Production typesets the step as
    # "2.Remove the monitor stand." where Staging prints "2. Remove the monitor
    # stand." - the same words, the number just set tight against them. Taken
    # out on one side only, it read as a changed heading on every numbered step
    # in the manual. Digits only: a single letter would swallow "e.g." and a
    # roman numeral "i.e.".
    r"|(?:(?<=\s)|^)\d{1,2}[.)](?=[^\W\d_])"
    r"|[\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219]",
    re.IGNORECASE,
)


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


def _normalise(text: str, fold_case: bool = True) -> str:
    """What the element SAYS, with everything two faithful re-exports may
    legitimately disagree about taken out: the wrap hyphen at a line break, the
    printed page cross-reference whose number follows the pagination, curly
    against straight quotes, and full-width against ASCII glyphs.

    `fold_case=False` keeps capitalisation, for the one comparison that asks
    whether it changed."""
    text = i18n.fold_cjk_spaces(i18n.fold_width(i18n.fold_digits(_CONTROL_RE.sub("", text or ""))))
    text = _SOFT_HYPHEN_RE.sub(r"\1\2", text)
    text = _PAGE_REF_RE.sub(" ", text)
    text = _INLINE_CALLOUT_RE.sub(" ", text)
    # "5)" closing an open bracket is text, not a list marker: "HEVC(H26 5)".
    text = _LIST_MARKER_TOKEN_RE.sub(
        lambda m: m.group(0) if m.group(0).endswith(")") and text[:m.start()].count("(") > text[:m.start()].count(")")
        else " ",
        text,
    )
    text = _WS_RE.sub(" ", text.translate(_QUOTES)).strip()
    return text.casefold() if fold_case else text


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _loose(key: str) -> str:
    """A key with its punctuation gone - equal loose keys mean the two texts
    differ only in a colon, a full stop or a quote mark."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", key)).strip()


def _ends_open(text: str) -> bool:
    """True when this text does not look finished - the tell of a paragraph
    that carries on over a page break."""
    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] not in ".!?:;)。？！"


def _starts_continuation(text: str) -> bool:
    """True when this text does not look like the start of something new."""
    stripped = (text or "").lstrip()
    return bool(stripped) and (stripped[0].islower() or stripped[0] in ",;)")


# --- collecting a chapter's elements ---------------------------------------


_DIGITS_RE = re.compile(r"\d+")
# "46", "Page 46", "46 / 61", "46 of 61" once their numbers are masked.
_PAGE_NUMBER_KEY_RE = re.compile(r"^(?:page\s*)?#(?:\s*(?:/|of)\s*#)?$")
_FURNITURE_CACHE: dict[tuple, tuple[set, set]] = {}


def _furniture_key(text: str) -> str:
    """A line's text with its numbers masked, so a page number and a footer
    carrying one ("ST04_UM_V2_EN.indb 46") read as the same line on every page
    they appear on."""
    return _DIGITS_RE.sub("#", _WS_RE.sub(" ", text).strip().casefold())


def _furniture(doc: fitz.Document) -> tuple[set, set]:
    """`(repeated texts, occupied bands)` for this document's page furniture.

    Scanned once over the whole document: a line near the top or bottom edge
    that says the same thing (numbers aside) on several pages is a running
    header or footer, and a band near the edge that carries a short line on
    most pages is one even when its wording changes from section to section -
    which is exactly what a "chapter name" running head does.

    Only body-sized text is considered. A chapter title is printed at the top
    of its page, in the same band the running head uses, on as many pages as
    there are chapters - and it is the one thing in the band that must never be
    dropped, so size is what separates the two.
    """
    key = (doc_key(doc), doc.page_count)
    hit = _FURNITURE_CACHE.get(key)
    if hit is not None:
        return hit

    body, _ = body_style(doc)
    ceiling = (body or 0) * _FURNITURE_MAX_SIZE if body else 0.0
    texts: dict[str, set[int]] = {}
    bands: dict[int, set[int]] = {}
    for page_index in range(doc.page_count):
        rect = doc[page_index].rect
        top = rect.y0 + rect.height * _FURNITURE_BAND
        bottom = rect.y1 - rect.height * _FURNITURE_BAND
        try:
            data = doc[page_index].get_text("dict")
        except Exception:
            continue
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
                if not text or len(text) > _CHROME_MAX_CHARS:
                    continue
                y0, y1 = float(line["bbox"][1]), float(line["bbox"][3])
                if y1 > top and y0 < bottom:
                    continue  # body text, not an edge line
                if ceiling and _line_style(line)[0] > ceiling:
                    continue  # a heading printed at the top of its page
                # Keyed by edge: a running head repeats at the SAME edge. A body
                # line that merely spills to the top of one page ("Channels: 8")
                # and another at the bottom of a later one ("Channels: 2") is
                # content, and dropping it reported the line missing.
                texts.setdefault((_furniture_key(text), y0 < top), set()).add(page_index)
                if _at_edge((0.0, y0, 0.0, y1), rect):
                    bands.setdefault(_band_of(y0, rect), set()).add(page_index)

    pages = max(1, doc.page_count)
    out = (
        {t for (t, _), seen in texts.items() if len(seen) >= _FURNITURE_MIN_PAGES},
        {b for b, seen in bands.items() if len(seen) / pages >= _FURNITURE_BAND_SHARE},
    )
    _FURNITURE_CACHE[key] = out
    return out


def _band_of(y0: float, rect) -> int:
    return int(y0 / max(1.0, rect.height) * _FURNITURE_BANDS)


def reset_furniture_cache() -> None:
    _FURNITURE_CACHE.clear()
    _SQUASHED_PAGES.clear()
    _TABLE_BOX_CACHE.clear()
    _NAMES_CACHE.clear()
    _FIG_INDEX.clear()
    _MARGIN_CACHE.clear()
    _LABEL_WORDS.clear()
    _CHAR_CACHE.clear()
    _LINE_GEOM_CACHE.clear()
    _FIGURE_MIN_BY_DOC.clear()
    _FILL_CACHE.clear()
    _ICON_CACHE.clear()
    _UNDERLINE_CACHE.clear()
    _PAGE_ART_CACHE.clear()
    _ART_FP_CACHE.clear()


def _is_furniture(
    doc: fitz.Document, page_index: int, text: str, bbox: tuple, size: float, body: float
) -> bool:
    """Is this line the page's furniture rather than the chapter's content?"""
    if len(text) > _CHROME_MAX_CHARS:
        return False
    if body and size > body * _FURNITURE_MAX_SIZE:
        return False  # a heading, never a running head
    rect = doc[page_index].rect
    top = rect.y0 + rect.height * _FURNITURE_BAND
    bottom = rect.y1 - rect.height * _FURNITURE_BAND
    if bbox[3] > top and bbox[1] < bottom:
        return False  # inside the body of the page
    # A table's own cell - a row printed low on the page, or a value wrapped
    # onto a line of its own ("720(1440) x" / "480") - is never a page number
    # or running footer, however much it looks like one.
    if _inside_table(doc, page_index, bbox):
        return False
    texts, bands = _furniture(doc)
    key = _furniture_key(text)
    # Repeated text is furniture only in the outermost strip - or anywhere in
    # the edge band when it is shaped like a page number. A procedure that
    # opens "1. Open the System Settings menu." near the top of page after page
    # repeats too, and dropping it as a running header left every such
    # procedure one step short in Staging.
    if key in texts and (_at_edge(bbox, rect) or (_PAGE_NUMBER_KEY_RE.match(key)
                                                   and _alone_on_row(doc, page_index, bbox))):
        return True
    # A band holding text on most pages is where running headers sit - but in a
    # document set tight to the top of its pages it is also where every page's
    # first line of body text sits ("KONFORMITÄTSERKLÄRUNG", "• Direktiva LVD
    # 2014/35/EU"). A running header stands apart from the body; a first line
    # runs straight on into the next one.
    return _at_edge(bbox, rect) and _band_of(bbox[1], rect) in bands and _stands_apart(doc, page_index, bbox)


_TABLE_BOX_CACHE: dict[tuple, list] = {}


def _inside_table(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """This line sits inside a ruled table detected on the page."""
    key = (doc_key(doc), page_index)
    if key not in _TABLE_BOX_CACHE:
        from pdfval.extractor import _page_table_bboxes
        try:
            _TABLE_BOX_CACHE[key] = [tuple(b) for b in _page_table_bboxes(doc, page_index)]
        except Exception:
            _TABLE_BOX_CACHE[key] = []
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in _TABLE_BOX_CACHE[key])


def _alone_on_row(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """No other text shares this line's row: a page number prints by itself,
    a table's "150" has its row's other cells ("3810", "3321") beside it."""
    height = max(1.0, bbox[3] - bbox[1])
    for lb, _ in _text_lines(doc, page_index):
        if abs(lb[0] - bbox[0]) < 0.5 and abs(lb[1] - bbox[1]) < 0.5:
            continue  # the line itself
        overlap = min(lb[3], bbox[3]) - max(lb[1], bbox[1])
        if overlap >= 0.5 * height:
            return False
    return True


def _stands_apart(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """Is this line set apart from the page's body - a gap wider than line
    spacing between it and the nearest line toward the middle of the page?"""
    rect = doc[page_index].rect
    height = max(1.0, bbox[3] - bbox[1])
    upper = (bbox[1] + bbox[3]) / 2 < (rect.y0 + rect.y1) / 2
    gaps = [
        (lb[1] - bbox[3]) if upper else (bbox[1] - lb[3])
        for lb, _ in _text_lines(doc, page_index)
        if (lb[1] >= bbox[3] - 0.5 if upper else lb[3] <= bbox[1] + 0.5)
    ]
    return not gaps or min(gaps) >= max(4.0, 0.8 * height)


def _at_edge(bbox: tuple, rect) -> bool:
    """In the outermost strip of the page, where only furniture is printed."""
    return (
        bbox[3] <= rect.y0 + rect.height * _FURNITURE_EDGE
        or bbox[1] >= rect.y1 - rect.height * _FURNITURE_EDGE
    )


def _line_style(line: dict) -> tuple[float, float, float, int]:
    """(size, bold fraction, italic fraction, dominant colour) for a line."""
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return 0.0, 0.0, 0.0, 0
    total = sum(len(s["text"]) for s in spans) or 1
    size = max(spans, key=lambda s: len(s["text"])).get("size", 0.0)
    bold = sum(len(s["text"]) for s in spans if s.get("flags", 0) & _BOLD_FLAG) / total
    italic = sum(len(s["text"]) for s in spans if s.get("flags", 0) & _ITALIC_FLAG) / total
    color = max(spans, key=lambda s: len(s["text"])).get("color", 0)
    return float(size), bold, italic, int(color)


_LINE_SPACING_MIN_GAPS = 2  # fewer baseline gaps than this is not enough to know a block's own rhythm


def _line_spacing(doc: fitz.Document, el: "Element") -> float | None:
    """The median baseline-to-baseline gap between this element's own lines,
    as a multiple of its font size - its leading, read straight off the page.
    None for a single-line element (nothing to measure a gap between) or one
    that could not be read."""
    if el is None or not el.boxes:
        return None
    gaps: list[float] = []
    for page_index, bbox in el.boxes:
        try:
            lines = [ln for ln in _page_lines(doc, page_index, bbox[1] - 2, bbox[3] + 2, 0.0, None)
                     if _inside(ln["bbox"], [bbox], threshold=0.5)]
        except Exception:
            continue
        lines.sort(key=lambda ln: ln["base"])
        for prev, cur in zip(lines, lines[1:]):
            size = prev["size"] or cur["size"]
            gap = cur["base"] - prev["base"]
            if size > 0 and gap > 0:
                gaps.append(gap / size)
    if len(gaps) < _LINE_SPACING_MIN_GAPS:
        return None
    gaps.sort()
    return gaps[len(gaps) // 2]


def _inside(bbox: tuple, regions: list[tuple], threshold: float = 0.6) -> bool:
    x0, y0, x1, y1 = bbox
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    for rx0, ry0, rx1, ry1 in regions:
        ix0, iy0 = max(x0, rx0), max(y0, ry0)
        ix1, iy1 = min(x1, rx1), min(y1, ry1)
        if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) / area >= threshold:
            return True
    return False


_SUPERSCRIPT_FLAG = 1 << 0
_FOOTNOTE_REF_RE = re.compile(r"^\s*[\d*†‡§]{1,3}\s*$")

_UNDERLINE_CACHE: dict[tuple, list] = {}
_UNDERLINE_MAX_HEIGHT = 2.5  # pt: a rule this thin or thinner, sitting right under text, is an underline
_UNDERLINE_OVERSHOOT_MAX = 20.0  # pt: a rule may run this much past either end of the line and still be its underline
# A punctuation-stripped word that marks the phrase around it as a URL/link's
# visible text, not prose - "https", "www" and the dot before a domain never
# survive tokenising ("https", "www", "benq", "com" print as four bare words),
# so a phrase carrying any of these is a link, already judged by `links.py`.
_URL_LIKE_WORDS = {"http", "https", "www"}


def _page_underlines(doc: fitz.Document, page_index: int) -> list:
    """Every thin horizontal rule on this page - a candidate underline drawn
    beneath whatever text sits just above it. A PDF has no "underline" font
    flag the way bold or italic do; an underline is always a separately drawn
    line, found the same way `_has_underline` finds one under a hyperlink."""
    key = (doc_key(doc), page_index)
    if key not in _UNDERLINE_CACHE:
        try:
            _UNDERLINE_CACHE[key] = [
                d["rect"] for d in doc[page_index].get_drawings()
                if d.get("rect") is not None and d["rect"].height <= _UNDERLINE_MAX_HEIGHT and d["rect"].width > 2
            ]
        except Exception:
            _UNDERLINE_CACHE[key] = []
    return _UNDERLINE_CACHE[key]


def _line_underlined(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """A thin rule running under most of this line's width, close enough
    beneath its baseline to be its underline rather than a border or a rule
    under a different line.

    A NOTE/TIP/WARNING panel's own top and bottom border is drawn the same
    way - a thin, wide horizontal rule - and can sit close enough above or
    below the panel's first or last line of text to pass the overlap and
    gap checks too. What tells the two apart is reach: an underline is drawn
    to fit the words it marks, so it does not run far past either end of the
    line; a panel border runs the width of the whole panel, well past this
    one line sitting inside it.
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    if width <= 0:
        return False
    for r in _page_underlines(doc, page_index):
        overlap = min(x1, r.x1) - max(x0, r.x0)
        overshoot = max(0.0, x0 - r.x0) + max(0.0, r.x1 - x1)
        if overlap >= 0.5 * width and overshoot <= _UNDERLINE_OVERSHOOT_MAX and y1 - 1 <= r.y0 <= y1 + 4:
            return True
    return False


_RAISED_MARKS = {"tm": "\u2122", "sm": "\u2120", "r": "\u00ae", "(r)": "\u00ae"}


def _raised_marks(spans: list[dict]) -> list[dict]:
    """A trademark mark set as small raised letters ("Google TV" + a raised
    "TM") read as the mark itself ("Google TV™"), glued to the word before it
    - the way the other document prints it with the ™ character."""
    body = max((float(sp.get("size") or 0) for sp in spans), default=0.0)
    out: list[dict] = []
    for sp in spans:
        text = (sp.get("text") or "").strip()
        mark = _RAISED_MARKS.get(text.casefold())
        base = (sp.get("origin") or (0.0, 0.0))[1]
        others = [(o.get("origin") or (0.0, 0.0))[1] for o in spans if o is not sp and (o.get("text") or "").strip()]
        if mark and body and float(sp.get("size") or 0) < 0.8 * body and others and base < max(others) - 1.0:
            if out:
                out[-1] = {**out[-1], "text": (out[-1].get("text") or "").rstrip() + mark}
                continue
            sp = {**sp, "text": mark}
        out.append(sp)
    return out


def _footnote_ref(span: dict) -> bool:
    return bool(span.get("flags", 0) & _SUPERSCRIPT_FLAG) and bool(_FOOTNOTE_REF_RE.match(span.get("text") or ""))


def _page_lines(
    doc: fitz.Document, page_index: int, y0: float, y1: float, body_size: float = 0.0,
    titles: set[str] | None = None,
) -> list[dict]:
    """Every text line printed between `y0` and `y1` on this page, in reading
    order, with its geometry and styling - minus the page furniture."""
    page = doc[page_index]
    out: list[dict] = []
    from pdfval import ocr
    if ocr.is_scanned_page(doc, page_index):
        return _scanned_page_lines(doc, page_index, y0, y1, body_size)
    try:
        data = page.get_text("dict")
    except Exception:
        return out
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            # A footnote reference set as a superscript ("jack¹") is a pointer
            # to a note, not a word: Staging marks its connector legend with
            # them and Production does not, and glued onto the word they read
            # as "jack" -> "jack1", a content change that isn't one.
            line = {**line, "spans": [{**sp, "text": _CONTROL_RE.sub("", sp.get("text") or "")}
                                      for sp in line.get("spans", []) if not _footnote_ref(sp)]}
            line["spans"] = _raised_marks(line["spans"])
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not text:
                continue
            bx0, by0, bx1, by1 = (float(v) for v in line["bbox"])
            centre = (by0 + by1) / 2
            if centre < y0 - 1 or centre > y1 + 1:
                continue
            size, bold, italic, color = _line_style(line)
            # A section heading printed once is never furniture, however close to
            # the top of the page it sits: Staging starts "TCO Certified" 39pt down
            # an A4 page, inside the band its running headers use, and dropping it
            # there reported the heading missing from every such section. A title
            # repeated page after page is still a running header.
            is_title = (
                bool(titles) and normalize_title(text) in titles
                and _furniture_key(text) not in _furniture(doc)[0]
            )
            if not is_title and _is_furniture(doc, page_index, text, (bx0, by0, bx1, by1), size, body_size):
                continue  # running header / footer / page number
            spans = line.get("spans") or []
            base = max((float((sp.get("origin") or (0.0, by1))[1]) for sp in spans if (sp.get("text") or "").strip()),
                       default=by1)
            out.append({
                "text": text, "bbox": (bx0, by0, bx1, by1), "page": page_index, "base": base,
                "size": size, "bold": bold, "italic": italic, "color": color,
                "bold_words": tuple(
                    w for sp in line.get("spans", []) if sp.get("flags", 0) & _BOLD_FLAG
                    for w in _TOKEN_RE.findall((sp.get("text") or "").casefold())
                ),
                "underline_words": (
                    tuple(_TOKEN_RE.findall(text.casefold()))
                    if _line_underlined(doc, page_index, (bx0, by0, bx1, by1)) else ()
                ),
                "italic_words": tuple(
                    w for sp in line.get("spans", []) if _span_italic(sp)
                    for w in _TOKEN_RE.findall((sp.get("text") or "").casefold())
                ),
            })
    # Reading order by baseline, left to right along each printed line: a
    # superscript ("Google TV" + raised "TM brings together") tops out above
    # the rest of its line, and ordering by the top edge put that fragment
    # first - "TM brings together 6. When prompted..." - a sentence neither
    # document prints.
    out.sort(key=lambda ln: (ln["base"], ln["bbox"][0]))
    rows: list[list[dict]] = []
    for ln in out:
        if rows and abs(ln["base"] - rows[-1][0]["base"]) <= _SAME_BASELINE:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: ln["bbox"][0])]


def _scanned_page_lines(
    doc: fitz.Document, page_index: int, y0: float, y1: float, body_size: float,
) -> list[dict]:
    """A scanned page's lines, read by OCR. The page has no text layer, so
    without this every word on it would silently go uncompared. Style is not
    knowable from pixels - bold/italic/colour are left neutral so no style
    finding is ever raised from a guess."""
    from pdfval import ocr
    out: list[dict] = []
    for bbox, text in ocr.ocr_lines(_page_words(doc).ocr_words(page_index)):
        text = _CONTROL_RE.sub("", text).strip()
        centre = (bbox[1] + bbox[3]) / 2
        if not text or centre < y0 - 1 or centre > y1 + 1:
            continue
        size = max(1.0, (bbox[3] - bbox[1]) * 0.8)
        if _is_furniture(doc, page_index, text, bbox, size, body_size):
            continue
        out.append({
            "text": text, "bbox": bbox, "page": page_index, "base": bbox[3],
            "size": size, "bold": 0.0, "italic": 0.0, "color": 0,
            "bold_words": (), "underline_words": (), "italic_words": (), "ocr": True,
        })
    return out


def _dominant_size(lines: list[dict]) -> float:
    """The size most of the paragraph's characters are set at - not the largest
    line in it, which a single big drop-cap or a inline symbol would decide."""
    weights: dict[float, int] = {}
    for ln in lines:
        weights[round(ln["size"], 1)] = weights.get(round(ln["size"], 1), 0) + len(ln["text"])
    return max(weights.items(), key=lambda kv: kv[1])[0] if weights else 0.0


_ROW_GAP_EM = 3.0  # fragments of one printed line sit at most this many ems apart
_SAME_BASELINE = 0.5  # pt: pieces of one printed line share a baseline within this


def _merge_rows(lines: list[dict]) -> list[dict]:
    """Join the pieces of one printed line back into the line.

    An inline icon breaks a line's text into separate fragments - Production's
    "Select More [icon] on the home page and launch File Manager [icon]." comes
    out as "More" (bold), "on the home page and launch File Manager" and "." -
    and a paragraph cannot be judged by its style from a one-word fragment: the
    bold "More" alone looked like a heading and split the paragraph in two.
    """
    out: list[dict] = []
    for line in lines:
        prev = out[-1] if out else None
        if prev is not None:
            p0, p1 = prev["bbox"][1], prev["bbox"][3]
            l0, l1 = line["bbox"][1], line["bbox"][3]
            height = min(p1 - p0, l1 - l0)
            overlap = min(p1, l1) - max(p0, l0)
            gap = line["bbox"][0] - prev["bbox"][2]
            reach = _ROW_GAP_EM * max(prev["size"], line["size"], 1.0)
            # Pieces whose baselines differ are lines of two text columns side
            # by side (Production's safety symbols beside "THIS EQUIPMENT MUST
            # BE GROUNDED", 2pt apart; Staging's "electrician." 4pt off "alert"),
            # not one line broken by an icon - those pieces share a baseline.
            # Gluing them interleaved the columns word by word.
            side_by_side_columns = abs(prev.get("base", p1) - line.get("base", l1)) > _SAME_BASELINE
            if height > 0 and overlap >= 0.6 * height and -2.0 <= gap <= reach and not side_by_side_columns:
                a_len, b_len = len(prev["text"]), len(line["text"])
                total = (a_len + b_len) or 1
                joiner = "" if line["text"][:1] in ".,;:)!?\u2122\u2120\u00ae" else " "
                prev.update({
                    "text": prev["text"] + joiner + line["text"],
                    "bbox": (
                        min(prev["bbox"][0], line["bbox"][0]), min(p0, l0),
                        max(prev["bbox"][2], line["bbox"][2]), max(p1, l1),
                    ),
                    "size": prev["size"] if a_len >= b_len else line["size"],
                    "bold": (prev["bold"] * a_len + line["bold"] * b_len) / total,
                    "italic": (prev["italic"] * a_len + line["italic"] * b_len) / total,
                    "color": prev["color"] if a_len >= b_len else line["color"],
                    "bold_words": tuple(prev.get("bold_words", ())) + tuple(line.get("bold_words", ())),
                    "underline_words": tuple(prev.get("underline_words", ())) + tuple(line.get("underline_words", ())),
                    "italic_words": tuple(prev.get("italic_words", ())) + tuple(line.get("italic_words", ())),
                    "ocr": bool(prev.get("ocr") or line.get("ocr")),
                })
                continue
        out.append(dict(line))
    return out


def _paragraphs(
    lines: list[dict], heading_titles: set[str] | None = None,
    doc: fitz.Document | None = None,
) -> list[Element]:
    """Merge lines into paragraphs.

    This is where "the same paragraph wrapped differently" stops being a
    difference: the comparison never sees a line, only the paragraph it belongs
    to, so three lines here against four there is the same element.
    """
    out: list[tuple] = []

    def flush(run: list[dict]) -> None:
        if not run:
            return
        text = " ".join(ln["text"] for ln in run)
        clean = _WS_RE.sub(" ", text).strip()
        if doc is not None and _looks_like_extraction_garbage(clean):
            box = (
                min(ln["bbox"][0] for ln in run), min(ln["bbox"][1] for ln in run),
                max(ln["bbox"][2] for ln in run), max(ln["bbox"][3] for ln in run),
            )
            recovered = _ocr_recover_text(doc, run[0]["page"], box)
            if recovered and not _looks_like_extraction_garbage(recovered):
                text = clean = recovered
        chars = sum(len(ln["text"]) for ln in run) or 1
        out.append(((round(run[0]["bbox"][1], 1), run[0]["bbox"][0]), Element(
            kind=KIND_TEXT,
            text=clean,
            key=_normalise(text),
            boxes=[(run[0]["page"], (
                min(ln["bbox"][0] for ln in run), min(ln["bbox"][1] for ln in run),
                max(ln["bbox"][2] for ln in run), max(ln["bbox"][3] for ln in run),
            ))],
            size=_dominant_size(run),
            bold=sum(ln["bold"] * len(ln["text"]) for ln in run) / chars,
            italic=sum(ln["italic"] * len(ln["text"]) for ln in run) / chars,
            color=run[0]["color"],
            bold_words=tuple(w for ln in run for w in ln.get("bold_words", ())),
            underline_words=tuple(w for ln in run for w in ln.get("underline_words", ())),
            italic_words=tuple(w for ln in run for w in ln.get("italic_words", ())),
            wraps=tuple(_wrap_pairs([ln["text"] for ln in run])),
            ocr=any(ln.get("ocr") for ln in run),
        )))

    def same_column(prev: dict, line: dict) -> bool:
        # Two lines of one paragraph share a column - their x-ranges
        # overlap, or nearly do (a hanging indent shifts a wrapped line a
        # little right of its bullet). A line from another column entirely
        # has no such overlap, however close its y happens to fall - and,
        # when two columns run side by side with only a narrow gutter
        # between them, may pass the edge-gap check anyway; the START of the
        # two lines is then far apart (a genuine hanging indent moves it only
        # a little), which same_column also has to rule out.
        x_gap = max(prev["bbox"][0], line["bbox"][0]) - min(prev["bbox"][2], line["bbox"][2])
        x0_jump = abs(prev["bbox"][0] - line["bbox"][0])
        return x_gap <= _PARAGRAPH_COLUMN_SLACK and x0_jump <= _PARAGRAPH_X0_JUMP_MAX

    def continues(run: list[dict], line: dict) -> bool:
        prev = run[-1]
        height = max(1.0, prev["bbox"][3] - prev["bbox"][1])
        gap = line["bbox"][1] - prev["bbox"][3]
        # A bold line followed by a regular one is a heading and its
        # paragraph, even when they sit as close as two wrapped lines - but
        # only when the bold line is heading-length: a long bold line is
        # emphasis inside a paragraph, not a title above one.
        weight_flip = len(prev["text"]) <= 90 and (
            (prev["bold"] >= 0.8 and line["bold"] <= 0.2)
            or (prev["bold"] <= 0.2 and line["bold"] >= 0.8)
        )
        # A line that IS a heading title stands alone: Staging sets
        # "User interface" and "Home screen" a line apart at one size, and
        # read as one paragraph they paired with neither heading.
        titled = bool(heading_titles) and (
            normalize_title(prev["text"]) in heading_titles
            or normalize_title(line["text"]) in heading_titles
        )
        # A line that opens with its OWN list marker ("a. Detach...", "3.
        # Decide...") starts a new item, never continues the run above it - a
        # genuinely wrapped continuation line never begins with a fresh
        # marker. Without this, "B. Grommet mounting" merged straight into
        # "a. Detach the C-clamp...", "b. Install the stand..." and every
        # item after it into ONE element spanning the whole list: each
        # item's own marker was still found inside that blob, but every
        # item's `pages`/search scope became the WHOLE merged blob's, so the
        # holder search on the other side matched the wrong neighbouring
        # sentence (or nothing at all) and a genuine a./b./c. -> i./ii./iii.
        # renumbering went unreported.
        new_item = bool(_ITEM_MARKER_RE.match(line["text"]))
        return (
            gap <= height * _PARAGRAPH_GAP
            and abs(line["size"] - prev["size"]) <= _PARAGRAPH_SIZE_STEP
            and not weight_flip
            and not titled
            and not new_item
            and same_column(prev, line)
        )

    # Text set in two wide columns side by side (Staging's safety symbols
    # beside "THIS EQUIPMENT MUST BE GROUNDED") comes out of the page line by
    # line across both columns. Closing the paragraph at every switch cut each
    # column into one-line scraps read in alternation - "power outlet please
    # consult a qualified equilateral triangle is intended to alert" - so each
    # wide column keeps its own paragraph open while the other is read. Narrow
    # label columns (a term beside its description) keep the old behaviour.
    wide = lambda ln: ln["bbox"][2] - ln["bbox"][0] >= _PARAGRAPH_WIDE_COLUMN  # noqa: E731
    runs: list[list[dict]] = []
    for line in lines:
        target = next((r for r in reversed(runs) if continues(r, line)), None)
        if target is None:
            if wide(line):
                closing = [r for r in runs if same_column(r[-1], line) or not wide(r[-1])]
            else:
                closing = list(runs)
            for r in closing:
                flush(r)
                runs.remove(r)
            target = []
            runs.append(target)
        target.append(line)
    for r in runs:
        flush(r)
    out.sort(key=lambda item: item[0])
    return [el for _, el in out]


# "NOTE:", "Important", "TIP" printed on their own line: the label of a callout,
# set apart from its text in one document and run into it in the other. Not
# content of its own - the callout's text is compared, and a changed callout
# type is reported as such.
_BARE_CALLOUT_RE = re.compile(
    r"^\s*(" + "|".join(sorted({re.escape(w) for w in i18n._CALLOUT_WORDS}, key=len, reverse=True)) + r")\s*[:：]?\s*$",
    re.IGNORECASE,
)
# The same label word, but leading a sentence it was never split apart from
# ("Note Refer to..." - one printed line, no colon) rather than standing
# alone on its own. Stripped, not matched-and-discarded: what is left of the
# sentence is still worth comparing on its own honest terms.
_LEADING_CALLOUT_RE = re.compile(
    r"^\s*(?:" + "|".join(sorted({re.escape(w) for w in i18n._CALLOUT_WORDS}, key=len, reverse=True)) + r")\s*[:：]?\s+",
    re.IGNORECASE,
)


def _is_fragment(el: Element) -> bool:
    """A diagram's callout number ("2", "3.") printed beside the artwork rather
    than inside it. Not a sentence, and extracted as its own text on one side
    and glued to its label on the other - so it would be reported as missing or
    added in nearly every diagram chapter while meaning nothing."""
    text = el.text.strip()
    return len(text) <= 3 and not any(ch.isalpha() for ch in text)


def _ocr_recover_text(doc: fitz.Document, page_index: int, bbox) -> str:
    """Best-effort reading of a region whose text layer came back unreadable -
    a subset font with no (or a broken) ToUnicode map, most often on a
    multi-language compliance/RoHS table. OCR reads the rendered glyphs
    directly, bypassing the font's character map entirely, so it still works
    where the text layer cannot. Empty wherever Tesseract is unavailable or
    the region genuinely has no legible text; the caller keeps the original
    (unreadable) extraction in that case rather than losing the row."""
    from pdfval import ocr

    try:
        words = ocr.words_in(_page_words(doc).ocr_words(page_index), tuple(float(v) for v in bbox), margin=2)
    except Exception:
        return ""
    return " ".join(words)


def _table_element(table: dict, page_index: int, doc: fitz.Document | None = None) -> Element:
    rows = [[_normalise(c or "") for c in row] for row in (table.get("rows") or [])]
    text = " | ".join(
        " ".join((c or "").strip() for c in row if (c or "").strip())
        for row in (table.get("rows") or []) if any((c or "").strip() for c in row)
    )
    if doc is not None and _looks_like_extraction_garbage(text):
        recovered = _ocr_recover_text(doc, page_index, table["bbox"])
        if recovered and not _looks_like_extraction_garbage(recovered):
            text = recovered
    return Element(
        kind=KIND_TABLE,
        text=text,
        key=_normalise(text),
        boxes=[(page_index, tuple(float(v) for v in table["bbox"]))],
        rows=len(rows),
        cols=max((len(r) for r in rows), default=0),
        cells=tuple(tuple(r) for r in rows),
        spans=tuple(sum(1 for c in row if c is None) for row in (table.get("cells") or [])),
        grid=tuple(
            tuple((float(c[0]), float(c[2])) if c else None for row_cell in [row] for c in row_cell)
            for row in (table.get("cells") or [])
        ),
        icon_column=bool(table.get("icon_column")),
        headers=tuple((c or "").strip() for c in ((table.get("rows") or [[]])[0] or []) if (c or "").strip()),
        row_boxes=tuple(
            (page_index, _row_bbox(cells, table["bbox"]))
            for cells in list(table.get("cells") or []) + [[]] * max(0, len(rows) - len(table.get("cells") or []))
        )[: len(rows)],
    )


_TABLE_ANCHOR_MIN_ROWS = 2      # smaller than this is barely a table, a weak landmark
_TABLE_SIGNATURE_CHARS = 400    # enough of a big table's own words to tell it apart from a
                                 # near-identical sibling ("with speaker" vs "without"), not so
                                 # much that two huge, mostly-shared tables blur together


def _document_tables(doc: fitz.Document, pdf_path: str) -> list[Element]:
    """Every table in the document, page by page - regardless of any heading's
    exclusion. A scroll-sync landmark needs the tables `collect_elements`
    deliberately drops from a skipped section (a per-country RoHS declaration,
    a Q&A index...) just as much as any other, since the reader still scrolls
    past them."""
    out: list[Element] = []
    for page_index in range(doc.page_count):
        try:
            regions = get_tables(pdf_path, page_index, doc)
        except Exception:
            continue
        for t in regions:
            el = _table_element(t, page_index, doc)
            if el.rows >= _TABLE_ANCHOR_MIN_ROWS and el.key:
                out.append(el)
    out.sort(key=lambda e: (e.page, e.bbox[1]))
    return out


def _page_y_share(doc: fitz.Document, page_index: int, y: float) -> float:
    height = doc[page_index].rect.height or 1.0
    return max(0.0, min(1.0, y / height))


def table_anchors(
    expected: fitz.Document, actual: fitz.Document, expected_path: str, actual_path: str
) -> list[dict]:
    """Extra scroll-sync waypoints, one per table BOTH documents carry - the
    surest landmark a chapter has between two headings, and the only kind of
    waypoint left once a whole section (a per-country RoHS declaration, a
    Q&A index) is excluded from Content Validation altogether: a chapter that
    crams several such tables onto one page in Production and spreads them
    across several in Staging drifts out of step between its surrounding
    headings without one - Production's "with speaker" table lines up next to
    Staging's "without speaker" copy purely because both happen to sit at the
    same fraction of the way through the chapter.

    Matched the same way headings are - by text, in document order, tolerant
    of either side having a table the other does not - so same-shaped
    siblings (a "without speaker" table against a "with speaker" one, nine
    rows shared and one added) still pair with their own true counterpart
    rather than each other.
    """
    exp_tables = _document_tables(expected, expected_path)
    act_tables = _document_tables(actual, actual_path)
    if not exp_tables or not act_tables:
        return []
    exp_entries = [TocEntry(level=0, title=e.key[:_TABLE_SIGNATURE_CHARS], page=e.page, y=e.bbox[1])
                   for e in exp_tables]
    act_entries = [TocEntry(level=0, title=e.key[:_TABLE_SIGNATURE_CHARS], page=e.page, y=e.bbox[1])
                   for e in act_tables]
    out: list[dict] = []
    for m in match_toc_entries(exp_entries, act_entries):
        if m.expected_index is None or m.actual_index is None:
            continue
        exp_el, act_el = exp_tables[m.expected_index], act_tables[m.actual_index]
        label = (exp_el.cells[0][0] if exp_el.cells and exp_el.cells[0] else "") or "Table"
        out.append({
            "title": f"Table — {label}"[:60], "level": 0, "sync_only": True,
            "prod": {"page": exp_el.page + 1, "y": _page_y_share(expected, exp_el.page, exp_el.bbox[1])},
            "stage": {"page": act_el.page + 1, "y": _page_y_share(actual, act_el.page, act_el.bbox[1])},
        })
    return out


# Paragraph-level scroll-sync waypoints. Headings and tables are landmarks a
# chapter has a handful of; between two of them the viewer can only interpolate
# by proportion, and two documents that paginate the same chapter differently
# (59 pages against 52) drift apart in the middle of it - the reader sees
# Production's page 53 beside Staging's 47. Every matched paragraph is a
# landmark, and there are hundreds of them, so the two panes can track each
# other line for line instead of chapter for chapter.
_CONTENT_ANCHORS_PER_PAGE = 4    # more than this on one Production page is more than the eye needs
_CONTENT_ANCHOR_GAP = 24.0       # pt: two waypoints closer than this are one place


def content_anchors(chapters: list, expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A sync-only waypoint per matched paragraph, figure or table - anything
    `pair_elements` paired up and both sides actually print.

    Only genuinely matched pairs qualify: a group one side has alone says
    nothing about where the other side is, and a pair marked "moved" says the
    opposite of what a waypoint means.
    """
    out: list[dict] = []
    per_page: Counter = Counter()
    last_y: dict[int, float] = {}
    for chapter in chapters or []:
        for pair in getattr(chapter, "pairs", None) or []:
            exp_group, act_group = pair[0], pair[1]
            note = pair[2] if len(pair) > 2 else ""
            if not exp_group or not act_group or note == "moved":
                continue
            a, b = exp_group[0], act_group[0]
            if not getattr(a, "boxes", None) or not getattr(b, "boxes", None):
                continue
            exp_page, exp_bbox = a.boxes[0]
            act_page, act_bbox = b.boxes[0]
            if not (0 <= exp_page < expected.page_count and 0 <= act_page < actual.page_count):
                continue
            if per_page[exp_page] >= _CONTENT_ANCHORS_PER_PAGE:
                continue
            if abs(exp_bbox[1] - last_y.get(exp_page, -1e9)) < _CONTENT_ANCHOR_GAP:
                continue
            per_page[exp_page] += 1
            last_y[exp_page] = exp_bbox[1]
            out.append({
                "title": (getattr(a, "text", "") or "")[:60], "level": 0, "sync_only": True,
                "prod": {"page": exp_page + 1, "y": _page_y_share(expected, exp_page, exp_bbox[1])},
                "stage": {"page": act_page + 1, "y": _page_y_share(actual, act_page, act_bbox[1])},
            })
    return out


_COLUMN_GAP = 40.0   # body lines starting this far apart sit in different columns
_ICON_COLUMN_GAP = 6.0   # pt between an icon column and the first text column
_ICON_MAX_SIDE = 40.0    # pt: larger artwork is a figure, not a table icon
# A "Warning" label this far into the table's width sits inside a description
# cell (Production's Update row), not under the table where a callout ends it.
_CALLOUT_IN_CELL_SHARE = 0.3
_HAS_WORD_RE = re.compile(r"\w", re.UNICODE)  # not _WORD_RE: that name is a whitespace splitter further down


def _fill_unruled_cells(doc: fitz.Document, page_index: int, table: dict) -> dict:
    """Value cells the grid detector could not see, filled from the page.

    Production's spec tables rule only the label column: the value columns
    have no vertical lines, so the extractor gives each row its label cell and
    None for the rest - and the values ("43''", "4 × Arm Cortex-A73") were
    compared against nothing, while the text filter dropped them as table text.
    The header row still has every column's box, so a missing cell is rebuilt
    from its column's x-range and its row's height, and filled with the words
    printed there."""
    rows = [list(r) for r in (table.get("rows") or [])]
    cells = [list(c) for c in (table.get("cells") or [])]
    if len(cells) < 2 or not cells[0] or not all(cells[0]):
        return table
    head = cells[0]
    try:
        words = doc[page_index].get_text("words")
    except Exception:
        return table
    filled = 0
    for i in range(1, min(len(cells), len(rows))):
        real = [c for c in cells[i] if c]
        if not real:
            continue
        y0, y1 = min(c[1] for c in real), max(c[3] for c in real)
        for j, c in enumerate(cells[i]):
            if c is not None or j >= len(head) or j >= len(rows[i]):
                continue
            hx0, _, hx1, _ = head[j]
            # A cell another cell of the row already spans is merged, not
            # unseen: Staging's shipping row prints ST5504 and ST6504 in one
            # cell, and refilling the empty half split that merge in two.
            if any(rc[0] < hx1 - 2 and rc[2] > hx0 + 2 for rc in real):
                continue
            text = " ".join(
                w[4] for w in sorted(words, key=lambda w: (round(w[1]), w[0]))
                if hx0 <= (w[0] + w[2]) / 2 <= hx1 and y0 <= (w[1] + w[3]) / 2 <= y1
            )
            if text:
                rows[i][j] = text
                cells[i][j] = (hx0, y0, hx1, y1)
                filled += 1
    if not filled:
        return table
    return {**table, "rows": rows, "cells": cells}


def _extend_header_only(doc: fitz.Document, page_index: int, table: dict, y_end: float) -> dict:
    """A table the extractor read as its header row alone, extended down
    through the body rows printed under it.

    Production draws its menu tables as a shaded header bar over body rows
    separated by hairlines only - no vertical rules - so the grid detector sees
    the bar and nothing else, and the rows became loose text compared against
    Staging's real table ("start time" missing) or its label lines. Body rows
    are the lines under the bar, inside its width, until a gap or a line that
    starts left of the table. A new row starts where a first-column line
    begins below everything printed in the row so far, so a wrapped cell
    ("Power save mode" / "start time") stays one cell."""
    rows = table.get("rows") or []
    if len(rows) != 1:
        return table
    tx0, _, tx1, ty1 = (float(v) for v in table["bbox"])
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return table
    lines, header_xs = [], []
    ty0 = float(table["bbox"][1])
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = _CONTROL_RE.sub("", "".join(s.get("text", "") for s in line.get("spans", []))).strip()
            x0, y0, x1, y1 = (float(v) for v in line["bbox"])
            if text and y0 >= ty0 - 1 and y1 <= ty1 + 1 and _HAS_WORD_RE.search(text):
                header_xs.append(x0)
            if text and y0 >= ty1 - 1 and (y0 + y1) / 2 <= y_end + 1:
                size = max((float(s.get("size", 0)) for s in line.get("spans", [])), default=0.0)
                lines.append((y0, x0, y1, x1, text, size))
    lines.sort()
    # Another table's header bar under this one ends it.
    try:
        bars = [float(d["rect"].y0) for d in doc[page_index].get_drawings()
                if d.get("fill") and d["rect"].y0 > ty1 + 2 and d["rect"].width >= 0.8 * (tx1 - tx0)
                and 8 <= d["rect"].height <= 40]
    except Exception:
        bars = []
    stop_y = min(bars, default=float("inf"))
    body, bottom, body_size = [], ty1, None
    for y0, x0, y1, x1, text, size in lines:
        # A WARNING/NOTE box inset inside a row's description cell prints at
        # its own, smaller size - Production's "Update" row carries one - and
        # is still that row, not the end of the table. Only a size change near
        # the label column signals real furniture starting below the table.
        near_label = x0 <= tx0 + _CALLOUT_IN_CELL_SHARE * (tx1 - tx0)
        if (y0 - bottom > 0.9 * (y1 - y0) or x0 < tx0 - 2 or x1 > tx1 + 4 or y1 > stop_y + 2
                or (body_size is not None and abs(size - body_size) > 0.6 and near_label)
                or (_BARE_CALLOUT_RE.match(text) and near_label)):
            break  # a gap, text outside the table, a heading or a callout at the table's edge
        body_size = size if body_size is None else body_size
        body.append((y0, x0, y1, x1, text))
        bottom = max(bottom, y1)
    if not body:
        return table
    # Column starts come from where a printed line begins, not from every
    # fragment: "2. Select [icon] ." breaks into pieces, and the "." after the
    # icon is not a third column.
    begins: list[float] = []
    for y0, x0, y1, x1, text in body:
        if not _HAS_WORD_RE.search(text):
            continue  # "/" between two icons, a lone "." - never a column of its own
        before = [b for b in body if abs(b[0] - y0) < 2 and b[3] <= x0 + 1 and b is not None and b[1] < x0]
        if not before or x0 - max(b[3] for b in before) > 30:
            begins.append(x0)
    starts: list[float] = []
    for x in sorted(begins):
        if not starts or x - starts[-1] > _COLUMN_GAP:
            starts.append(x)
    # The body can print MORE columns than the header rules - Production's
    # spec table rules only "Menu item" | "Description", but "Description"'s
    # own width holds an unruled item-name column ahead of the description
    # text. Extending on the body's own column count (not the header's) is
    # what lets `_printed_columns` see and report that real mismatch instead
    # of the table staying just its header row, unboxed past that point.
    if len(starts) < 2:
        return table
    col_of = lambda x: max([i for i, s in enumerate(starts) if x >= s - 1] or [0])  # noqa: E731
    bounds = [tx0] + [s - 3 for s in starts[1:]] + [tx1]
    # Icons printed in a column of their own, left of the first text column and
    # not under the header text ("[⌄]/[⌃] | Expand/Collapse | Show/Hide…"): a
    # printed column the text-only column finder cannot see. Staging draws the
    # same icons inside the label cell, and without this the two tables read as
    # the same 2 columns when Production prints 3.
    icon_column = False
    try:
        icons = [tuple(float(v) for v in info["bbox"]) for info in doc[page_index].get_image_info()]
    except Exception:
        icons = []
    icons = [b for b in icons
             if b[0] >= tx0 - 2 and b[2] <= starts[0] - _ICON_COLUMN_GAP and b[1] >= ty1 - 1 and b[3] <= bottom + 1
             and (b[2] - b[0]) <= _ICON_MAX_SIDE and (b[3] - b[1]) <= _ICON_MAX_SIDE]
    if len(icons) >= 2 and header_xs and min(header_xs) >= max(b[2] for b in icons) + 4:
        icon_column = True
    # The hairlines Production rules between rows are the row boundaries when
    # it draws them: a description whose first line sits above its label
    # ("1. Open the Settings menu" over "Settings menu") is still that row.
    try:
        rules = sorted({round(float(d["rect"].y0), 1) for d in doc[page_index].get_drawings()
                        if not d.get("fill") and d["rect"].height < 1.5 and d["rect"].width > 20
                        and d["rect"].x0 >= tx0 - 2 and d["rect"].x1 <= tx1 + 2 and ty1 + 2 < d["rect"].y0 < bottom})
    except Exception:
        rules = []
    band_of = lambda y0, y1: sum(1 for r in rules if r < (y0 + y1) / 2)  # noqa: E731
    new_rows, new_cells, texts, row_y, band = [], [], None, None, None
    for y0, x0, y1, x1, text in body:
        col = col_of(x0)
        starts_row = texts is not None and (
            (band_of(y0, y1) != band) if rules else (col == 0 and y0 >= row_y[1] - 2 and bool(texts[0]))
        )
        if texts is None or starts_row:
            band = band_of(y0, y1)
            if texts is not None:
                new_rows.append([" ".join(t) for t in texts])
                new_cells.append([(bounds[i], row_y[0], bounds[i + 1], row_y[1]) for i in range(len(starts))])
            texts, row_y = [[] for _ in starts], [y0, y1]
        texts[col].append(text)
        row_y[1] = max(row_y[1], y1)
    new_rows.append([" ".join(t) for t in texts])
    new_cells.append([(bounds[i], row_y[0], bounds[i + 1], row_y[1]) for i in range(len(starts))])
    return {**table,
            "rows": list(rows) + new_rows,
            "cells": list(table.get("cells") or [[]])[:1] + new_cells,
            "bbox": (tx0, float(table["bbox"][1]), tx1, bottom),
            "icon_column": icon_column}


def _row_bbox(cells: list, fallback: tuple) -> tuple:
    """A table row's box: the union of its cells, or the table's box when the
    extractor gave the row no cells."""
    real = [c for c in (cells or []) if c]
    if not real:
        return tuple(float(v) for v in fallback)
    return (float(min(c[0] for c in real)), float(min(c[1] for c in real)),
            float(max(c[2] for c in real)), float(max(c[3] for c in real)))


def _header_bbox(table: dict) -> tuple | None:
    cells = [c for c in ((table.get("cells") or [[]])[0] or []) if c]
    if not cells:
        return None
    return (min(c[0] for c in cells), min(c[1] for c in cells),
            max(c[2] for c in cells), max(c[3] for c in cells))


_FILL_COVER = 0.5   # a filled shape must cover this share of the header row


def _header_fill(doc: fitz.Document, page_index: int, bbox: tuple | None) -> int | None:
    """The background colour painted behind a table's header row, or None when
    the header row sits on plain paper. White and near-white count as none."""
    if bbox is None:
        return None
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    best, best_cover = None, 0.0
    try:
        drawings = doc[page_index].get_drawings()
    except Exception:
        return None
    for d in drawings:
        fill = d.get("fill")
        rect = d.get("rect")
        if not fill or rect is None:
            continue
        cover = _intersection(bbox, tuple(rect)) / area
        if cover > best_cover:
            best, best_cover = fill, cover
    if best is None or best_cover < _FILL_COVER or min(best[:3]) >= 0.95:
        return None
    r, g, b = (int(round(v * 255)) for v in best[:3])
    return (r << 16) | (g << 8) | b


_ARTWORK_TEXT_SIZE = 0.85   # text set below this share of body size is a label
_ARTWORK_OVERLAP = 0.3      # share of a line's area lying on an image to be "on" it


_TEXT_FIGURE_SHARE = 0.2    # rows of text covering this share of a "figure" make it a text block
_PICTURE_SHARE = 0.5        # ... unless an embedded image covers this much of it


def _intersection(a: tuple, b: tuple) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _rect_area(r: "fitz.Rect") -> float:
    """A `fitz.Rect`'s area without `Rect.get_area()` - not on every PyMuPDF
    release a bare `PyMuPDF>=X` floor in requirements.txt still installs;
    `width`/`height` are on every version there has ever been."""
    return r.width * r.height


_TABLE_FIGURE_OVERLAP = 0.4  # a "figure" this much inside a table is the table's rules
_TABLE_FIGURE_SHARE = 0.5    # ... but only counted when it also fills this much of the table
                             # itself - a picture in one cell of a much bigger table is not
                             # a vector tracing of the table's own rules
_TEXT_FIGURE_ROWS = 4        # this many rows of text inside a "figure" make it a text block


_FILL_BAND_MAX_HEIGHT = 40.0  # pt
_FILL_BAND_MIN_ASPECT = 10.0


def _is_fill_band(fig, rows: list[dict]) -> bool:
    """A table header or banner fill - a long, low drawn bar with a line of
    text printed on it ("Item | Function | Range") - is not a picture."""
    x0, y0, x1, y1 = (float(v) for v in fig.bbox)
    width, height = x1 - x0, y1 - y0
    if getattr(fig, "kind", "") != "vector" or not 0 < height <= _FILL_BAND_MAX_HEIGHT \
            or width / height < _FILL_BAND_MIN_ASPECT:
        return False
    return any(y0 <= (r["bbox"][1] + r["bbox"][3]) / 2 <= y1 and r["bbox"][0] < x1 and r["bbox"][2] > x0
               for r in rows)


def _mostly_text(bbox: tuple, rows: list[dict], artwork: list[tuple],
                 tables: list[tuple] | None = None, all_rows: list[dict] | None = None) -> bool:
    """A "figure" that is really a block of text. Vector drawing clustering
    reads the rules and icons of a UI-element table as one large figure; kept,
    it is compared as a picture against nothing and reported as a figure only
    one document has, on top of the table rows it contains."""
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if sum(_intersection(bbox, r) for r in artwork) / area >= _PICTURE_SHARE:
        return False
    # A table's header fill and a callout's background panel are drawings too,
    # and clustered into one "figure" with the table beneath them. Measured
    # against only the text kept for prose, the table's own cells did not
    # count, and the table was reported as a figure only Staging has. But a
    # checklist/spec table also legitimately PRINTS a picture inside one of its
    # own cells (an "item name | item picture" unboxing table) - that picture
    # is a small fraction of the table's own area, not a vector tracing of the
    # table's rules, so it only counts here when the figure itself fills most
    # of the table(s) it sits in.
    if tables:
        covering = [t for t in tables if _intersection(bbox, t) > 0]
        if sum(_intersection(bbox, t) for t in covering) / area >= _TABLE_FIGURE_OVERLAP:
            table_area = sum((t[2] - t[0]) * (t[3] - t[1]) for t in covering)
            if area >= _TABLE_FIGURE_SHARE * max(table_area, 1e-6):
                return True
    text = all_rows if all_rows is not None else rows
    # A tall table or callout panel is mostly white space between its rows, so
    # area alone undercounts it: several full text rows inside the region make
    # it a block of text, whatever share of its area they cover.
    inside = [row for row in text if _inside(row["bbox"], [bbox], threshold=0.8)]
    if len(inside) >= _TEXT_FIGURE_ROWS:
        return True
    return sum(_intersection(bbox, row["bbox"]) for row in text) / area >= _TEXT_FIGURE_SHARE


def _raster_rects(doc: fitz.Document, page_index: int) -> list[tuple]:
    """Where embedded images are drawn on this page."""
    try:
        return [
            tuple(float(v) for v in info["bbox"])
            for info in doc[page_index].get_image_info()
            if info.get("bbox")
        ]
    except Exception:
        return []


def _on_artwork(row: dict, artwork: list[tuple], body_size: float) -> bool:
    """Is this line part of a picture rather than of the page?"""
    if body_size and row["size"] < body_size * _ARTWORK_TEXT_SIZE:
        return True
    return _inside(row["bbox"], artwork, threshold=_ARTWORK_OVERLAP)


_TRIM_MIN_X_OVERLAP_SHARE = 0.5  # a line must have at least this much of its OWN width inside the
                                  # figure's x-range to count as printed on it - a left column's text
                                  # wrapping a few points into a right-column figure's gutter, in a
                                  # two-column layout, only ever grazes the edge, never most of the line


def _trim_figure(bbox: tuple, lines: list[dict]) -> tuple:
    """Pull a figure's box off the prose lines kept inside its top or bottom
    quarter, so the figure is measured - and boxed - as the artwork alone."""
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    for ln in lines:
        lx0, ly0, lx1, ly1 = ln["bbox"]
        overlap = min(lx1, x1) - max(lx0, x0)
        if overlap <= 0 or overlap < _TRIM_MIN_X_OVERLAP_SHARE * max(1.0, lx1 - lx0) or ly1 <= y0 or ly0 >= y1:
            continue
        if ly0 - bbox[1] <= 0.25 * height:
            y0 = max(y0, ly1)
        elif bbox[3] - ly1 <= 0.25 * height:
            y1 = min(y1, ly0)
    return (x0, y0, x1, y1) if y1 - y0 >= _FIGURE_MIN_SIDE else bbox


_LEADING_ICON_GAP = 12.0  # pt: this close to where a line of text starts, it is that line's own icon


def _piece_of_matched(fig: "Element", figs: list["Element"], matched: set[int]) -> bool:
    """Is this "missing" picture only a slice of one already found?

    One document builds an illustration out of strips - Production's OSD
    screenshot is a title bar, a body and a footer, each its own image - where
    the other prints the whole thing as one picture. The big piece matches and
    the strips butted against it are left over, each reported missing though
    every pixel of them is printed on the other side inside the picture that
    did match."""
    if not fig.boxes:
        return False
    page_index, bbox = fig.boxes[0]
    mine = fitz.Rect(bbox) + (-3, -3, 3, 3)
    for j in matched:
        other = figs[j] if 0 <= j < len(figs) else None
        if other is None or not other.boxes:
            continue
        page_other, bbox_other = other.boxes[0]
        if page_other == page_index and mine.intersects(fitz.Rect(bbox_other)):
            return True
    return False


def _note_badges(doc: fitz.Document, fig: "Element") -> bool:
    """Is this "picture" really the small badge (or column of badges) that
    marks a note, each one set immediately before a line of text?

    Production draws its note icon as vector shapes, and two notes one under
    the other cluster into a single tall, narrow figure that no longer looks
    icon-sized to `_is_leading_icon`. Staging marks the same notes its own
    way - its own icon and a printed "NOTE:" / "TIP:" heading - so reporting
    the badge as a picture Staging is missing is just the two documents'
    callout styling, which is never expected to match."""
    if not fig.boxes or fig.width > _ICON_MAX_SIDE:
        return False
    page_index, (x0, y0, x1, y1) = fig.boxes[0]
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return False
    beside = 0
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            if not "".join(sp.get("text", "") for sp in line.get("spans", [])).strip():
                continue
            lx0, ly0, lx1, ly1 = line["bbox"]
            centre = (ly0 + ly1) / 2
            if y0 - 2 <= centre <= y1 + 2 and 0 <= lx0 - x1 <= _LEADING_ICON_GAP:
                beside += 1
    return beside > 0


def _is_leading_icon(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """A small image sitting immediately before a line of text - Production's
    pencil icon marking a NOTE it never spells out in words - is that line's
    own icon, not a figure of its own.

    Checked in absolute points, never scaled to body text size: an icon is
    icon-sized on every page it prints on, whatever the surrounding prose is
    set at. `_figure_min_side` scales the OTHER way - up for a large body, down
    for a small one, so a real figure that is only small because its whole
    page is small (a logo set in a document with 5pt body text) is not
    dropped - and on a page set that small, the same scaling can pull an
    ordinary icon above the lowered bar. This check catches that case back.
    """
    x0, y0, x1, y1 = bbox
    if x1 - x0 > _ICON_MAX_SIDE or y1 - y0 > _ICON_MAX_SIDE:
        return False
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return False
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not text:
                continue
            lx0, ly0, lx1, ly1 = line["bbox"]
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap >= 0.5 * min(y1 - y0, ly1 - ly0) and 0 <= lx0 - x1 <= _LEADING_ICON_GAP:
                return True
    return False


# A page whose body text was exported with its glyphs drawn as vector paths
# rather than kept as real characters (a font-embedding restriction, most
# often seen on regulatory/compliance pages carrying several languages) reads
# as one huge, densely-packed "figure" - hundreds to thousands of tiny paths,
# not the few dozen a real illustration's own strokes ever need, spanning
# most of the page's own width because that is simply where the text column
# runs. A full page scanned and embedded as one raster image is the same
# problem from the other side: neither is "a picture", both are just how
# that one page's content happens to be delivered on this side, and pairing
# either against the OTHER side's normal figure or normal text can only ever
# misfire, since there is nothing of matching shape to pair it with.
_VECTORIZED_TEXT_DENSITY = 0.006  # primitives per pt^2 - several times any real illustration's own detail
_PAGE_SCALE_WIDTH_FRAC = 0.5      # spans at least half the page's width
_FULL_PAGE_RASTER_FRAC = 0.75     # a raster image this large a share of the page is a full-page scan


def _is_page_scale_content(doc: fitz.Document, page_index: int, f) -> bool:
    rect = doc[page_index].rect
    if f.kind == "vector":
        area = max(1.0, f.display_width * f.display_height)
        density = f.primitives / area
        return (
            density >= _VECTORIZED_TEXT_DENSITY
            and rect.width > 0 and f.display_width / rect.width >= _PAGE_SCALE_WIDTH_FRAC
        )
    if f.kind == "raster":
        page_area = max(1.0, rect.width * rect.height)
        return (f.display_width * f.display_height) / page_area >= _FULL_PAGE_RASTER_FRAC
    return False


def _marker_immediately_before(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """A short list marker ("9.", "17.", "a)", a bullet glyph) is printed
    immediately to the LEFT of this icon, on the same line - so the icon is a
    numbered/bulleted item's own inline artwork, not a NOTE/TIP/WARNING icon
    living in the page margin. A genuine margin icon has nothing printed
    before it but margin whitespace; this one has the item's own marker,
    which just happened to end up extracted as a separate text element from
    its label (see `list_label_layout_changes`) - the split that otherwise
    makes this icon geometrically indistinguishable from a real margin icon.
    """
    x0, y0, x1, y1 = bbox
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return False
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            ly0, ly1 = line["bbox"][1], line["bbox"][3]
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap < 0.5 * min(y1 - y0, ly1 - ly0):
                continue
            last = spans[-1]
            lx1 = last["bbox"][2]
            if 0 <= x0 - lx1 <= _LEADING_ICON_GAP and _LIST_MARKER_TOKEN_RE.fullmatch(last["text"].strip()):
                return True
    return False


_INK_DPI = 48
_INK_WHITE = 240


def _ink_bbox(doc: fitz.Document, page_index: int, bbox: tuple) -> tuple:
    """The part of a figure's box that is actually drawn on.

    Staging embeds its line drawings with a blank margin round the artwork, so
    its image box is ~25% bigger than the picture in it. Measured and compared
    by that box, the same rear-panel diagram read as a different figure ("in
    Production only" beside "in Staging only") and every accessory picture as
    "resized 85pt -> 105pt". Trimmed to its ink, it is the same picture.
    """
    try:
        rect = fitz.Rect(bbox) & doc[page_index].rect
        if rect.is_empty:
            return bbox
        zoom = _INK_DPI / 72.0
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect,
                                         colorspace=fitz.csGRAY, alpha=False)
        w, h, data = pix.width, pix.height, pix.samples
        # Ink is whatever differs from the region's own background, read off
        # its border - Staging sets some drawings inside a grey panel, and
        # "not white" kept the whole panel as the picture.
        border = [data[x] for x in range(w)] + [data[(h - 1) * w + x] for x in range(w)]
        border += [data[y * w] for y in range(h)] + [data[y * w + w - 1] for y in range(h)]
        background = sorted(border)[len(border) // 2] if border else 255
        tolerance = max(12, 255 - _INK_WHITE)
        ink = lambda v: abs(v - background) > tolerance  # noqa: E731
        cols = [x for x in range(w) if any(ink(data[y * w + x]) for y in range(h))]
        rows = [y for y in range(h) if any(ink(data[y * w + x]) for x in range(w))]
        if not cols or not rows:
            return bbox
        x0 = rect.x0 + cols[0] / zoom
        y0 = rect.y0 + rows[0] / zoom
        x1 = rect.x0 + (cols[-1] + 1) / zoom
        y1 = rect.y0 + (rows[-1] + 1) / zoom
        if x1 - x0 < _FIGURE_MIN_SIDE / 2 or y1 - y0 < _FIGURE_MIN_SIDE / 2:
            return bbox
        return (x0, y0, x1, y1)
    except Exception:
        return bbox


def _figure_element(doc: fitz.Document, page_index: int, info, bbox: tuple | None = None) -> Element:
    raw = bbox or tuple(float(v) for v in info.bbox)
    bbox = _ink_bbox(doc, page_index, raw)
    return Element(
        kind=KIND_FIGURE,
        text="",
        key="",
        boxes=[(page_index, bbox)],
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fp=imagefp.fingerprint(doc, page_index, bbox),
        drawn=getattr(info, "kind", "") == "vector",
        raw_bbox=raw,
        raw_fp=imagefp.fingerprint(doc, page_index, raw) if raw != bbox else None,
    )


def _stitch_across_pages(elements: list[Element]) -> list[Element]:
    """Join a paragraph that a page break cut in half.

    Production ends page 12 mid-sentence and finishes it on page 13; Staging,
    paginating differently, prints the whole thing on one page. Without this
    the two sides hold two elements against one and the chapter reports three
    differences for text that is identical - which is exactly the
    "next page continuous" case this check must never report.
    """
    out: list[Element] = []
    for el in elements:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.kind == KIND_TEXT and el.kind == KIND_TEXT
            and el.page > prev.page
            and _ends_open(prev.text) and _starts_continuation(el.text)
        ):
            prev.wraps = prev.wraps + tuple(_wrap_pairs([prev.text, el.text])) + el.wraps
            prev.text = f"{prev.text} {el.text}".strip()
            prev.key = _normalise(prev.text)
            prev.boxes = prev.boxes + el.boxes
            # The continuation's own styling travels with its words: dropping
            # it left bold or underline on the second page never compared.
            prev.bold_words = prev.bold_words + el.bold_words
            prev.underline_words = prev.underline_words + el.underline_words
            prev.italic_words = prev.italic_words + el.italic_words
            continue
        out.append(el)
    return out


_HEADING_CAPTION_REACH = 16.0  # pt: a text this close under/beside a figure is that figure's own
                                # caption, whatever it says - never promoted into a second heading


def _classify(elements: list[Element], heading_titles: set[str], body_size: float) -> None:
    """Name each text element for what it is: a heading, a note, or prose."""
    figures = [e for e in elements if e.kind == KIND_FIGURE]
    for el in elements:
        if el.kind != KIND_TEXT:
            continue
        label = i18n.match_callout_label(el.text)
        if label:
            el.kind = KIND_NOTE
            el.label = label
            continue
        title = normalize_title(el.text)
        # An icon's own caption ("WEEE", "Battery") can repeat a section's
        # exact title word for word - the recycling symbol IS labelled "WEEE" -
        # without being a second copy of that heading. A heading is never a
        # figure's caption, so a title match sitting right under or beside one
        # is read as the caption it visibly is; promoted to a heading, its
        # real content (an icon losing its caption) would go uncompared, since
        # only true content elements are diffed.
        if title and title in heading_titles and len(el.text) <= 90 and not _near_a_figure(
            el, [f for f in figures if f.page == el.page], reach=_HEADING_CAPTION_REACH
        ):
            el.kind = KIND_HEADING
            continue
        # A short line set well above body size is a printed heading even when
        # nothing bookmarks it - a chapter's sub-headings routinely are not.
        if len(el.text) <= 90 and body_size and el.size >= body_size * 1.25:
            el.kind = KIND_HEADING


def collect_elements(
    doc: fitz.Document,
    pdf_path: str | None,
    start: tuple[int, float],
    end: tuple[int, float] | None,
    heading_titles: set[str],
    body_size: float,
    skip: list[tuple] | None = None,
    skipped: list[str] | None = None,
) -> tuple[list[Element], list[int]]:
    """Everything printed from `start` to `end`, in reading order, across as
    many pages as the chapter runs to.

    Returns `(elements, pages)`. `pages` is every page the chapter touches on
    this side - which is what makes "the pages will be different on both sides"
    a non-issue: each side is read over its own page range and compared by what
    it says, never by where it says it.
    """
    start_page, start_y = start
    end_page = end[0] if end else doc.page_count - 1
    end_page = min(end_page, doc.page_count - 1)
    elements: list[Element] = []
    pages: list[int] = []

    for page_index in range(max(0, start_page), end_page + 1):
        rect = doc[page_index].rect
        y0 = start_y - 2 if page_index == start_page else rect.y0
        y1 = end[1] if (end and page_index == end[0]) else rect.y1
        if y1 - y0 < 6:
            continue
        listing = _listing_page(doc, page_index)
        if listing:
            if skipped is not None:
                skipped.append(f"{listing} (p.{page_index + 1})")
            continue
        pages.append(page_index)

        tables = []
        if pdf_path:
            try:
                tables = [
                    t for t in get_tables(pdf_path, page_index, doc, allow_tall_cells=True)
                    if y0 - 2 <= (t["bbox"][1] + t["bbox"][3]) / 2 <= y1 + 2
                ]
            except Exception:
                tables = []
        tables = [_extend_header_only(doc, page_index, t, y1) for t in tables]
        tables = [_fill_unruled_cells(doc, page_index, t) for t in tables]
        try:
            figures = [
                f for f in get_figures(doc, page_index, print_ready=True)
                if f.display_width >= _figure_min_side(doc) and f.display_height >= _figure_min_side(doc)
                and y0 - 2 <= (f.bbox[1] + f.bbox[3]) / 2 <= y1 + 2
                and not _is_leading_icon(doc, page_index, f.bbox)
                and not _is_page_scale_content(doc, page_index, f)
            ]
        except Exception:
            figures = []

        table_regions = [tuple(float(v) for v in t["bbox"]) for t in tables]
        artwork = _raster_rects(doc, page_index)
        # Rows first: whether text belongs to a figure is a question about a
        # whole printed line, never about a fragment of one.
        all_lines = _page_lines(doc, page_index, y0, y1, body_size, heading_titles)
        page_rows = _merge_rows(list(all_lines))
        rows = _merge_rows([
            ln for ln in all_lines
            if not _inside(ln["bbox"], table_regions)  # the table element carries it
        ])
        # Whether a detected figure is really a picture - not a table's own
        # header fill or a spec table's rows and icons vector-clustered into
        # one shape (`_mostly_text`, `_is_fill_band`) - is decided here, before
        # any of ITS text is taken out of `rows` below. Deciding it after, from
        # a `lines` already built on the full candidate list, stripped a
        # rejected figure's printed words for the picture it turned out not to
        # be, and never gave them back as the paragraph they actually are - a
        # whole table (Burn-in Cleaner, Reset All and all) read as one such
        # cluster vanished from the comparison completely, on neither side of
        # it: not a figure (rightly rejected) and not text either (already
        # gone by the time that rejection happened).
        figures = [
            f for f in figures
            if not _mostly_text(tuple(float(v) for v in f.bbox), rows, artwork, table_regions, page_rows)
            and not _is_fill_band(f, page_rows)
        ]
        figure_regions = [tuple(float(v) for v in f.bbox) for f in figures]
        # A figure's detected box is often too big: inline icons cluster into
        # one "figure" with the screenshot beneath them and the sentence and
        # table rows between. So inside a figure box, text is only the figure's
        # when it is printed ON the artwork, or set smaller than body text the
        # way diagram and screenshot labels are. A body-sized label beside an
        # icon ("Pop-up messages", "Clear All") belongs to the page.
        lines = [
            row for row in rows
            if not (_inside(row["bbox"], figure_regions) and _on_artwork(row, artwork, body_size))
        ]

        page_elements = [
            el for el in _paragraphs(lines, heading_titles, doc)
            if not _is_fragment(el) and not _BARE_CALLOUT_RE.match(el.text)
        ]
        for t in tables:
            table_el = _table_element(t, page_index, doc)
            table_el.header_fill = _header_fill(doc, page_index, _header_bbox(t))
            page_elements.append(table_el)
        page_elements += [
            _figure_element(doc, page_index, f, _trim_figure(tuple(float(v) for v in f.bbox), lines))
            for f in figures
        ]
        if skip:
            kept = []
            for el in page_elements:
                title = _in_spans(el.page, el.bbox[1], skip)
                if title:
                    if skipped is not None and title not in skipped:
                        skipped.append(title)
                    continue
                kept.append(el)
            page_elements = kept
        page_elements.sort(key=lambda e: (round(e.bbox[1], 1), e.bbox[0]))
        elements.extend(page_elements)

    elements = _stitch_across_pages(elements)
    _classify(elements, heading_titles, body_size)
    for i, el in enumerate(elements):
        el.order = i
    return elements, pages


# --- pairing the two sides -------------------------------------------------


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _wrap_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """(last word of a line, first word of the next) at each break between
    `lines` joined into one text - where a space the joining put there is the
    wrap's, not the text's."""
    out: list[tuple[str, str]] = []
    for before, after in zip(lines, lines[1:]):
        tail, head = _TOKEN_RE.findall(before or ""), _TOKEN_RE.findall(after or "")
        if tail and head:
            out.append((tail[-1].casefold(), head[0].casefold()))
    return out


_ITALIC_FONT_RE = re.compile(r"italic|oblique|[-,]it$", re.IGNORECASE)


def _span_italic(span: dict) -> bool:
    """A span set in an italic or oblique face - by the font's flag, or by its
    name when the flag is not set (many embedded subsets leave it off)."""
    if not (span.get("text") or "").strip():
        return False
    return bool(span.get("flags", 0) & _ITALIC_FLAG) or bool(_ITALIC_FONT_RE.search(span.get("font") or ""))


def _tokens(text: str) -> "Counter[str]":
    return Counter(_TOKEN_RE.findall(text or ""))


def _containment(a: str, b: str) -> float:
    """How much of the SHORTER text is inside the longer one.

    The two documents do not always extract the same amount of a thing. A spec
    table whose value columns pdfplumber reads on one side and misses on the
    other comes out as "Screen size | Panel type" against "Screen size 43\" 55\"
    | Panel type ADS ADS": a plain similarity says those are different tables
    and reports one missing and one added, when they are plainly the same
    table. Containment says the first is inside the second, which is the truth.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    shared = sum((ta & tb).values())
    return shared / min(sum(ta.values()), sum(tb.values()))


def _ratio(a: str, b: str) -> float:
    """How alike two pieces of text are, by their wording."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# Containment is only evidence when the contained text is itself substantial.
# "Settings" is contained in any paragraph that mentions the settings menu, and
# pairing Staging's "Settings" heading with such a paragraph reported the whole
# paragraph as reworded.
_CONTAINMENT_MIN_TOKENS = 6
_CONTAINMENT_MIN_SHARE = 0.3    # the shorter text is at least this share of the longer
_TABLE_CONTAINMENT_MIN_TOKENS = 3


def _contained_score(a: Element, b: Element) -> float:
    ta, tb = len(_TOKEN_RE.findall(a.key)), len(_TOKEN_RE.findall(b.key))
    short, longer = min(ta, tb), max(ta, tb, 1)
    if a.kind == KIND_TABLE and b.kind == KIND_TABLE:
        # A table one document splits into several small grids is still the
        # same table, however small each piece is against the whole.
        if short < _TABLE_CONTAINMENT_MIN_TOKENS:
            return 0.0
    else:
        if short < _CONTAINMENT_MIN_TOKENS or short / longer < _CONTAINMENT_MIN_SHARE:
            return 0.0
        if (a.kind == KIND_HEADING) != (b.kind == KIND_HEADING):
            return 0.0  # a heading is never "inside" a paragraph
    return _containment(a.key, b.key) * _CONTAINMENT_WEIGHT


def _figure_similarity(a: Element, b: Element) -> float:
    """How alike two figures look: the better of comparing their inked areas and
    comparing their boxes as detected. Trimming removes a blank margin one
    document adds, but can also cut a diagram's dimension lines differently on
    each side - neither view alone is reliable, the better of the two is."""
    best = 0.0
    for fa in (a.fp, a.raw_fp):
        for fb in (b.fp, b.raw_fp):
            if fa is None or fb is None:
                continue
            try:
                best = max(best, imagefp.similarity(fa, fb))
            except Exception:
                continue
    return best


def _pairable(a: Element, b: Element) -> float:
    """How well two elements answer to each other, 0-1. Only elements of a
    comparable kind pair at all: a table never stands in for a paragraph."""
    # Content is compared topic by topic: an element never answers to one under
    # a different shared heading. A "Connection" cell in Settings once paired
    # with the link "USB connection" under Connections, two topics away.
    if a.section and b.section and a.section != b.section:
        return 0.0
    if a.kind == KIND_FIGURE or b.kind == KIND_FIGURE:
        return _figure_similarity(a, b) if a.kind == b.kind else 0.0
    if (a.kind == KIND_TABLE) != (b.kind == KIND_TABLE):
        return 0.0
    # A callout (Tip / Note / Warning) answers only to a callout: a Tip's bullet
    # once paired with a table row of similar wording. Only the same words set
    # as a callout on one side and plain on the other still pair.
    if (a.kind == KIND_NOTE) != (b.kind == KIND_NOTE) and _ratio(a.key, b.key) < 0.9:
        return 0.0
    return max(_ratio(a.key, b.key), _contained_score(a, b))


def _merged_table_rows(elements: list[Element]) -> tuple[list[tuple], list, list, list]:
    """Every element's rows - cells, grid, spans and row_boxes together, kept
    row-aligned with each other - with a later fragment's REPEATED header row
    dropped (the same row `_stitch_continued_tables` drops for the dedicated
    missing-header check) and a row itself split by the page break joined into
    the row it belongs to.

    A table split across page fragments is required to reprint its header on
    each continuation; keeping that repeat here would count it as a data row
    (or, when it extracts with an odd extra cell, a whole extra column) purely
    because of where the two documents happened to break the page. The four
    lists are built in the SAME pass so a row dropped or merged from `cells`
    is dropped or merged from `grid`/`spans`/`row_boxes` too - building them
    separately let the two fall out of step by one row, so `_cells_in` read
    the wrong row's span count for every row after the mismatch and reported a
    "cells merged/split" that was really just this misalignment.
    """
    cells_out: list[tuple] = []
    grid_out: list = []
    spans_out: list = []
    boxes_out: list = []
    header = elements[0].cells[0] if elements[0].cells else None
    for i, e in enumerate(elements):
        rows, grid, spans, boxes = list(e.cells), list(e.grid), list(e.spans), list(e.row_boxes)
        if i > 0 and header and rows and _headers_match(_row_text(header), _row_text(rows[0])):
            rows, grid, spans, boxes = rows[1:], grid[1:], spans[1:], boxes[1:]
        # A row itself split by the page break: the continuation page's first
        # row repeats no label (its leading cell is blank) and carries only the
        # rest of the previous row's description - joined into that row here,
        # or the tail read as text missing from the row it actually belongs to.
        if i > 0 and cells_out and rows and _is_row_continuation(rows[0]):
            cont = rows.pop(0)
            cells_out[-1] = tuple(
                f"{prev} {c}".strip() if (prev or "").strip() and (c or "").strip()
                else (prev or c)
                for prev, c in zip(cells_out[-1], cont)
            )
            if grid:
                grid.pop(0)
            if spans:
                spans.pop(0)
            if boxes:
                boxes.pop(0)
        cells_out.extend(rows)
        grid_out.extend(grid)
        spans_out.extend(spans)
        boxes_out.extend(boxes)
    return cells_out, grid_out, spans_out, boxes_out


def _is_row_continuation(row: tuple) -> bool:
    """A table row split by a page break: its leading (label) cell is blank
    but a later cell carries the rest of the previous row's text."""
    if not row or (row[0] or "").strip():
        return False
    return any((c or "").strip() for c in row[1:])


def merge_elements(elements: list[Element]) -> Element | None:
    """Several elements read as one - what the other document printed as a
    single paragraph, table or figure.

    The two documents break their text into blocks differently: a marker drawn
    beside its step here and inside it there, a paragraph one side sets as two.
    Comparing the MERGED text of a group against the merged text of its
    counterpart is what stops that being reported, and it is the same rule as
    the wrap and the page break - how the words are divided up is not what the
    documents say.
    """
    if not elements:
        return None
    if len(elements) == 1:
        return elements[0]
    text = " ".join(e.text for e in elements if e.text).strip()
    chars = sum(len(e.text) for e in elements) or 1
    kind = KIND_TEXT
    for candidate in (KIND_TABLE, KIND_FIGURE, KIND_NOTE, KIND_HEADING):
        if any(e.kind == candidate for e in elements):
            kind = candidate
            break
    # A table split across pages carries its header again on each
    # continuation page - required, not a defect (see `table_header_repeats`) -
    # so a repeat is dropped here the same way `_stitch_continued_tables`
    # drops it for that check. Counting it as a data row shifted every row
    # after it and could even read as an extra column from that one
    # mis-joined row, a "table shape changed" that was really just the header
    # printing again exactly where it is supposed to.
    if kind == KIND_TABLE:
        cell_rows, grid_rows, span_rows, box_rows = _merged_table_rows(elements)
        table_rows: list[tuple] | None = cell_rows
        grid, spans, row_boxes = tuple(grid_rows), tuple(span_rows), tuple(box_rows)
    else:
        table_rows = None
        grid = tuple(row for e in elements for row in e.grid)
        spans = tuple(v for e in elements for v in e.spans)
        row_boxes = tuple(rb for e in elements for rb in e.row_boxes)
    return Element(
        kind=kind,
        text=text,
        key=_normalise(text),
        boxes=[box for e in elements for box in e.boxes],
        order=elements[0].order,
        size=max(e.size for e in elements),
        bold=sum(e.bold * len(e.text) for e in elements) / chars,
        italic=sum(e.italic * len(e.text) for e in elements) / chars,
        color=elements[0].color,
        bold_words=tuple(w for e in elements for w in e.bold_words),
        underline_words=tuple(w for e in elements for w in e.underline_words),
        italic_words=tuple(w for e in elements for w in e.italic_words),
        wraps=tuple(w for e in elements for w in e.wraps)
        + tuple(_wrap_pairs([e.text for e in elements if e.text])),
        label=next((e.label for e in elements if e.label), ""),
        rows=len(table_rows) if table_rows is not None else sum(e.rows for e in elements),
        cols=max((e.cols for e in elements), default=0),
        cells=tuple(table_rows) if table_rows is not None else tuple(row for e in elements for row in e.cells),
        width=max((e.width for e in elements), default=0.0),
        height=max((e.height for e in elements), default=0.0),
        fp=next((e.fp for e in elements if e.fp is not None), None),
        header_fill=next((e.header_fill for e in elements if e.header_fill is not None), None),
        icon_column=any(e.icon_column for e in elements),
        headers=next((e.headers for e in elements if e.headers), ()),
        ocr=any(e.ocr for e in elements),
        grid=grid,
        spans=spans,
        row_boxes=row_boxes,
        section=elements[0].section,
    )


def _token_f1(a: str, b: str) -> float:
    """Multiset token overlap as an F1 - rises as a group's words approach its
    counterpart's, and falls as unrelated words are added, which is exactly the
    property needed to decide how many elements belong in a group."""
    ta, tb = _tokens(a), _tokens(b)
    total = sum(ta.values()) + sum(tb.values())
    return 2 * sum((ta & tb).values()) / total if total else 0.0


def _family(el: Element) -> str:
    """Which elements may stand in for each other: prose with prose (heading,
    paragraph, note), tables with tables, figures with figures."""
    if el.kind == KIND_FIGURE:
        return KIND_FIGURE
    if el.kind == KIND_TABLE:
        return KIND_TABLE
    return KIND_TEXT


def _group_score(left: list[Element], right: list[Element]) -> float:
    a, b = merge_elements(left), merge_elements(right)
    if a is None or b is None:
        return 0.0
    return _pairable(a, b)


_MAX_MERGE = 3  # how many of one side's elements may answer to one of the other's
_ABSORB_MIN = 0.85  # share of a further piece's words the other side's group must hold to take it in
_MEMBER_MIN = 0.5  # share of each group member's words the other side must carry


def _group_x_range(el: Element) -> tuple[float, float] | None:
    boxes = [b[1] for b in el.boxes]
    return (min(b[0] for b in boxes), max(b[2] for b in boxes)) if boxes else None


def _column_ok(group: list[Element]) -> bool:
    """A group of more than one TEXT element only reads as one merged
    paragraph when every member's x-range overlaps the rest's - a table row's
    label cell and its description cell can sit close enough in y to look like
    the next line of the same paragraph, and without this guard got spliced
    into one sentence with the label's words landing mid-sentence ("1. Open
    the Settings menu. Settings menu 2. Select."). Tables and figures are
    exempt: a whole table is one element already, never a passenger here.
    """
    if len(group) <= 1 or any(e.kind != KIND_TEXT for e in group):
        return True
    ranges = [r for r in (_group_x_range(e) for e in group) if r]
    if len(ranges) <= 1:
        return True
    lo, hi = max(r[0] for r in ranges), min(r[1] for r in ranges)
    return hi - lo >= -_PARAGRAPH_COLUMN_SLACK


_FUTURE_REACH = 30  # how far down the other run an element's counterpart is looked for


def _has_future(el: Element, other: list[Element], start: int) -> bool:
    """`el` answers to something further down `other`, from `start` on."""
    floor = _FIGURE_PAIR_MIN if el.kind == KIND_FIGURE else _PAIR_MIN_RATIO
    return any(_family(o) == _family(el) and _pairable(el, o) >= floor
               for o in other[start + 1:start + 1 + _FUTURE_REACH])


def _unused_share(piece: Element, whole: list[Element], taken: list[Element]) -> float:
    """Share of `piece`'s words that `whole` prints and `taken` has not
    already answered for - a sentence printed a second time is not part of
    the paragraph its first copy already matched."""
    want = Counter(_TOKEN_RE.findall(piece.key))
    if not want:
        return 0.0
    left_over = Counter(_TOKEN_RE.findall(" ".join(e.key for e in whole))) \
        - Counter(_TOKEN_RE.findall(" ".join(e.key for e in taken)))
    return sum((want & left_over).values()) / sum(want.values())


def _align_run(left: list[Element], right: list[Element]) -> list[tuple]:
    """Pair up two runs the ordered pass could not match, allowing one element
    on one side to answer to several on the other.

    Walks both runs together, and at each step takes the smallest combination
    (1:1, then 1:2 / 2:1, up to 3:3) that clears the pairing floor. When
    nothing does, the element that cannot be placed is emitted on its own -
    genuinely missing or genuinely added.
    """
    pairs: list[tuple] = []
    i = j = 0
    while i < len(left) and j < len(right):
        best, best_key = None, None
        for m in range(1, min(_MAX_MERGE, len(left) - i) + 1):
            for n in range(1, min(_MAX_MERGE, len(right) - j) + 1):
                left_group, right_group = left[i:i + m], right[j:j + n]
                # A figure has no words, so a paragraph merged into its group
                # costs the figure match nothing - which is how a line of
                # Production text ended up boxed as part of a figure finding.
                families = {_family(e) for e in left_group + right_group}
                # A borderless table whose header band pdfplumber traced but
                # whose body rows it read as ordinary paragraphs (no lines to
                # follow) is still the SAME table as the one detected whole on
                # the other side - so exactly one TABLE against a run of TEXT
                # is let through, everything else mixed stays blocked.
                table_vs_text = (
                    families == {KIND_TABLE, KIND_TEXT}
                    and sum(e.kind == KIND_TABLE for e in left_group + right_group) == 1
                )
                if (len(families) > 1 and not table_vs_text) or (KIND_FIGURE in families and (m > 1 or n > 1)):
                    continue
                if not (_column_ok(left_group) and _column_ok(right_group)):
                    continue
                score = _group_score(left_group, right_group)
                floor = _FIGURE_PAIR_MIN if left[i].kind == KIND_FIGURE else _PAIR_MIN_RATIO
                if score < floor:
                    continue
                a, b = merge_elements(left_group), merge_elements(right_group)
                # Every element in a group has to be part of the match. Scored
                # as a whole, a group can carry a passenger: an unrelated NOTE
                # in front of two headings still "contains" the headings the
                # other side prints, and was reported as a note reworded into
                # "User interface Home screen".
                if (m > 1 or n > 1) and a.kind != KIND_FIGURE and (
                    any(_containment(e.key, b.key) < _MEMBER_MIN for e in left_group)
                    or any(_containment(e.key, a.key) < _MEMBER_MIN for e in right_group)
                ):
                    continue
                # Ranked by how completely the two groups' words cover each
                # other, not by the pairing score: containment says a fragment
                # of a table is "in" the whole table, and would happily pair the
                # first fragment and leave the second one reported as missing.
                # Coverage grows as the second fragment joins the group. Ties go
                # to the smaller grouping.
                rank = score if a.kind == KIND_FIGURE else _token_f1(a.key, b.key)
                key = (round(rank, 3), -(m + n))
                if best_key is None or key > best_key:
                    best, best_key = (m, n), key
        if best is None:
            # Whichever side's next element has no future here at all moves on:
            # a run of diagram labels only Production prints must not push
            # Staging's next paragraph out as "added" when that paragraph's
            # own counterpart is only a few labels further down Production's
            # run - it was then re-paired as "moved", though in its place.
            future_left = _has_future(left[i], right, j)
            future_right = _has_future(right[j], left, i)
            if future_right and not future_left:
                pairs.append(([left[i]], []))
                i += 1
                continue
            if future_left and not future_right:
                pairs.append(([], [right[j]]))
                j += 1
                continue
            ahead_left = max(
                (_group_score([left[i]], right[j:j + n]) for n in range(1, min(_MAX_MERGE, len(right) - j) + 1)),
                default=0.0,
            )
            ahead_right = max(
                (_group_score(left[i:i + m], [right[j]]) for m in range(1, min(_MAX_MERGE, len(left) - i) + 1)),
                default=0.0,
            )
            if ahead_left >= ahead_right:
                pairs.append(([], [right[j]]))
                j += 1
            else:
                pairs.append(([left[i]], []))
                i += 1
            continue
        m, n = best
        # A paragraph one side prints as more pieces than `_MAX_MERGE` allows
        # (Production's one "You can set..." block against Staging's six lines
        # and bullets): keep taking the next piece while it is plainly part of
        # the other side's text, rather than leaving the tail "only in Staging".
        while j + n < len(right) and _family(right[j + n]) == _family(left[i]) and _column_ok(right[j:j + n + 1]) \
                and _unused_share(right[j + n], left[i:i + m], right[j:j + n]) >= _ABSORB_MIN:
            n += 1
        while i + m < len(left) and _family(left[i + m]) == _family(right[j]) and _column_ok(left[i:i + m + 1]) \
                and _unused_share(left[i + m], right[j:j + n], left[i:i + m]) >= _ABSORB_MIN:
            m += 1
        pairs.append((left[i:i + m], right[j:j + n]))
        i += m
        j += n
    pairs.extend(([e], []) for e in left[i:])
    pairs.extend(([], [e]) for e in right[j:])
    return pairs


def _section_at(anchors: list[tuple]) -> "callable":
    """page, y -> the topic (a matched heading's id) printed there on this side."""
    def at(page: int, y: float) -> str:
        section = ""
        for anchor_page, anchor_y, topic in anchors:
            if (anchor_page, anchor_y) <= (page, y + 1):
                section = topic
            else:
                break
        return section
    return at


def assign_sections(elements: list[Element], anchors: list[tuple]) -> None:
    """Tag each element with its topic: the nearest heading above it that both
    documents have, matched as a pair - whatever page either side prints it on.
    A heading only one side has is not a boundary; its content belongs to the
    shared heading above it (Staging bookmarks "Switching to blank screen"
    inside a topic Production keeps whole)."""
    at = _section_at(anchors)
    for el in elements:
        el.section = at(el.page, el.bbox[1])


def pair_elements(exp: list[Element], act: list[Element]) -> list[tuple]:
    """Align the two sides' elements, in reading order, by what they say.

    Returns a list of `(exp_elements, act_elements)` groups: usually one
    against one, sometimes several against one where the two documents divide
    the same content differently, and one side empty where content really is
    only in one of them.

    Order first - a chapter is read top to bottom, and its elements come in the
    same order in both documents far more often than not - then the runs that
    did not line up are aligned against each other by content, and anything
    still left over is offered one last chance to match somewhere else in the
    chapter, which is how content that MOVED is recognised as moved rather than
    reported as deleted and added.
    """
    matcher = difflib.SequenceMatcher(
        a=[e.key or f"\x00{e.kind}{i}" for i, e in enumerate(exp)],
        b=[e.key or f"\x00{e.kind}{i}" for i, e in enumerate(act)],
        autojunk=False,
    )
    pairs: list[tuple] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(([a], [b]) for a, b in zip(exp[i1:i2], act[j1:j2]))
        else:
            pairs.extend(_align_run(list(exp[i1:i2]), list(act[j1:j2])))

    # Last pass: an element still one-sided may simply have moved within the
    # chapter, which is a difference of its own but not a loss.
    lone_exp = [(k, p[0][0]) for k, p in enumerate(pairs) if p[0] and not p[1]]
    lone_act = [(k, p[1][0]) for k, p in enumerate(pairs) if p[1] and not p[0]]
    used: set[int] = set()
    for ki, a in lone_exp:
        best, best_score = None, 0.0
        for kj, b in lone_act:
            if kj in used or _short_label_mismatch(a, b):
                continue
            score = _pairable(a, b)
            floor = _FIGURE_PAIR_MIN if a.kind == KIND_FIGURE else _PAIR_MIN_RATIO
            if score >= floor and score > best_score:
                best, best_score = kj, score
        if best is not None:
            used.add(best)
            # Out of sequence only when content both sides print in place sits
            # between the two: with nothing matched in between, the element is
            # where Production has it, however the runs around it were divided.
            lo, hi = sorted((ki, best))
            crossed = any(p is not None and p[0] and p[1] and (len(p) < 3 or p[2] != "moved")
                          for p in pairs[lo + 1:hi])
            pairs[ki] = ([a], [pairs[best][1][0]], "moved" if crossed else "")
            pairs[best] = None
    pairs = [p for p in pairs if p is not None]
    return [(p[0], p[1], p[2] if len(p) > 2 else "") for p in pairs]


_SHORT_LABEL_MAX_WORDS = 4  # a phrase this short is a caption/label, not a sentence


def _short_label_mismatch(a: "Element", b: "Element") -> bool:
    """True when two short labels/captions only LOOK alike: SequenceMatcher's
    character ratio, run over so few characters, is dominated by whichever
    words they happen to share ("grommet ring" against "grommet mounting"
    scores 0.79 on their shared "grommet" prefix alone, well past the pairing
    floor, though the two say different things). This is what lets this last,
    whole-chapter search latch a diagram's one-word caption onto an unrelated
    label elsewhere in the chapter and call it "moved" - the position check
    every earlier, order-scoped pairing pass already had is gone by this
    point, so a short label needs its OWN content check: unless the two share
    their last content word too, they are not the same label.
    """
    if a.kind == KIND_FIGURE or a.kind != b.kind:
        return False
    wa, wb = _TOKEN_RE.findall(a.key), _TOKEN_RE.findall(b.key)
    if not wa or not wb or max(len(wa), len(wb)) > _SHORT_LABEL_MAX_WORDS:
        return False
    return wa[-1] != wb[-1]


# --- list numbering ---------------------------------------------------------
#
# List markers are taken out of every comparison key (see `_normalise`), so a
# marker drawn beside its item in one document and inside it in the other is
# not a wording difference. That same step would hide a procedure renumbered
# 1, 2, 3 -> a, b, c - which IS a difference, and a serious one: every "repeat
# step 3" around it stops resolving. So markers get their own comparison, over
# the whole chapter: each list item is identified by its own words, paired with
# the item that says the same thing on the other side, and the two markers'
# STYLES are compared.

_ITEM_MARKER_RE = re.compile(
    r"(?:(?<=\s)|^)((?:\d{1,2}|[a-z]|ii|iii|iv|vi|vii|viii|ix|xi|xii) ?[.)]"
    r"|\((?:\d{1,2}|[a-z]|ii|iii|iv|vi|vii|viii|ix|xi|xii)\)"
    r"|[\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219\u25cf\u25cb])(?=\s)",
    re.IGNORECASE,
)
_ITEM_MIN_KEY = 6          # an item's own words must identify it
_ITEM_KEY_CHARS = 80       # ... compared over its opening, not its whole length
_MARKER_VERB = {
    "number": "numbers", "letter": "letters",
    "roman": "numbers with roman numerals", "bullet": "bullets", "dash": "marks with dashes",
}
_FOOTNOTE_MARK_RE = re.compile(r"^\s*([*\u2020\u2021\u00a7]{1,2})(?=[\s:.]|$)")


def _footnote_mark(text: str) -> str:
    """The footnote-style mark ("*", "†", "‡", "§") leading a printed line, if
    any - a single symbol a word-level text diff drops as not a word at all."""
    m = _FOOTNOTE_MARK_RE.match(text or "")
    return m.group(1) if m else ""
_DASH_MARKERS = {"-", "–", "—"}


# --- inline icons ------------------------------------------------------------
_ICON_MAX_SIDE = 40.0       # an image no larger than this, inside a text block, is an inline icon
_ICON_MIN_SIDE = 5.0
# A compound icon button (two glyphs sharing one pill-shaped outline - a
# volume-down/volume-up pair, say) is wider than a single icon but still only
# one line tall, so it gets a wider berth on width alone rather than being
# excluded purely for holding two symbols instead of one.
_ICON_MAX_WIDE = 60.0
# `_cluster_rects` rebuilds its whole output list on every merge, so a run of
# N mostly-separate primitives can cost well past O(N^2) before it settles -
# fine for a handful of icon fragments, not for a page with hundreds/thousands
# of raw path segments (a detailed technical illustration, a densely hatched
# diagram), which was measured to make a single page's icon lookup hang for
# minutes. Two independent guards keep that from ever being reached: a
# primitive too big in either dimension to be part of a compact icon glyph
# (a table rule spanning most of the page, a large illustration's outline)
# is dropped before it ever reaches clustering, since it's both useless for
# icon detection and, being large, the single biggest driver of that
# merge cost; and even after that filter, a page still offering more
# icon-sized candidates than this is unusual enough (fine hatching, a dense
# schematic) that clustering it isn't worth the risk - skipped rather than
# gambled on.
_RAW_PRIMITIVE_MAX_SIDE = 80.0
_RAW_CLUSTER_MAX_PRIMITIVES = 400
_ICON_HUE_DELTA = 25.0      # degrees of dominant hue before a colour change is reported
_ICON_SAME = 0.6            # appearance match below which a paired icon is a different picture
_ICON_COLOURED_SHARE = 0.08 # share of an icon's pixels that must be coloured for it to have a hue
_ICON_CACHE: dict[tuple, list] = {}


def _overlaps_existing(bbox: tuple, existing: list[tuple], threshold: float = 0.5) -> bool:
    x0, y0, x1, y1 = bbox
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    for ex0, ey0, ex1, ey1 in existing:
        ix0, iy0 = max(x0, ex0), max(y0, ey0)
        ix1, iy1 = min(x1, ex1), min(y1, ey1)
        if ix1 > ix0 and iy1 > iy0 and ((ix1 - ix0) * (iy1 - iy0)) / area >= threshold:
            return True
    return False


def _page_icons(doc: fitz.Document, page_index: int) -> list[tuple]:
    key = (doc_key(doc), page_index)
    if key not in _ICON_CACHE:
        try:
            infos = doc[page_index].get_image_info(xrefs=True)
        except Exception:
            infos = []
        boxes = [tuple(float(v) for v in i["bbox"]) for i in infos]
        # Icons DRAWN as vector shapes count too: Production draws its safety
        # symbols where Staging embeds them as images, and seeing only the
        # images reported every one of them as "Icon added in Staging".
        try:
            boxes += [tuple(float(v) for v in f.bbox) for f in get_vector_figures(doc, page_index, print_ready=True)]
        except Exception:
            pass
        icon_boxes = [
            b for b in boxes
            if _ICON_MIN_SIDE <= b[2] - b[0] <= _ICON_MAX_SIDE
            and _ICON_MIN_SIDE <= b[3] - b[1] <= _ICON_MAX_SIDE
        ]
        # A compact multi-path glyph on a remote-control/button diagram can be
        # too PLAIN (a bare outline box - broken artwork with nothing left
        # inside it) or too small relative to a real illustration to pass
        # get_vector_figures' "is this a real figure" bar, on EITHER side,
        # regardless of which one is actually broken - so it never reaches the
        # icon pool at all and a broken icon goes uncompared entirely, silent.
        # Clustering the page's raw vector primitives directly, bypassing that
        # figure-worthiness filter, recovers it; the icon-size bound is what
        # keeps this from pulling in a table's own ruling lines or a full
        # diagram, and the overlap check keeps it from double-counting an icon
        # already found above.
        try:
            raw_rects = [
                r for d in doc[page_index].get_drawings() if d.get("rect")
                for r in [tuple(float(v) for v in d["rect"])]
                if r[2] - r[0] <= _RAW_PRIMITIVE_MAX_SIDE and r[3] - r[1] <= _RAW_PRIMITIVE_MAX_SIDE
            ]
            if 0 < len(raw_rects) <= _RAW_CLUSTER_MAX_PRIMITIVES:
                for bbox, _count in _cluster_rects(raw_rects, gap=2.0):
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if (_ICON_MIN_SIDE <= h <= _ICON_MAX_SIDE and _ICON_MIN_SIDE <= w <= _ICON_MAX_WIDE
                            and not _overlaps_existing(bbox, icon_boxes)):
                        icon_boxes.append(bbox)
        except Exception:
            pass
        _ICON_CACHE[key] = [(page_index, b) for b in icon_boxes]
    return _ICON_CACHE[key]


def _icon_anchor(doc: fitz.Document, page_index: int, bbox: tuple) -> tuple[str, str]:
    """The word printed just before the icon on its line ("after", word), or
    the word just after it when the icon starts the line ("before", word)."""
    cy = (bbox[1] + bbox[3]) / 2
    try:
        words = [w for w in doc[page_index].get_text("words") if w[1] - 2 <= cy <= w[3] + 2]
    except Exception:
        words = []
    before = [w for w in words if w[2] <= bbox[0] + 1 and bbox[0] - w[2] < 60]
    clip = lambda w: w if len(w) <= 20 else w[:20] + "…"  # noqa: E731  (a CJK run is one "word")
    if before:
        return "after", clip(max(before, key=lambda w: w[2])[4])
    after = [w for w in words if w[0] >= bbox[2] - 1 and w[0] - bbox[2] < 60]
    if after:
        return "before", clip(min(after, key=lambda w: w[0])[4])
    return "", ""


def _icon_colour(doc: fitz.Document, page_index: int, bbox: tuple) -> tuple[float | None, float]:
    """(dominant hue in degrees or None when the icon is grey, coloured share)."""
    import colorsys
    import math
    try:
        pix = doc[page_index].get_pixmap(clip=fitz.Rect(*bbox), dpi=200, alpha=False)
    except Exception:
        return None, 0.0
    n, samples = pix.n, pix.samples
    xs = ys = 0.0
    coloured = total = 0
    for k in range(0, len(samples) - n + 1, n):
        r, g, b = samples[k] / 255, samples[k + 1] / 255, samples[k + 2] / 255
        if r > 0.92 and g > 0.92 and b > 0.92:
            continue  # the paper behind the icon
        total += 1
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s > 0.25 and v > 0.15:
            coloured += 1
            xs += math.cos(2 * math.pi * h)
            ys += math.sin(2 * math.pi * h)
    share = coloured / total if total else 0.0
    if share < _ICON_COLOURED_SHARE:
        return None, share
    return (math.degrees(math.atan2(ys, xs)) % 360), share


def _icon_colour_diff(ia: tuple, ib: tuple, side: str, ca: str, cb: str) -> dict:
    (pa, ba), word = ia
    (pb, bb), _ = ib
    return {
        "type": "icon-colour", "kind": KIND_TEXT,
        "summary": (f"Icon colour changed — the icon {side} “{word}” is {ca} in Production "
                    f"and {cb} in Staging."),
        "detail": "",
        "exp": Element(kind=KIND_TEXT, text=word, key=_normalise(word), boxes=[(pa, ba)]),
        "act": Element(kind=KIND_TEXT, text=word, key=_normalise(word), boxes=[(pb, bb)]),
    }


def _icon_look_diff(doc_a: fitz.Document, ia: tuple, doc_b: fitz.Document, ib: tuple, side: str) -> dict | None:
    """The same icon beside the same word, drawn as a different picture: the
    user reports these (Staging's redrawn "More" and "Settings" icons score
    0.36-0.47 against Production's; an unchanged icon scores 0.9 and up)."""
    (pa, ba), word = ia
    (pb, bb), _ = ib
    try:
        look = imagefp.similarity(imagefp.fingerprint(doc_a, pa, ba), imagefp.fingerprint(doc_b, pb, bb))
    except Exception:
        return None
    if look >= _ICON_SAME:
        return None
    return {
        "type": "icon-changed", "kind": KIND_TEXT,
        "summary": f"Icon looks different in Staging — the icon {side} “{word}” is a different picture.",
        "detail": f"appearance match {look:.0%}",
        "exp": Element(kind=KIND_TEXT, text=word, key=_normalise(word), boxes=[(pa, ba)]),
        "act": Element(kind=KIND_TEXT, text=word, key=_normalise(word), boxes=[(pb, bb)]),
    }


def _hue_name(hue: float | None) -> str:
    if hue is None:
        return "grey"
    for limit, name in ((15, "red"), (45, "orange"), (70, "yellow"), (170, "green"), (200, "teal"),
                        (255, "blue"), (290, "purple"), (345, "pink"), (360, "red")):
        if hue < limit:
            return name
    return "red"


def _same_picture_nearby(doc_mine: fitz.Document, icon: tuple, doc_other: fitz.Document,
                         other_blocks: list[Element]) -> bool:
    """The other document prints the same picture beside the paired block, at
    any size: Production's 30pt drawn safety symbol is an icon there and a
    46pt image - a figure, too big to be an icon - in Staging. Not missing."""
    page_index, bbox = icon
    try:
        mine = imagefp.fingerprint(doc_mine, page_index, bbox)
    except Exception:
        return False
    for el in other_blocks:
        for p_other, (x0, y0, x1, y1) in el.boxes:
            try:
                candidates = [tuple(float(v) for v in i["bbox"]) for i in doc_other[p_other].get_image_info()]
                candidates += [tuple(float(v) for v in f.bbox)
                               for f in get_vector_figures(doc_other, p_other, print_ready=True)]
            except Exception:
                continue
            for c in candidates:
                if c[2] < x0 - _NEARBY_PICTURE or c[0] > x1 + _NEARBY_PICTURE or c[3] < y0 - _NEARBY_PICTURE or c[1] > y1 + _NEARBY_PICTURE:
                    continue
                try:
                    if imagefp.similarity(mine, imagefp.fingerprint(doc_other, p_other, c)) >= _ICON_SAME:
                        return True
                except Exception:
                    continue
    return False


_NEARBY_PICTURE = 60.0  # points around a paired block to look for the same picture
_ICON_GAP_REACH = 30.0  # pt: how far a word may sit from an icon it anchors, matching `collect()`'s own reach


def _unrendered_mark_nearby(doc: fitz.Document, group: list["Element"], word: str,
                            claimed: set | None = None) -> bool:
    """Whether the OTHER document draws ANY ink at all - no size, shape or
    primitive-count floor, unlike every other icon/figure detector in this
    file - beside the same word, in the block being compared.

    A compact menu-button icon (a box with thin internal bars) draws every
    one of those bars thinner than a real illustration ever is, and the
    normal vector-figure detector filters exactly that out everywhere else on
    purpose - without it, a table's own ruling reads as a figure. That trade
    is right for figures; it means an icon drawn this way is invisible to the
    normal detector, though it is genuinely printed. A "missing"/"added" claim
    must not be asserted over marks the detector simply could not classify -
    checked here with no floor at all, only in the small reach beside the one
    word already in question, so a stray unrelated mark elsewhere on the page
    cannot excuse an icon that really is missing.

    Ink already accounted for by a DIFFERENT, already-detected icon (`_page_icons`,
    anchored to its own separate word - e.g. a "Settings" gear icon a few words
    away from an unrelated "Select" this function is checking) does not count:
    that icon was already compared under its own anchor, and letting its ink
    excuse a wholly different, genuinely missing icon nearby was a real bug -
    a common step-list word like "Select" recurs once per step, so the reach
    around ANY of its occurrences can overlap a NEIGHBOURING step's own real
    icon on the same line, wrongly "confirming" an unrelated step's icon.
    """
    norm = _normalise(word)
    pages = sorted({p for e in group for p, _ in e.boxes})
    for p in pages:
        try:
            words = doc[p].get_text("words")
        except Exception:
            continue
        # Only ink a DIFFERENT icon of this comparison already answers for is
        # discounted. An icon this side prints but never offered up for
        # comparison - its nearest word is a full stop, so it is anchored to
        # nothing ("works the same with [icon].") - answers for nothing under
        # the old rule, and the other side's copy of that very icon read as
        # added. It is printed all the same, and it is what is being asked about.
        known_icons = ([fitz.Rect(b) for _, b in claimed] if claimed is not None
                       else [fitz.Rect(b) for _, b in _page_icons(doc, p)])
        for w in words:
            if _normalise(w[4]) != norm:
                continue
            x0, y0 = w[0] - _ICON_GAP_REACH, w[1] - 4
            x1, y1 = w[2] + _ICON_GAP_REACH, w[3] + 4
            reach = fitz.Rect(x0, y0, x1, y1)
            try:
                for d in doc[p].get_drawings():
                    r = d.get("rect")
                    if (r is not None and r.width > 0 and r.height > 0 and r.intersects(reach)
                            and not any(r.intersects(k) for k in known_icons)):
                        return True
                for info in doc[p].get_image_info():
                    b = fitz.Rect(info["bbox"])
                    if b.intersects(reach) and not any(b.intersects(k) for k in known_icons):
                        return True
            except Exception:
                continue
    return False


_ARTWORK_OVER_ICON = 4.0  # a region this many times the icon's area is a picture holding it


def _inside_artwork(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """Is this small shape a detail drawn inside a larger picture?"""
    r = fitz.Rect(bbox)
    area = max(1.0, _rect_area(r))
    try:
        boxes = _page_artwork_boxes(doc, page_index)
    except Exception:
        return False
    return any(_rect_area(art) >= _ARTWORK_OVER_ICON * area and art.contains(r)
               for art in boxes)


def icon_changes(exp_group: list[Element], act_group: list[Element],
                 expected: fitz.Document, actual: fitz.Document,
                 exp_topic: list[Element] | None = None, act_topic: list[Element] | None = None) -> list[dict]:
    """Inline icons inside a paired block - the ⚙ in "System Settings [icon]
    menu", the icons in a table cell - compared by the word they sit beside:
    an icon only one side prints, or the same icon in a clearly different
    colour. The order of two icons beside the same word is not a change."""
    # A callout's own icon (Note / Warning / Important / TIP / Caution) is
    # style, not content: the user only wants it flagged when it is missing
    # outright, never when it is merely drawn differently or a different
    # colour on the two sides. In practice a callout icon is often a small,
    # sparse vector glyph that does not reliably extract as a detectable
    # image/figure at all on one rendering style even when it is genuinely
    # printed there (confirmed: Staging's TIP lightbulb, visibly identical to
    # Production's, produced no image/vector-figure candidate whatsoever) -
    # so a missing/added check built on that extraction cannot be trusted and
    # would report a printed icon as absent. Excluded from comparison
    # entirely, both for style/colour and for missing/added.
    def collect(doc: fitz.Document, group: list[Element]) -> dict[tuple, list]:
        out: dict[tuple, list] = {}
        seen: set = set()
        for el in group:
            if el.kind == KIND_FIGURE:
                continue
            for page_index, (x0, y0, x1, y1) in el.boxes:
                for icon in _page_icons(doc, page_index):
                    _, (ix0, iy0, ix1, iy1) = icon
                    cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
                    if icon in seen or not (x0 - 30 <= cx <= x1 + 30 and y0 - 4 <= cy <= y1 + 4):
                        continue
                    seen.add(icon)
                    # A detail DRAWN INSIDE an illustration - the sticker and
                    # the rubber in the monitor diagram - is part of that
                    # picture, not an icon beside the words. It can still fall
                    # within reach of a text block set alongside the picture,
                    # where it read as an icon "beside" whatever word was
                    # nearest and was reported missing although the other side
                    # prints the same illustration. The picture itself is
                    # compared as a figure.
                    if _inside_artwork(doc, page_index, (ix0, iy0, ix1, iy1)):
                        continue
                    # A callout's own icon sits in the margin to the LEFT of
                    # its whole text block, beside the top of it - an inline
                    # icon ("System Settings [icon] menu") sits INSIDE a line
                    # of running text, never outside its column. Checked on
                    # the icon's own position, not by anchoring it to a word:
                    # a wrapped callout's icon often sits beside a later
                    # wrapped line, not the first word.
                    #
                    # The "near the top of its own block" half of that check
                    # only holds when the block IS one callout's own element -
                    # two separate callouts that print close enough together
                    # to merge into one Element (Production's "For BenQ IFP…"
                    # note immediately above its own "Do not keep the device
                    # powered…" tip) puts the SECOND callout's icon well past
                    # that merged element's own top, so it read as a genuine
                    # content icon needing a match - one Staging correctly
                    # excludes for its OWN, separately-kept element, so
                    # nothing on that side is ever offered to pair against
                    # it. `_is_leading_icon` (independent of how the two
                    # paragraphs happened to merge - it asks only whether the
                    # icon sits right before the actual PRINTED LINE next to
                    # it) is accepted as an ALTERNATIVE to the top-of-block
                    # test, but never instead of the margin one: an inline
                    # icon ("Settings [icon] menu") can also happen to sit
                    # right before a line's own start when it wraps to open
                    # one, and only living in the margin - outside the text
                    # column altogether - tells a callout's icon apart from it.
                    # Neither half of that check actually rules out an inline
                    # icon whose own list marker ("9.", "17.") was extracted as
                    # a SEPARATE element from its label - a common split (see
                    # `list_label_layout_changes`) that leaves the icon sitting
                    # just left of THIS element's own start, indistinguishable
                    # by either signal above from a genuine margin callout: a
                    # short element is always "near its own top", and a
                    # genuine callout icon likewise always has text starting
                    # right after it. What tells them apart is what is printed
                    # immediately to the icon's OWN left: a callout icon has
                    # nothing there but page margin, an inline button icon has
                    # the item's own marker.
                    if (ix1 <= x0 + 2 and (
                        cy <= y0 + max(20.0, (y1 - y0) * 0.35)
                        or _is_leading_icon(doc, page_index, (ix0, iy0, ix1, iy1))
                    ) and not _marker_immediately_before(doc, page_index, (ix0, iy0, ix1, iy1))):
                        continue
                    side, word = _icon_anchor(doc, page_index, icon[1])
                    # A callout's own icon (Note / Warning / TIP) is callout
                    # styling - Staging's design - and a bullet is no anchor.
                    # The same icon leading a WHOLE paragraph, with no separate
                    # "TIP:" label at all, plays the identical role - Production's
                    # own convention for the same callout, just without the word -
                    # so it is held to the same rule: not a content icon tied to
                    # the one word it happens to sit beside.
                    first_word = (el.text.split() or [""])[0] if el.text else ""
                    leading_block = bool(
                        side == "before" and word and first_word
                        and _normalise(word) == _normalise(first_word).rstrip(".,;:")
                        and cy <= y0 + max(20.0, (y1 - y0) * 0.35)
                    )
                    if not word or not _TOKEN_RE.search(word) or _BARE_CALLOUT_RE.match(word) or leading_block:
                        continue
                    out.setdefault((side, _normalise(word)), []).append((icon, word))
        return out

    icons_a, icons_b = collect(expected, exp_group), collect(actual, act_group)
    if not icons_a and not icons_b:
        return []

    def colour(doc: fitz.Document, entry: tuple) -> str:
        return _hue_name(_icon_colour(doc, *entry[0])[0])

    out: list[dict] = []
    leftover_a: list[tuple] = []
    leftover_b: list[tuple] = []
    for key in list(dict.fromkeys(list(icons_a) + list(icons_b))):
        la, lb = icons_a.get(key, []), icons_b.get(key, [])
        leftover_a += la[len(lb):]
        leftover_b += lb[len(la):]
        pairs = list(zip(la, lb))
        # Colours compared as a set beside the same word: two icons printed
        # in the other order (a red and a blue status light) are no change.
        names_a = Counter(colour(expected, ia) for ia, _ in pairs)
        names_b = Counter(colour(actual, ib) for _, ib in pairs)
        recoloured: set[int] = set()
        if names_a != names_b:
            for n_pair, (ia, ib) in enumerate(pairs):
                ca, cb = colour(expected, ia), colour(actual, ib)
                if ca != cb and names_a[ca] > names_b[ca]:
                    out.append(_icon_colour_diff(ia, ib, key[0], ca, cb))
                    recoloured.add(n_pair)
        for n_pair, (ia, ib) in enumerate(pairs):
            if n_pair not in recoloured:
                looked = _icon_look_diff(expected, ia, actual, ib, key[0])
                if looked:
                    out.append(looked)
    # An icon anchored to the word before it on one side and the word after
    # it on the other ("More [icon] on") is the same icon: pair the leftovers
    # of the block in order before calling any of them missing or added.
    for ia, ib in zip(leftover_a, leftover_b):
        ca, cb = colour(expected, ia), colour(actual, ib)
        if ca != cb:
            out.append(_icon_colour_diff(ia, ib, "beside", ca, cb))
        else:
            looked = _icon_look_diff(expected, ia, actual, ib, "beside")
            if looked:
                out.append(looked)
    n = min(len(leftover_a), len(leftover_b))
    # The other side's whole topic: the two documents divide a table row or a
    # step into blocks differently, and an icon beside the same word in a
    # neighbouring block of the topic is the same icon, not a missing one.
    # Counted, not just looked up: "button" has an icon twice in Production's
    # topic and once in Staging's, so one of them really is missing.
    if leftover_a[n:] or leftover_b[n:]:
        by_word_a, by_word_b = Counter(), Counter()
        for (_, word), v in collect(expected, exp_topic or []).items():
            by_word_a[word] += len(v)
        for (_, word), v in collect(actual, act_topic or []).items():
            by_word_b[word] += len(v)
    else:
        by_word_a = by_word_b = Counter()
    for extra_side in (("icon-missing", leftover_a[n:]), ("icon-added", leftover_b[n:])):
        kind, entries = extra_side
        mine, theirs = (by_word_a, by_word_b) if kind == "icon-missing" else (by_word_b, by_word_a)
        entries = [e for e in entries if mine[_normalise(e[1])] > theirs[_normalise(e[1])]]
        doc_mine, doc_other, other_blocks = ((expected, actual, act_group) if kind == "icon-missing"
                                             else (actual, expected, exp_group))
        # Icons the other side DID offer for comparison, so their ink cannot
        # excuse this one (see `_unrendered_mark_nearby`).
        claimed_other = {ic for v in (icons_b if kind == "icon-missing" else icons_a).values()
                         for ic, _ in v}
        entries = [e for e in entries if not _same_picture_nearby(doc_mine, e[0], doc_other, other_blocks)
                   and not _unrendered_mark_nearby(doc_other, other_blocks, e[1], claimed_other)]
        for (icon, word) in entries:
            where = f"beside “{word}”"
            verb = "missing in Staging" if kind == "icon-missing" else "added in Staging"
            page_index, bbox = icon
            has = Element(kind=KIND_TEXT, text=word, key=_normalise(word), boxes=[(page_index, bbox)])
            other_group = act_group if kind == "icon-missing" else exp_group
            spot = merge_elements([e for e in other_group if e.kind != KIND_FIGURE])
            out.append({
                "type": kind, "kind": KIND_TEXT,
                "summary": (f"Icon {verb} — the icon {where} is printed in "
                            f"{'Production' if kind == 'icon-missing' else 'Staging'} only."),
                "detail": "",
                "exp": has if kind == "icon-missing" else spot,
                "act": spot if kind == "icon-missing" else has,
            })
    return out


def _marker_phrase(marker: str) -> str:
    """“step 5”, “item b”, “a bulleted item” - a marker as a reader names it."""
    kind = _marker_kind(marker)
    label = marker.strip("()").rstrip(".)").strip()  # "(a)", "a)" and "a ." all read as just "a"
    if kind == "number":
        return f"step {label}"
    if kind in ("letter", "roman"):
        return f"item {label}"
    return "a dashed item" if kind == "dash" else "a bulleted item"


def marker_changes(exp_group: list[Element], act_group: list[Element],
                   expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """List markers read off the PAGE for a paired stretch of text.

    The text checks only see a marker typed into the text. Staging draws the
    bullets before "(a) Display" and "(b) Wall mount bracket" as marks of their
    own, Production prints the same lines with none, and neither side's
    extracted text had a list item to compare - so the added bullets went
    unreported. Only lines whose markers the text checks cannot see are asked."""
    texts_a = [e for e in exp_group if e.kind == KIND_TEXT and e.key]
    texts_b = [e for e in act_group if e.kind == KIND_TEXT and e.key]
    out: list[dict] = []
    # A footnote-style mark ( * † ‡ § ) at the very start of a line is a single
    # symbol, not a word - the word-level text checks tokenize on \w+ and drop
    # it as noise, so "*: Charging..." against "Charging..." reads as identical
    # text on both sides. Checked here, on the RAW printed line, before any
    # list-item detection can short-circuit the rest of this function.
    for b in texts_b:
        if len(b.key) < _LIST_ITEM_MIN_KEY:
            continue
        phrase = " ".join(b.key.split()[:4])
        a = next((e for e in texts_a if phrase and phrase in e.key), None)
        if a is None:
            continue
        mark_a, mark_b = _footnote_mark(a.text), _footnote_mark(b.text)
        if mark_a == mark_b:
            continue
        if mark_a and not mark_b:
            out.append({
                "type": "text", "kind": KIND_TEXT,
                "summary": (f"Text missing in Staging — “{mark_a}” marks “{b.key[:60]}” "
                            f"in Production; Staging prints the line with no mark."),
                "detail": "", "exp": a, "act": b,
            })
        else:
            out.append({
                "type": "text", "kind": KIND_TEXT,
                "summary": (f"Text added in Staging — “{mark_b}” marks “{b.key[:60]}” "
                            f"in Staging; Production prints the line with no mark."),
                "detail": "", "exp": a, "act": b,
            })
    if not texts_a or not texts_b or _list_items_in(texts_a) or _list_items_in(texts_b):
        return out
    added: list[tuple] = []
    removed: list[tuple] = []
    restyled: list[tuple] = []
    for b in texts_b:
        if len(b.key) < _LIST_ITEM_MIN_KEY:
            continue  # too short to pair by containment alone - "C" matches almost anything
        phrase = " ".join(b.key.split()[:4])
        a = next((e for e in texts_a if phrase and phrase in e.key), None)
        if a is None:
            continue
        stage_marker = _marker_before(actual, b.pages, b.key, near=(b.page, b.bbox[1]))
        prod_marker = _marker_before(expected, a.pages, b.key, near=(a.page, a.bbox[1]))
        if stage_marker is None or prod_marker is None or stage_marker == prod_marker:
            continue
        item = (prod_marker, stage_marker, b.key, a, b)
        if stage_marker and not prod_marker:
            # The same callout restyling, the other way round: Production sets
            # the note as plain lines beside its icon, Staging bullets them
            # inside its "WARNING:" panel.
            if _callout_restyled(a, b, expected, actual):
                continue
            added.append(item)
        elif prod_marker and not stage_marker:
            # Unless it is the two documents' own way of setting a note:
            # Production bullets the lines of a note it marks with a small
            # icon, Staging prints the same words in a labelled "NOTE:" /
            # "TIP:" panel, where a bullet would be redundant. The same
            # layout is never expected of the other document.
            if _callout_restyled(a, b, expected, actual):
                continue
            removed.append(item)
        elif _marker_kind(prod_marker) != _marker_kind(stage_marker):
            restyled.append(item)
    for kind, items in (("list-marker-added", added), ("list-marker-missing", removed), ("numbering", restyled)):
        if not items:
            continue
        count = len(items)
        noun = "item" if count == 1 else "items"
        first = items[0][2][:70]
        others = f" (and {count - 1} more)" if count > 1 else ""
        if kind == "list-marker-added":
            summary = (f"List marker added in Staging — “{first}”{others} is {_marker_phrase(items[0][1])} "
                       f"in Staging but has no marker in Production.")
        elif kind == "list-marker-missing":
            summary = (f"List marker missing in Staging — “{first}”{others} is {_marker_phrase(items[0][0])} "
                       f"in Production but has no marker in Staging.")
        else:
            summary = (f"List numbering changed — “{first}” is {_marker_phrase(items[0][0])} in Production "
                       f"but {_marker_phrase(items[0][1])} in Staging"
                       f"{f' (and {count - 1} more like it)' if count > 1 else ''}.")
        out.append({
            "type": kind, "kind": KIND_TEXT, "summary": summary,
            "detail": (f"Production: {', '.join(i[0] or 'none' for i in items)} · "
                       f"Staging: {', '.join(i[1] or 'none' for i in items)}"),
            "exp": merge_elements([i[3] for i in items]), "act": merge_elements([i[4] for i in items]),
            "items": list(items),
        })
    return out


# --- shading behind text ------------------------------------------------------
_SHADE_WHITE = 0.96        # a fill lighter than this in every channel is the page, not a box
_SHADE_COVER = 0.85        # the box must lie under this much of the element
_FILL_CACHE: dict[tuple, list] = {}


def _page_fills(doc: fitz.Document, page_index: int) -> list["fitz.Rect"]:
    key = (doc_key(doc), page_index)
    if key not in _FILL_CACHE:
        page = doc[page_index]
        fills = []
        try:
            for d in page.get_drawings():
                fill = d.get("fill")
                r = fitz.Rect(d["rect"])
                if (fill and min(fill) < _SHADE_WHITE and r.width >= 30 and r.height >= 12
                        and _rect_area(r) < 0.6 * _rect_area(page.rect)):
                    fills.append(r)
        except Exception:
            pass
        _FILL_CACHE[key] = fills
    return _FILL_CACHE[key]


def _callout_restyled(own: Element, other: Element | None,
                      own_doc: fitz.Document | None, other_doc: fitz.Document | None) -> bool:
    """Is this "lost" list marker just the two documents' own way of setting a
    note? Production marks one with a small icon beside the line and bullets
    what follows; Staging redesigns it as a shaded box headed "NOTE:" or
    "TIP:", where a bullet would be redundant. Either side's callout styling
    answers for the other's - the layouts are not expected to match."""
    if other is None or own_doc is None or other_doc is None:
        return False
    if other.kind == KIND_NOTE or other.label:
        return True  # the other side heads it "NOTE:" / "TIP:" instead
    if _on_shading(other_doc, other) and not _on_shading(own_doc, own):
        return True  # set apart as a panel there, as a bulleted line here
    return _has_note_icon(own_doc, own) or _has_note_icon(other_doc, other)


def _on_shading(doc: fitz.Document, el: Element) -> bool:
    if not el.boxes:
        return False
    page_index, bbox = el.boxes[0]
    r = fitz.Rect(bbox)
    area = max(1.0, _rect_area(r))
    return any(_rect_area(f & r) >= _SHADE_COVER * area for f in _page_fills(doc, page_index))


_LOCAL_LEADING_MIN_SAMPLES = 3   # nearby paragraphs needed to know what "normal" leading looks like here
_LEADING_STEP = 0.25             # relative change from that normal, on the SAME side, that stands out


def _local_leading(doc: fitz.Document, elements: list[Element], section: str, skip: Element) -> float | None:
    """The typical line-spacing of body paragraphs near `skip`, on ONE side -
    what a callout or table cell in this same spot is set AGAINST, not an
    absolute figure. A document reset in a whole new template runs looser or
    tighter everywhere at once, and comparing raw ratios across documents
    would flag every callout in it; comparing each side's callout to its OWN
    neighbours cancels that out and leaves only a callout the template did
    NOT carry along with the rest of its own page."""
    ratios = [
        r for e in elements
        if e is not skip and e.kind == KIND_TEXT and e.section == section
        for r in [_line_spacing(doc, e)] if r is not None
    ]
    if len(ratios) < _LOCAL_LEADING_MIN_SAMPLES:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def line_spacing_changes(exp_group: list[Element], act_group: list[Element],
                         expected: fitz.Document, actual: fitz.Document,
                         exp_texts: list[Element], act_texts: list[Element]) -> list[dict]:
    """A callout or table cell set looser or tighter than the body text
    around it, on one side only - not a document reset in a new template
    (see `_local_leading`), and never a plain paragraph, which already types
    at whatever the template around it uses.
    """
    blocks_a = [e for e in exp_group if e.kind in (KIND_NOTE, KIND_TABLE)]
    blocks_b = [e for e in act_group if e.kind in (KIND_NOTE, KIND_TABLE)]
    if not blocks_a or not blocks_b:
        return []
    a, b = blocks_a[0], blocks_b[0]
    own_a, own_b = _line_spacing(expected, a), _line_spacing(actual, b)
    if own_a is None or own_b is None:
        return []
    base_a = _local_leading(expected, exp_texts, a.section, a)
    base_b = _local_leading(actual, act_texts, b.section, b)
    if base_a is None or base_b is None:
        return []
    rel_a, rel_b = own_a / base_a, own_b / base_b
    if abs(rel_b - rel_a) / rel_a < _LEADING_STEP:
        return []
    label = "table" if a.kind == KIND_TABLE else "callout"
    return [{
        "type": "line-spacing", "kind": a.kind,
        "summary": (f"Line spacing changed — Staging sets this {label}'s lines "
                    f"{'looser' if rel_b > rel_a else 'tighter'} than the text around it; "
                    f"Production sets it the same as its own surrounding text."),
        "detail": f"Relative to nearby text: {rel_a:.2f}x in Production, {rel_b:.2f}x in Staging",
    }]


def _has_note_icon(doc: fitz.Document, el: Element) -> bool:
    """A small icon (Production's usual pencil icon marking a NOTE it never
    spells out in words - see `_is_leading_icon`) sits immediately before this
    element's own printed line."""
    if not el.boxes:
        return False
    page_index, bbox = el.boxes[0]
    x0, y0, x1, y1 = bbox
    for _, (ix0, iy0, ix1, iy1) in _page_icons(doc, page_index):
        overlap = min(y1, iy1) - max(y0, iy0)
        if overlap >= 0.5 * min(y1 - y0, iy1 - iy0) and 0 <= x0 - ix1 <= _LEADING_ICON_GAP:
            return True
    return False


_NOTE_RUN_GAP = 8.0     # pt between two lines of one note's own run
_NOTE_RUN_INDENT = 14.0  # pt their left edges may differ by


def _in_icon_note(doc: fitz.Document, topic_elements: list[Element] | None, el: Element) -> bool:
    """`el` belongs to a note Production marks with an icon: the icon sits
    beside it, or beside an earlier line of the same unbroken run of lines -
    a note of three bullets draws its icon beside the first one only, and the
    third bullet is no less part of that note.
    """
    if _has_note_icon(doc, el):
        return True
    if not topic_elements:
        return False
    order = [e for e in topic_elements if e.boxes]
    at = next((i for i, e in enumerate(order) if e is el), None)
    if at is None:
        return False
    current = el
    for prev in reversed(order[:at]):
        page_prev, bbox_prev = prev.boxes[-1]
        page_cur, bbox_cur = current.boxes[0]
        # Two lines set tight can overlap their glyph boxes a little.
        slack = 0.6 * max(1.0, bbox_cur[3] - bbox_cur[1])
        if page_prev != page_cur or not (-slack <= bbox_cur[1] - bbox_prev[3] <= _NOTE_RUN_GAP) \
                or abs(bbox_cur[0] - bbox_prev[0]) > _NOTE_RUN_INDENT:
            return False
        if _has_note_icon(doc, prev):
            return True
        current = prev
    return False


def _previous_element(topic_elements: list[Element], el: Element) -> Element | None:
    """Whatever comes right before `el` in this topic's own reading order."""
    for i, e in enumerate(topic_elements):
        if e is el:
            return topic_elements[i - 1] if i > 0 else None
    return None


def _same_shaded_region(doc: fitz.Document, el_a: Element, el_b: Element) -> bool:
    """Whether these two elements sit inside the SAME shaded panel, not just
    each somewhere-shaded on their own - a note's own box is one thing;
    another paragraph shaded elsewhere on the same page is not evidence of
    anything shared with it."""
    if not el_a.boxes or not el_b.boxes:
        return False
    page_a, bbox_a = el_a.boxes[0]
    page_b, bbox_b = el_b.boxes[0]
    if page_a != page_b:
        return False
    rect_a, rect_b = fitz.Rect(bbox_a), fitz.Rect(bbox_b)
    area_a, area_b = max(1.0, _rect_area(rect_a)), max(1.0, _rect_area(rect_b))
    return any(
        _rect_area(fill & rect_a) >= _SHADE_COVER * area_a and _rect_area(fill & rect_b) >= _SHADE_COVER * area_b
        for fill in _page_fills(doc, page_a)
    )


def shading_changes(exp_group: list[Element], act_group: list[Element],
                    expected: fitz.Document, actual: fitz.Document,
                    exp_topic: list[Element] | None = None, act_topic: list[Element] | None = None) -> list[dict]:
    """The same text printed on a shaded box in one document only - Staging sets
    every "Important" note on a green panel Production does not have."""
    texts_a = [e for e in exp_group if e.kind != KIND_FIGURE and e.kind != KIND_TABLE]
    texts_b = [e for e in act_group if e.kind != KIND_FIGURE and e.kind != KIND_TABLE]
    if not texts_a or not texts_b:
        return []
    shaded_a = any(_on_shading(expected, e) for e in texts_a)
    shaded_b = any(_on_shading(actual, e) for e in texts_b)
    if shaded_a and not shaded_b:
        return [{
            "type": "shading", "kind": texts_b[0].kind,
            "summary": ("Background shading missing in Staging — Production prints the text on a shaded box, "
                        "Staging on the plain page."),
            "detail": "",
        }]
    # Staging's shaded panels are its design: shading it ADDS where Production
    # has none of its own is expected and never reported. But Production
    # marking this same spot as a note the WORDLESS way - a small pencil icon
    # beside the line, no "NOTE:" label and no shading of its own - still
    # needs Staging to set it apart SOMEHOW, the way Staging's redesign
    # already does for every note that does carry a printed label. Losing the
    # icon with nothing (no shading, no label) taking its place drops the
    # reader's only sign this line is a note and not just another sentence.
    if shaded_a and shaded_b:
        return []
    if shaded_b and not shaded_a:
        if any(_has_note_icon(expected, e) for e in texts_a):
            return []  # Production's icon-only note, Staging upgrades it to a full panel - expected.
        # Production treats this paragraph as ordinary text of its own -
        # no icon, no shading. Staging shading it anyway is only ever this
        # document's own design UNLESS it is really the tail of the PREVIOUS
        # paragraph's note box, grown to also cover a paragraph Production
        # keeps separate and plain: the box gets its content from Production's
        # actual note, then swallows the next, unrelated sentence too, so a
        # reader sees ordinary instructions dressed up as part of the note.
        if exp_topic is not None and act_topic is not None:
            prev_exp = _previous_element(exp_topic, texts_a[0])
            prev_act = _previous_element(act_topic, texts_b[0])
            if (prev_exp is not None and prev_act is not None
                    and _has_note_icon(expected, prev_exp)
                    and not _in_icon_note(expected, exp_topic, texts_a[0])
                    and _same_shaded_region(actual, prev_act, texts_b[0])):
                return [{
                    "type": "shading", "kind": texts_b[0].kind,
                    "summary": ("Extra content pulled into Staging's note box — Production keeps this text "
                                "as its own separate paragraph right after the note, but Staging's shaded "
                                "note box has grown to cover it too."),
                    "detail": "",
                    # Not a styling nuance like the other "shading" findings:
                    # ordinary instructions are being presented to the reader as
                    # part of a NOTE, which changes what the document tells them
                    # to do. Flagged critical so it is drawn and ranked like
                    # content lost outright, not filed under "how text is set".
                    "critical": True,
                }]
        return []
    if any(_has_note_icon(expected, e) for e in texts_a):
        return [{
            "type": "shading", "kind": texts_b[0].kind,
            "summary": ("Note styling missing in Staging — Production marks this as a note with a small "
                        "icon beside the text, Staging prints it as plain text with no shading or other "
                        "note styling to replace it."),
            "detail": "",
            # A note a reader can no longer tell from ordinary text is a real
            # loss, not a styling nuance: drawn red, like missing content.
            "critical": True,
        }]
    return []


def _marker_kind(marker: str) -> str:
    """number / letter / roman / bullet / dash. A list printed with "-" where
    Production prints "•" is marked differently, though both are "bullets"."""
    marker = (marker or "").strip()
    return "dash" if marker in _DASH_MARKERS else classify_marker(marker)


def _list_items_in(elements: list[Element]) -> list[tuple]:
    """`(kind, marker, key, element, position)` for every list item printed in
    these elements, in reading order."""
    out: list[tuple] = []
    for el in elements:
        if el.kind == KIND_FIGURE or not el.text:
            continue
        # "…on page 23." is a page reference, never the next item's number.
        parts = _ITEM_MARKER_RE.split(_PAGE_REF_RE.sub(" ", el.text))
        for position, i in enumerate(range(1, len(parts) - 1, 2)):
            marker = parts[i].strip()
            key = _normalise(parts[i + 1])[:_ITEM_KEY_CHARS]
            if len(key) >= _ITEM_MIN_KEY:
                out.append((classify_marker(marker), marker, key, el, position))
    return out


def _starts_list(marker: str, kind: str) -> bool:
    value = marker.rstrip(".)").strip().casefold()
    return (kind, value) in (("number", "1"), ("letter", "a"), ("roman", "i"))


def _span_of(markers: list[str]) -> str:
    return markers[0] if markers[0] == markers[-1] else f"{markers[0]}\u2013{markers[-1]}"


def numbering_changes(exp: list[Element], act: list[Element]) -> list[dict]:
    """Lists whose items are the same on both sides but marked in another style.

    Items pair on their own words, and only where those words occur once on
    each side - the "never guess" rule the rest of the pairing uses. Changed
    items are then grouped back into the lists they belong to, so a renumbered
    five-step procedure is one finding naming all five markers, not five.
    """
    def by_key(items: list[tuple]) -> dict[str, list[tuple]]:
        grouped: dict[str, list[tuple]] = {}
        for item in items:
            grouped.setdefault((item[3].section, item[2]), []).append(item)  # same words, same topic
        return grouped

    # Words printed more than once ("Select a language from the list." in two
    # rows of one table) pair in order, when both sides print them the same
    # number of times. Dropped outright, as before, a renumbered list went
    # unreported whenever two of its steps read alike.
    exp_items, act_items = by_key(_list_items_in(exp)), by_key(_list_items_in(act))
    pairs = [pair for k in exp_items.keys() & act_items.keys()
             if len(exp_items[k]) == len(act_items[k])
             for pair in zip(exp_items[k], act_items[k])]
    changed = sorted(
        (pair for pair in pairs if _marker_kind(pair[0][1]) != _marker_kind(pair[1][1])),
        key=lambda pair: (pair[0][3].order, pair[0][4]),
    )

    runs: list[list[tuple]] = []
    for pair in changed:
        run = runs[-1] if runs else None
        previous = run[-1] if run else None
        if (
            previous is not None
            and (previous[0][0], previous[1][0]) == (pair[0][0], pair[1][0])
            and pair[0][3].order - previous[0][3].order <= 1
            and not _starts_list(pair[0][1], pair[0][0])
        ):
            run.append(pair)
        else:
            runs.append([pair])

    out: list[dict] = []
    for run in runs:
        exp_markers = [p[0][1] for p in run]
        act_markers = [p[1][1] for p in run]
        count = len(run)
        exp_els = list({id(p[0][3]): p[0][3] for p in run}.values())
        act_els = list({id(p[1][3]): p[1][3] for p in run}.values())
        out.append({
            "type": "numbering",
            "kind": KIND_TEXT,
            "summary": (
                f"List numbering changed — “{run[0][0][2][:70]}” is {_marker_phrase(exp_markers[0])} in Production "
                f"but {_marker_phrase(act_markers[0])} in Staging"
                f"{f' (and {count - 1} more like it)' if count > 1 else ''}."
            ),
            "detail": f"First item: “{run[0][0][2][:70]}”. Production: {', '.join(exp_markers)} · Staging: {', '.join(act_markers)}",
            "exp": merge_elements(exp_els),
            "act": merge_elements(act_els),
            # (Production marker, Staging marker, item key, Production element,
            # Staging element) per item, so a report can box each marker itself.
            "items": [(p[0][1], p[1][1], p[0][2], p[0][3], p[1][3]) for p in run],
        })
    return out


def bullet_glyph_changes(exp: list[Element], act: list[Element]) -> list[dict]:
    """A bulleted list drawn with a DIFFERENT bullet character - Production's
    small round "•" against Staging's larger "●", say - both classify as the
    same "bullet" kind, so `numbering_changes` (which only catches switching
    BETWEEN kinds: number, letter, roman, bullet) never sees it. Still a real,
    visible style change across a whole list, one glyph swapped for another
    throughout - not just a numbering-style change."""
    def by_key(items: list[tuple]) -> dict[tuple, list[tuple]]:
        grouped: dict[tuple, list[tuple]] = {}
        for item in items:
            grouped.setdefault((item[3].section, item[2]), []).append(item)
        return grouped

    exp_items, act_items = by_key(_list_items_in(exp)), by_key(_list_items_in(act))
    pairs = [pair for k in exp_items.keys() & act_items.keys()
             if len(exp_items[k]) == len(act_items[k])
             for pair in zip(exp_items[k], act_items[k])]
    changed = [
        pair for pair in pairs
        if _marker_kind(pair[0][1]) == "bullet" and _marker_kind(pair[1][1]) == "bullet"
        and pair[0][1].strip() != pair[1][1].strip()
    ]
    groups: dict[tuple[str, str], list[tuple]] = {}
    for pair in changed:
        groups.setdefault((pair[0][1].strip(), pair[1][1].strip()), []).append(pair)

    out: list[dict] = []
    for (exp_glyph, act_glyph), group in groups.items():
        group.sort(key=lambda pair: (pair[0][3].order, pair[0][4]))
        exp_els = list({id(p[0][3]): p[0][3] for p in group}.values())
        act_els = list({id(p[1][3]): p[1][3] for p in group}.values())
        count = len(group)
        out.append({
            "type": "list-marker-glyph", "kind": KIND_TEXT,
            "summary": (
                f"Bullet style changed in Staging — Production marks list items with “{exp_glyph}”, "
                f"Staging with “{act_glyph}”."
                + (f" Same change on {count} items in this chapter." if count > 1 else "")
            ),
            "detail": f"First item: “{group[0][0][2][:70]}”.",
            "exp": merge_elements(exp_els),
            "act": merge_elements(act_els),
        })
    return out



# Relative marker font-size change big enough to flag - a bullet/number drawn
# at roughly this much larger or smaller than its counterpart no longer reads
# as "the same marker", even though body-text font-size drift generally is not
# reported (see `_MINOR_TYPES` / content.py's style check).
_MARKER_SIZE_RATIO = 0.15


def _marker_size_at(doc: fitz.Document, el: Element, marker: str) -> float:
    """The font size the marker glyph is drawn at, read off the page rather
    than off `el`'s own dominant size (which is the ITEM's body text, not
    necessarily the marker beside/before it). 0.0 when it cannot be measured -
    a vector-drawn dot has no font size at all."""
    marker = (marker or "").strip()
    if not el.boxes or not marker:
        return 0.0
    page_index, bbox = el.boxes[0]
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return 0.0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            lb = line.get("bbox")
            if not lb or not (bbox[1] - 2 <= lb[1] <= bbox[3] + 2):
                continue
            for span in line.get("spans", []):
                if marker in span.get("text", ""):
                    return float(span.get("size", 0.0))
    return 0.0


def marker_size_changes(exp: list[Element], act: list[Element],
                        expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """The same list items, marked with the SAME kind of marker on both sides
    (e.g. a bullet on both), but that marker prints at a noticeably different
    SIZE in Staging - a bullet drawn much larger or smaller than its Production
    counterpart. A marker whose KIND changed (bullet -> number) is
    `numbering_changes`'s finding, not this one - here the words and the
    marker's style both match, only its size does not.
    """
    def by_key(items: list[tuple]) -> dict[tuple, list[tuple]]:
        grouped: dict[tuple, list[tuple]] = {}
        for item in items:
            grouped.setdefault((item[3].section, item[2]), []).append(item)
        return grouped

    exp_items, act_items = by_key(_list_items_in(exp)), by_key(_list_items_in(act))
    same_kind_pairs = [
        pair for k in exp_items.keys() & act_items.keys()
        if len(exp_items[k]) == len(act_items[k])
        for pair in zip(exp_items[k], act_items[k])
        if _marker_kind(pair[0][1]) == _marker_kind(pair[1][1])
    ]

    changed: list[tuple] = []
    for exp_it, act_it in same_kind_pairs:
        exp_size = _marker_size_at(expected, exp_it[3], exp_it[1])
        act_size = _marker_size_at(actual, act_it[3], act_it[1])
        # Measured against the item's own text: a 5pt bullet on Production's
        # 5pt regulatory text is the same bullet as Staging's 11pt one on 11pt
        # text. Only a marker out of proportion to its words has changed size.
        exp_body, act_body = exp_it[3].size or exp_size, act_it[3].size or act_size
        if exp_size and act_size and exp_body and act_body:
            rel_e, rel_s = exp_size / exp_body, act_size / act_body
            if abs(rel_s - rel_e) / rel_e >= _MARKER_SIZE_RATIO:
                changed.append((exp_it, act_it, exp_size, act_size))
    changed.sort(key=lambda t: (t[0][3].order, t[0][4]))

    runs: list[list[tuple]] = []
    for item in changed:
        run = runs[-1] if runs else None
        previous = run[-1] if run else None
        if (
            previous is not None
            and item[0][3].order - previous[0][3].order <= 1
            and not _starts_list(item[0][1], item[0][0])
        ):
            run.append(item)
        else:
            runs.append([item])

    out: list[dict] = []
    for run in runs:
        exp_els = list({id(t[0][3]): t[0][3] for t in run}.values())
        act_els = list({id(t[1][3]): t[1][3] for t in run}.values())
        count = len(run)
        exp_size, act_size = run[0][2], run[0][3]
        out.append({
            "type": "marker-size",
            "kind": KIND_TEXT,
            "summary": (
                f"List marker size changed — “{run[0][0][2][:70]}”'s marker prints at "
                f"{exp_size:.0f}pt in Production but {act_size:.0f}pt in Staging"
                f"{f' (and {count - 1} more like it)' if count > 1 else ''}."
            ),
            "detail": f"First item: “{run[0][0][2][:70]}”. Production: {exp_size:.1f}pt · Staging: {act_size:.1f}pt",
            "exp": merge_elements(exp_els),
            "act": merge_elements(act_els),
        })
    return out


# --- tables continued onto another page --------------------------------------


def table_header_repeats(actual: fitz.Document, actual_path: str | None, pages: list[int]) -> list[dict]:
    """A Staging table that continues onto the next page must reprint its header
    row there. Present: nothing to report. Missing: a bug. Production is not
    consulted - the rule is about the document being validated."""
    if not actual_path:
        return []
    tables: list[tuple[int, dict]] = []
    for page_index in pages:
        try:
            tables.extend((page_index, t) for t in get_tables(actual_path, page_index, actual))
        except Exception:
            continue
    out: list[dict] = []
    for anchor_page, t in _stitch_continued_tables(actual, tables):
        for cont in t.get("continuations") or []:
            if cont.get("header_repeated"):
                continue
            where = Element(kind=KIND_TABLE, text=cont.get("first_row") or "",
                            boxes=[(cont["page"], tuple(float(v) for v in cont["bbox"]))])
            out.append({
                "type": "table-header-repeat", "kind": KIND_TABLE, "exp": None, "act": where,
                "summary": (f"Table head missing on the continued page — the table that starts on Staging "
                            f"p.{anchor_page + 1} continues onto p.{cont['page'] + 1} and its head row is not "
                            f"printed again there, so the continued rows have no column headings."),
                "detail": (f"Header row: “{(t.get('header') or '')[:100]}” · first row on p.{cont['page'] + 1}: "
                           f"“{(cont.get('first_row') or '')[:100]}”"),
            })
    return out


# --- list items (ul / li) -----------------------------------------------------

_LIST_ITEM_MIN_KEY = 12
_MARKER_REACH = 30.0          # points left of an item's first word its marker may sit
_MARKER_TAIL_RE = re.compile(
    r"(\(\d{1,2}\)|\((?:ii|iii|iv|vi|vii|viii|ix|xi|xii)\)|\([a-z]\)"
    r"|\d{1,2} ?[.)]|(?:ii|iii|iv|vi|vii|viii|ix|xi|xii) ?[.)]|[a-z] ?[.)]"
    r"|[\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219\u25cf\u25a0\u2013\u2014-])\s*$",
    re.IGNORECASE,
)
_CHAR_CACHE: dict[tuple, list] = {}


def _page_chars(doc: fitz.Document, page_index: int) -> list[tuple]:
    key = (doc_key(doc), page_index)
    if key not in _CHAR_CACHE:
        out = []
        try:
            for block in doc[page_index].get_text("rawdict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        for ch in span.get("chars", []):
                            out.append((tuple(ch["bbox"]), ch["c"]))
        except Exception:
            pass
        _CHAR_CACHE[key] = out
    return _CHAR_CACHE[key]


def _marker_before(
    doc: fitz.Document, pages: list[int], key: str, near: tuple[int, float] | None = None
) -> str | None:
    """What is printed just left of this item's first words on the page: a
    marker ("1.", "a)", "•", or a small drawn dot), "" when the words are found
    with nothing before them, None when they cannot be found at all.

    This is read off the page, not off the extracted paragraph, because the two
    documents extract markers differently and the extraction is exactly what
    cannot be trusted here: a table cell "(W × H × D)" parsed as list item
    "D)", a wrapped line read as a new item.

    A short opening phrase can print twice on one page - a numbered heading
    ("2. Remove the back cover.") followed by a body paragraph that happens to
    open with the very same words ("Remove the back cover from its bottom as
    illustrated."). Taking the page's first hit for a 4-word phrase then reads
    the heading's own "2." as this item's marker. Widened to as many of the
    item's own words as the page actually prints together, so a hit that stops
    short - the heading ends after "cover.", the item's words keep going - is
    never mistaken for the item's own line.

    Widening the phrase cannot separate two lines that print the EXACT same
    sentence twice, word for word - two assembly methods on one page both
    ending "Turn the handle counterclockwise to fix it to the desk.". `near`
    (the item's own known page and y-position, when the caller has it) breaks
    that tie by proximity instead of always taking whichever prints first.
    """
    words = key.split()

    def hits_for(page_index: int) -> list:
        span = min(4, len(words))
        try:
            hits = doc[page_index].search_for(" ".join(words[:span]))
        except Exception:
            return []
        while len(hits) > 1 and span < len(words):
            try:
                wider = doc[page_index].search_for(" ".join(words[:span + 1]))
            except Exception:
                break
            if not wider:
                break
            hits, span = wider, span + 1
        out: list = []
        for r in hits:
            prev = out[-1] if out else None
            if prev is not None and 0 <= r.y0 - prev.y1 <= prev.height and r.x0 < prev.x0:
                continue  # the same phrase wrapping onto its next line
            out.append(r)
        return out

    def read_at(page_index: int, r: "fitz.Rect") -> str | None:
        """The marker (or "" / None) this one hit's own line carries -
        None here means "this hit is mid-sentence, not a candidate at all",
        distinct from the function's own None (nothing found anywhere)."""
        # On the item's own line only: tightly set lines overlap their glyph
        # boxes, and the line above's "4." glued onto this "5." read "45..".
        left = sorted(
            ((bbox, c) for bbox, c in _page_chars(doc, page_index)
             if r.x0 - _MARKER_REACH <= bbox[2] <= r.x0 + 1 and r.y0 <= (bbox[1] + bbox[3]) / 2 <= r.y1),
            key=lambda bc: bc[0][0],
        )
        joined = "".join(c for _, c in left)
        m = _MARKER_TAIL_RE.search(joined)
        if m:
            # A marker starts its printed line. "10°C - 60°C" has a dash right
            # before its next word as well, and that is no list: say it cannot
            # be told, rather than calling it a marker or no marker.
            return "\x00" if joined[: m.start()].strip() else m.group(1)  # \x00: caller maps to None
        if any(ch.isalnum() for ch in joined):
            # A word right before them: the phrase runs on mid-sentence here
            # ("clamp and grommet mounting"), not on the item's own line - not
            # a candidate for this hit at all.
            return None
        if _icon_leads_line(doc, page_index, r):
            # A callout icon (Production's own lightbulb TIP mark, with no
            # separate "TIP:" word at all) sits right where a bullet would -
            # its own internal strokes (the flame, the rays) are small drawn
            # shapes too, and the check below would otherwise read one of
            # them as a bullet dot. The icon itself, not a list marker.
            return ""
        # A drawn dot is a bullet only when nothing printed ("*:") sits
        # between it and the words, and it is not a stroke of an icon image.
        printed_x1 = max((bbox[2] for bbox, c in left if c.strip()), default=None)
        try:
            icons = [fitz.Rect(i["bbox"]) for i in doc[page_index].get_image_info() if i.get("bbox")]
            for d in doc[page_index].get_drawings():
                dr = d.get("rect")
                if dr is None or dr.width > 9 or dr.height > 9:
                    continue
                if not (r.x0 - _MARKER_REACH <= dr.x1 <= r.x0 + 1 and dr.y0 < r.y1 and dr.y1 > r.y0):
                    continue
                if printed_x1 is not None and dr.x1 <= printed_x1:
                    continue
                centre = fitz.Point((dr.x0 + dr.x1) / 2, (dr.y0 + dr.y1) / 2)
                if any(icon.contains(centre) for icon in icons):
                    continue
                return "•"
        except Exception:
            pass
        return ""

    for page_index in pages:
        candidates = []
        for r in hits_for(page_index):
            result = read_at(page_index, r)
            if result is not None:
                candidates.append((r, None if result == "\x00" else result))
        if not candidates:
            continue
        if len(candidates) == 1 or near is None or near[0] != page_index:
            return candidates[0][1]
        r, result = min(candidates, key=lambda rc: abs((rc[0].y0 + rc[0].y1) / 2 - near[1]))
        return result
    return None


def _row_carries_marker(table: Element, key: str, marker: str, doc: fitz.Document | None = None) -> bool:
    """A row of `table` whose own first cell is `marker` ("4." / "4") and
    whose other cells open with the item's words. A first cell the table
    reader left empty - the number set on the row's top rule - is read off
    the page inside that cell."""
    want = re.sub(r"[.)\s]+$", "", marker or "").casefold()
    opening = _TOKEN_RE.findall(key)[:4]
    if not want or not opening:
        return False
    for i, row in enumerate(table.cells or ()):
        if not row:
            continue
        rest = [c for c in row[1:] if c and c.strip()]
        if not rest or _TOKEN_RE.findall(" ".join(rest).casefold())[:len(opening)] != opening:
            continue
        first = (row[0] or "").strip()
        if not first and doc is not None and i < len(table.row_boxes):
            page_index, (x0, y0, x1, y1) = table.row_boxes[i]
            cell = table.grid[i][0] if i < len(table.grid) and table.grid[i] and table.grid[i][0] else None
            if cell:
                x0, x1 = cell
            try:
                first = doc[page_index].get_textbox(fitz.Rect(x0, y0 - 4, x1, y1)).strip()
            except Exception:
                first = ""
        if re.sub(r"[.)\s]+$", "", first).casefold() == want:
            return True
    return False


def _icon_leads_line(doc: fitz.Document, page_index: int, r: "fitz.Rect") -> bool:
    """A real icon/figure (not a loose drawn shape) sits immediately left of
    this printed line, icon-sized and close enough to be its own leading
    mark - mirrors `_is_leading_icon`, from the icon's side rather than the
    line's."""
    try:
        figs = get_figures(doc, page_index, print_ready=True)
    except Exception:
        return False
    for f in figs:
        x0, y0, x1, y1 = f.bbox
        if x1 - x0 > _ICON_MAX_SIDE or y1 - y0 > _ICON_MAX_SIDE:
            continue
        overlap = min(r.y1, y1) - max(r.y0, y0)
        if overlap >= 0.5 * min(r.y1 - r.y0, y1 - y0) and 0 <= r.x0 - x1 <= _LEADING_ICON_GAP:
            return True
    return False


def list_structure_changes(exp: list[Element], act: list[Element],
                           expected: fitz.Document | None = None,
                           actual: fitz.Document | None = None,
                           partners: dict[int, list[Element]] | None = None) -> list[dict]:
    """List items that stopped being list items, or started.

    An item Production prints as "• Keep the remote dry" whose words Staging
    prints with no marker at all has lost its list formatting (the <li> became
    a plain paragraph) - and the reverse. A marker that only changed STYLE is
    `numbering_changes`; an item whose words are gone is a content difference.

    `partners` (element id -> the other side's elements it was paired with)
    names each item's own counterpart; searching the other side for the
    item's words is only the fallback for an item left unpaired.
    """
    partners = partners or {}
    # Kept copy by copy: a topic printing the same bulleted sentence twice
    # has two items to check, not one.
    exp_items = [((it[3].section, it[2]), it) for it in _list_items_in(exp)]
    act_items = [((it[3].section, it[2]), it) for it in _list_items_in(act)]
    out: list[dict] = []
    restyled: dict[tuple, list] = {}
    for items, other_items, own, other, kind, own_doc, other_doc in (
        (exp_items, act_items, exp, act, "list-marker-missing", expected, actual),
        (act_items, exp_items, act, exp, "list-marker-added", actual, expected),
    ):
        groups: dict[tuple, list] = {}
        other_marked = Counter(k for k, _ in other_items)
        seen: Counter = Counter()
        for (section, key), item in items:
            seen[(section, key)] += 1
            if seen[(section, key)] <= other_marked[(section, key)] or len(key) < _LIST_ITEM_MIN_KEY:
                continue
            # A numbered section heading ("2. Remove the back cover.") is not a
            # list item - its "2." is compared as a heading, not lost list
            # formatting - and the holder search below never accepts a heading
            # as an item's counterpart either. Without this, a heading with no
            # counterpart search target falls through to whichever body
            # paragraph happens to start with the same words and is wrongly
            # reported as having lost its marker.
            if item[3].kind == KIND_HEADING:
                continue
            # Same kind of block only: a list item never answers to a heading
            # ("proxy settings" inside "Configuring proxy settings").
            candidates = [e for e in other if e.kind not in (KIND_FIGURE, KIND_HEADING) and e.section == section]
            # The same sentence printed twice in a topic ("A special rear
            # projection screen is required." as the Rear note and again as
            # Rear Ceiling's bullet) answers copy for copy, in order - the
            # first copy on the other side is not every copy's counterpart.
            own_copies = [e for e in own if e.kind not in (KIND_FIGURE, KIND_HEADING)
                          and e.section == section and key in e.key]
            copies = [e for e in candidates if key in e.key]
            nth = next((k for k, e in enumerate(own_copies) if e is item[3]), 0)
            holder = copies[min(nth, len(copies) - 1)] if copies else None
            paired = [e for e in partners.get(id(item[3]), ()) if e.kind not in (KIND_FIGURE, KIND_HEADING)]
            if paired:
                # The element the item was actually paired with - its first
                # part holding the item's opening words, else its first part.
                opening = " ".join(key.split()[:4])
                holder = next((e for e in paired if opening in e.key), paired[0])
            if holder is None:
                # The item's own wrapped sentence can be split, on the OTHER
                # side, into its own element apart from its later lines (a
                # figure sitting between them, a page-template artefact) - the
                # full 80-char key is then never contained whole in any one
                # element, even though that element's own (shorter) text is a
                # clean, unbroken prefix of it. Matched the other way round for
                # exactly that case, floored so a short fragment can't match
                # everything.
                holder = next(
                    (e for e in candidates if e.key and len(e.key) >= _ITEM_MIN_KEY and e.key in key), None
                )
            if own_doc is None or other_doc is None:
                continue
            # A sentence split by extraction into several small fragments - a
            # marker column read as its own element, "fix" and "the C-clamp."
            # printed as two pieces of what is one sentence on the page - never
            # holds the item's full key in any ONE element's own text, so no
            # `holder` is ever found by containment alone. The marker is still
            # read straight off the page (see `_marker_before`), which does not
            # care how element collection carved up the words above it - only
            # WHICH pages to look at is needed, and every element already
            # placed in this section between them cover it.
            pages = (sorted({p for e in paired for p in e.pages}) if paired
                     else holder.pages if holder is not None else sorted({p for e in candidates for p in e.pages}))
            if not pages:
                continue
            # Confirmed on the pages: a marker printed before the item on this
            # side, and the same words found on the other side with none. Each
            # search is anchored to where that side's own element already sits
            # (`near`) - two assembly methods on one page can both end "Turn
            # the handle counterclockwise to fix it to the desk.", and only
            # the item's own known position tells the two apart.
            own_marker = _marker_before(own_doc, item[3].pages, key, near=(item[3].page, item[3].bbox[1]))
            if not own_marker:
                continue
            other_near = (holder.page, holder.bbox[1]) if holder is not None else None
            other_marker = _marker_before(other_doc, pages, key, near=other_near)
            if other_marker is None:
                continue
            if not other_marker and any(
                    t.kind == KIND_TABLE and _row_carries_marker(t, key, own_marker, other_doc)
                    for t in ([holder] if holder is not None else []) + candidates):
                # A table printing the number in a column of its own, set above
                # the item's line (Staging's "4." top-aligned in its cell) - the
                # line read finds nothing beside the words, the row has it.
                continue
            # The two documents set their callouts differently: Production
            # marks a note with a small icon and bullets its lines, Staging
            # prints the same words in a labelled, shaded "NOTE:" / "TIP:"
            # box, which needs no bullet. That is the note's styling on each
            # side, not a marker Staging lost, so the same layout is never
            # expected of it.
            if _callout_restyled(item[3], holder, own_doc, other_doc):
                continue
            if holder is None:
                # Nothing to box precisely - anchor the finding at the section's
                # own first element rather than not reporting it at all.
                holder = candidates[0] if candidates else item[3]
            if other_marker:
                # Both sides mark the item, in another style: a numbered step
                # lettered in Staging's table, a bullet printed as "-". The text
                # pairing never sees these - the item's words read differently
                # once Staging glues the note printed after it onto them.
                if kind == "list-marker-missing" and _marker_kind(other_marker) != _marker_kind(own_marker):
                    restyled.setdefault((id(item[3]), id(holder)), [item[3], holder, []])[2].append(
                        (own_marker, other_marker, key))
                continue
            groups.setdefault((id(item[3]), id(holder)), [item[3], holder, []])[2].append(item)
        for element, holder, found in groups.values():
            markers = ", ".join(i[1] for i in found[:6]) + (" …" if len(found) > 6 else "")
            first = found[0][2][:70]
            count = len(found)
            noun = "item" if count == 1 else "items"
            others = f" (and {count - 1} more)" if count > 1 else ""
            if kind == "list-marker-missing":
                summary = (f"List marker missing in Staging — “{first}”{others} is {_marker_phrase(found[0][1])} "
                           f"in Production but has no marker in Staging.")
                exp_el, act_el = element, holder
            else:
                summary = (f"List marker added in Staging — “{first}”{others} is {_marker_phrase(found[0][1])} "
                           f"in Staging but has no marker in Production.")
                exp_el, act_el = holder, element
            out.append({"type": kind, "kind": KIND_TEXT, "exp": exp_el, "act": act_el,
                        "summary": summary, "detail": ""})
    for element, holder, found in restyled.values():
        exp_markers = [f[0] for f in found]
        act_markers = [f[1] for f in found]
        count = len(found)
        out.append({
            "type": "numbering", "kind": KIND_TEXT, "exp": element, "act": holder,
            "summary": (
                f"List numbering changed — “{found[0][2][:70]}” is {_marker_phrase(exp_markers[0])} in Production "
                f"but {_marker_phrase(act_markers[0])} in Staging"
                f"{f' (and {count - 1} more like it)' if count > 1 else ''}."
            ),
            "detail": (f"First item: “{found[0][2][:70]}”. Production: {', '.join(exp_markers)} · "
                       f"Staging: {', '.join(act_markers)}"),
            "items": [(f[0], f[1], f[2], element, holder) for f in found],
        })
    return out


# --- list indent (ul / li alignment) ---------------------------------------
#
# The same list item on both sides, aligned differently WITHIN ITS LIST: a
# sub-item flattened to the first level or pushed a level deeper, or a wrapped
# second line that no longer lines up where Production lines it up (under the
# item's first word, for a hanging indent). Measured off the page, since an
# extracted paragraph has no geometry for a single item inside it.
#
# Relative, never absolute: the two documents are set on different page sizes
# with different margins (a 669pt page against A4), so every list in the
# document "moves" against the text column - a template difference, not a
# misaligned list, and reporting it put 28 identical findings in one run.
_INDENT_TOLERANCE = 6.0      # points an item may move before it is reported ...
_INDENT_MAX = 48.0           # ... and at most this: further is a different layout (a table cell,
                             # a second column), measured against the wrong line, not an indent
_NEXT_ITEM_RE = re.compile(r"^\s*(?:[•●▪◦·\-–—]|\(?[0-9]{1,2} ?[.)]|\(?[a-zA-Z] ?[.)])\s")
_LINE_GEOM_CACHE: dict[tuple, list] = {}


def _text_lines(doc: fitz.Document, page_index: int) -> list[tuple[tuple, str]]:
    """Every text line on the page as (bbox, text)."""
    key = (doc_key(doc), page_index)
    if key not in _LINE_GEOM_CACHE:
        out: list[tuple[tuple, str]] = []
        try:
            for block in doc[page_index].get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if text:
                        out.append((tuple(float(v) for v in line["bbox"]), text))
        except Exception:
            pass
        _LINE_GEOM_CACHE[key] = out
    return _LINE_GEOM_CACHE[key]


def _item_geometry(doc: fitz.Document, el: Element, key: str):
    """Where a list item prints: (text start, wrapped-line start or None, page,
    box), both starts measured from the left edge of the text column. None when
    the item's words cannot be found on the page."""
    phrase = " ".join(key.split()[:4])
    words = set(el.key.split())
    for page_index, bbox in el.boxes:
        clip = fitz.Rect(bbox) + (-3, -3, 3, 3)
        try:
            hits = doc[page_index].search_for(phrase, clip=clip)
        except Exception:
            hits = []
        if not hits:
            continue
        r = hits[0]
        left, _ = _text_column(doc, page_index)
        height = max(4.0, r.height)
        below = [
            (lb, text) for lb, text in _text_lines(doc, page_index)
            if r.y1 - 1 <= lb[1] <= r.y1 + 0.9 * height and lb[2] > r.x0 - _MARKER_REACH
            and lb[0] < clip.x1 and lb[3] <= clip.y1 + 2
        ]
        wrap = None
        if below:
            lb, text = min(below, key=lambda t: t[0][0])
            first = _normalise(text).split()[:1]
            if not _NEXT_ITEM_RE.match(text) and first and first[0] in words:
                wrap = lb[0] - left
                box = (max(left, r.x0 - _MARKER_REACH), r.y0 - 1, max(r.x1, bbox[2]), lb[3] + 1)
            else:
                box = (max(left, r.x0 - _MARKER_REACH), r.y0 - 1, max(r.x1, bbox[2]), r.y1 + 1)
        else:
            box = (max(left, r.x0 - _MARKER_REACH), r.y0 - 1, max(r.x1, bbox[2]), r.y1 + 1)
        return r.x0 - left, wrap, page_index, box
    return None


def list_indent_changes(exp: list[Element], act: list[Element],
                        expected: fitz.Document | None = None,
                        actual: fitz.Document | None = None,
                        scale: float = 1.0) -> list[dict]:
    """List items printed at another indent in Staging, one finding per kind of
    shift - every item that moved alike is boxed under it."""
    if expected is None or actual is None:
        return []

    def unique(items: list[tuple]) -> dict[str, tuple]:
        seen: dict[str, tuple | None] = {}
        for item in items:
            composite = (item[3].section, item[2])  # same words, same topic
            seen[composite] = None if composite in seen else item
        return {k: v for k, v in seen.items() if v is not None}

    # A table's grid and a numbered heading are not list geometry.
    listed = lambda els: [e for e in els if e.kind not in (KIND_TABLE, KIND_HEADING)]  # noqa: E731
    exp_items, act_items = unique(_list_items_in(listed(exp))), unique(_list_items_in(listed(act)))
    measured: list[tuple] = []
    for composite in exp_items.keys() & act_items.keys():
        key = composite[1]
        if len(key) < _LIST_ITEM_MIN_KEY:
            continue
        e, s = exp_items[composite], act_items[composite]
        ge = _item_geometry(expected, e[3], key)
        gs = _item_geometry(actual, s[3], key)
        if ge is not None and gs is not None:
            measured.append((e, s, ge, gs))
    measured.sort(key=lambda m: (m[0][3].order, m[0][4]))

    # The lists themselves: runs of items Production prints together - one
    # page, at most one element apart.
    lists: list[list[tuple]] = []
    for m in measured:
        prev = lists[-1][-1] if lists else None
        if prev is not None and prev[2][2] == m[2][2] and m[0][3].order - prev[0][3].order <= 1:
            lists[-1].append(m)
        else:
            lists.append([m])

    # Production's offsets are scaled to Staging's type size before comparing:
    # an indent of 25pt in 5pt type is the same indent as 60pt in 12pt type.
    tolerance = _INDENT_TOLERANCE * max(1.0, scale)
    limit = _INDENT_MAX * max(1.0, scale)
    def levels(xs: list[float], step: float) -> list[float]:
        starts: list[float] = []
        for x in sorted(xs):
            if not starts or x - starts[-1] > step:
                starts.append(x)
        return starts

    def level_of(x: float, starts: list[float]) -> int:
        return max(i for i, s in enumerate(starts) if x >= s - 0.5)

    # Only what a reader sees is reported: an item at another NESTING LEVEL of
    # its list, or a wrapped line that lost (or gained) its hanging indent. A
    # few points of offset from another page template never is.
    found: list[tuple] = []
    for items in lists:
        starts_e = levels([m[2][0] for m in items], tolerance)
        starts_s = levels([m[3][0] for m in items], tolerance)
        for e, s, ge, gs in items:
            lv_e, lv_s = level_of(ge[0], starts_e), level_of(gs[0], starts_s)
            if len(items) > 1 and lv_e != lv_s and abs(gs[0] - starts_s[0] - (ge[0] - starts_e[0]) * scale) <= limit:
                found.append(("indent", lv_s - lv_e, e, s, ge, gs, lv_e + 1, lv_s + 1))
            elif ge[1] is not None and gs[1] is not None:
                hang_e, hang_s = (ge[1] - ge[0]) * scale, gs[1] - gs[0]
                if (hang_e > tolerance) != (hang_s > tolerance):
                    found.append(("wrap", 1 if hang_s > tolerance else -1, e, s, ge, gs, hang_e, hang_s))

    groups: dict[tuple, list] = {}
    for item in sorted(found, key=lambda f: (f[2][3].order, f[2][4])):
        kind, shift = item[0], item[1]
        groups.setdefault((kind, shift if kind == "wrap" else (item[6], item[7])), []).append(item)

    def level_name(level: int) -> str:
        return "a main item (level 1)" if level == 1 else f"a sub-item (level {level})"

    out: list[dict] = []
    for (kind, _), items in groups.items():
        count = len(items)
        first = items[0][3][3].text.split("  ")[0][:70]
        others = f" (and {count - 1} more like it)" if count > 1 else ""
        if kind == "indent":
            lv_e, lv_s = items[0][6], items[0][7]
            summary = (f"List nesting changed — “{first}”{others} is {level_name(lv_s)} in Staging "
                       f"but {level_name(lv_e)} in Production.")
            detail = ""
        else:
            lost = items[0][1] < 0
            summary = (f"Hanging indent {'lost' if lost else 'added'} — the second line of “{first}”{others} "
                       + ("starts under the marker in Staging; in Production it lines up with the text."
                          if lost else
                          "lines up with the text in Staging; in Production it starts under the marker."))
            detail = ""
        boxes = [
            (Element(kind=KIND_TEXT, text=i[2][3].text, key=i[2][2], boxes=[(i[4][2], i[4][3])]),
             Element(kind=KIND_TEXT, text=i[3][3].text, key=i[3][2], boxes=[(i[5][2], i[5][3])]))
            for i in items
        ]
        out.append({
            "type": "list-indent", "kind": KIND_TEXT,
            "summary": summary, "detail": detail,
            "exp": boxes[0][0], "act": boxes[0][1],
            "repeats": boxes[1:],
        })
    return out


# --- hyperlinks -------------------------------------------------------------
#
# Links are read straight off each page's link annotations inside the chapter,
# identified by the words they sit on, and compared: a phrase linked on one side
# and plain on the other, a link that goes somewhere else, a link that goes
# nowhere. Where an internal link lands is compared by the HEADING it lands
# under, never by page number - the two documents paginate differently.

KIND_LINK = "link"
KIND_LABEL[KIND_LINK] = "Hyperlink"


_NAMES_CACHE: dict[int, dict] = {}
_LANDING_SLACK = 16.0  # points: a destination sits just above the heading it opens


def _named_destinations(doc: fitz.Document) -> dict:
    key = doc_key(doc)
    if key not in _NAMES_CACHE:
        try:
            _NAMES_CACHE[key] = doc.resolve_names() or {}
        except Exception:
            _NAMES_CACHE[key] = {}
    return _NAMES_CACHE[key]


def _resolve_named(doc: fitz.Document, name: str) -> tuple[int, float] | None:
    """`(page, top-origin y)` of a named destination.

    InDesign names its anchors in the local script ("...indd:錨點 10:98"), and
    the name on the link annotation comes back mis-decoded - its UTF-8 bytes
    read as Latin-1 - so it is looked up under both spellings. The resolved
    point is measured from the bottom of the page and is flipped here.
    """
    names = _named_destinations(doc)
    for candidate in (name, _redecode(name)):
        hit = names.get(candidate) if candidate else None
        if hit and hit.get("page") is not None and 0 <= hit["page"] < doc.page_count:
            to = hit.get("to")
            height = doc[hit["page"]].rect.height
            return hit["page"], (height - float(to[1])) if to else 0.0
    return None


def _redecode(name: str) -> str:
    try:
        return name.encode("latin-1").decode("utf-8")
    except Exception:
        return ""


def _landing_heading(entries: list[TocEntry], page: int, y: float, anchor: str) -> str | None:
    """The section a link lands in. A heading named in the link's own text wins
    when it is printed on the landing page; otherwise the heading the landing
    point sits under, allowing for a destination placed a few points above it."""
    words = normalize_title(anchor)
    # A link box over only the first line of a wrapped title ('"Wireless
    # projection (screen') still names the heading its words begin.
    named = [e for e in entries if e.page == page and normalize_title(e.title)
             and (normalize_title(e.title) in words
                  or (len(words.strip(" \"“”'‘’")) >= 12
                      and normalize_title(e.title).startswith(words.strip(" \"“”'‘’"))))]
    # "Wireless projection (screen casting)" names that heading, not the
    # "Projection" chapter heading its words also contain on the same page.
    named = [e for e in named if not any(
        o is not e and normalize_title(e.title) != normalize_title(o.title)
        and normalize_title(e.title) in normalize_title(o.title) for o in named)]
    if named:
        # A link drawn over a whole paragraph names more than one heading ("…select
        # Web Player. For more details, see Controlling the players"): the one it
        # means is the one it lands on, not the first one mentioned.
        return min(named, key=lambda e: abs(e.y - y)).title
    return heading_at(entries, page, y + _LANDING_SLACK)


def _link_target(doc: fitz.Document, link: dict, entries: list[TocEntry], anchor: str = "") -> str:
    kind = link.get("kind")
    if kind == fitz.LINK_URI:
        return (link.get("uri") or "").strip().rstrip("/").casefold()
    landing = None
    if kind == fitz.LINK_GOTO:
        page = link.get("page", -1)
        if page is not None and 0 <= page < doc.page_count:
            to = link.get("to")
            landing = (page, float(to.y) if to is not None else 0.0)
    elif kind == fitz.LINK_NAMED:
        landing = _resolve_named(doc, link.get("nameddest") or link.get("name") or "")
    if landing is None:
        return f"unresolved:{kind}"
    title = _landing_heading(entries, landing[0], landing[1], anchor)
    return f"section: {title}" if title else f"page {landing[0] + 1}"


def _link_broken(doc: fitz.Document, link: dict) -> str:
    """Why this link does not work, or "" when it does."""
    kind = link.get("kind")
    if kind == fitz.LINK_URI:
        uri = (link.get("uri") or "").strip().lower()
        return "" if uri.startswith(USABLE_SCHEMES) else f"its address “{link.get('uri') or ''}” cannot be opened"
    if kind == fitz.LINK_GOTO:
        page = link.get("page", -1)
        return "" if page is not None and 0 <= page < doc.page_count else "it points at a page that does not exist"
    if kind == fitz.LINK_NAMED:
        name = link.get("nameddest") or link.get("name") or ""
        if _resolve_named(doc, name) or _named_destination_resolves(doc, name):
            return ""
        return f"its destination “{_redecode(name) or name}” does not exist"
    rect = link.get("from")
    if rect is not None and (rect.width < 2 or rect.height < 2):
        return "its clickable area is too small to click"
    return ""


def collect_links(doc: fitz.Document, pages: list[int], span: tuple, entries: list[TocEntry]) -> list[Element]:
    """Every link annotation printed inside the chapter on this side."""
    (start_page, start_y), end = span
    out: list[Element] = []
    for page_index in pages:
        page = doc[page_index]
        y0 = start_y - 2 if page_index == start_page else page.rect.y0
        y1 = end[1] if (end and page_index == end[0]) else page.rect.y1
        try:
            links = page.get_links()
        except Exception:
            continue
        for link in links:
            rect = link.get("from")
            if rect is None or not (y0 <= (rect.y0 + rect.y1) / 2 <= y1):
                continue
            words = " ".join(page.get_text("text", clip=rect).split())
            box = (page_index, tuple(float(v) for v in rect))
            to = link.get("to")
            raw = (link.get("kind"), link.get("uri"), link.get("page"), link.get("nameddest"),
                   round(float(to.y)) if to is not None else None)
            prev = out[-1] if out else None
            # A link that wraps onto the next line is two annotations with one
            # destination - one link, as a reader sees it.
            if (
                prev is not None and getattr(prev, "_raw", None) == raw
                and prev.boxes[-1][0] == page_index
                and 0 <= rect.y0 - prev.boxes[-1][1][3] <= 1.5 * max(rect.height, 1.0)
            ):
                prev.text = f"{prev.text} {words}".strip()
                prev.key = _normalise(prev.text)
                prev.boxes.append(box)
                continue
            el = Element(kind=KIND_LINK, text=words, key=_normalise(words), boxes=[box])
            el._raw = raw
            el.label = _link_target(doc, link, entries, words)
            el.detail = _link_broken(doc, link)
            # A clickable area drawn over a whole paragraph is boxed on the words
            # it links - the heading it lands on - not on the paragraph.
            if rect.height > 20 and el.label.startswith("section: "):
                try:
                    hits = page.search_for(el.label[len("section: "):], clip=rect)
                except Exception:
                    hits = []
                if hits:
                    el.boxes = [(page_index, tuple(float(v) for v in hits[0]))]
            out.append(el)
    return out


def _same_address(x: str, y: str) -> bool:
    """One web address written two ways ("http://support.benq.com." /
    "https://Support.BenQ.com/") is one link target."""
    def norm(u: str) -> str:
        u = (u or "").strip().casefold().rstrip("./")
        return re.sub(r"^https?://(www\.)?", "", u)
    return bool(x) and norm(x) == norm(y)


def _pair_links(exp_links: list[Element], act_links: list[Element]) -> list[tuple]:
    """Pair each link with the one on the other side that sits on the same
    words. The two documents draw a link's clickable area differently - round
    the linked words in one, round the whole sentence in the other - so a link
    pairs when either one's words contain the other's, best fit first."""
    pairs: list[tuple] = []
    used: set[int] = set()
    for a in exp_links:
        if len(a.key) < 3:
            continue
        best, best_score = None, 0.0
        for k, b in enumerate(act_links):
            if k in used or len(b.key) < 3:
                continue
            # Only within the same topic: "Controlling the players" under PDF
            # Player once paired with the same words under Web Player.
            if a.section and b.section and a.section != b.section:
                continue
            if a.key == b.key:
                score = 2.0
            elif a.key in b.key or b.key in a.key:
                score = min(len(a.key), len(b.key)) / max(len(a.key), len(b.key))
            else:
                continue
            if score > best_score:
                best, best_score = k, score
        if best is not None:
            used.add(best)
            pairs.append((a, act_links[best]))
        else:
            pairs.append((a, None))
    pairs.extend((None, b) for k, b in enumerate(act_links) if k not in used and len(b.key) >= 3)
    return pairs


def _on_words(doc: fitz.Document | None, el: Element, words: str) -> Element:
    """The link boxed on `words` inside its clickable area, when they can be
    found there - Staging draws some links over a whole two-line NOTE sentence -
    otherwise the link as it is."""
    if doc is None or not words.strip():
        return el
    for page_index, bbox in el.boxes:
        try:
            hits = doc[page_index].search_for(words, clip=fitz.Rect(bbox) + (-1, -1, 1, 1))
        except Exception:
            hits = []
        if hits:
            narrowed = Element(kind=el.kind, text=el.text, key=el.key, label=el.label, detail=el.detail,
                               boxes=[(page_index, tuple(float(v) for v in r)) for r in hits],
                               section=el.section)
            return narrowed
    return el


def _phrase_anchor(doc: fitz.Document | None, elements: list[Element], phrase: str, section: str) -> Element | None:
    """Where `phrase` prints as plain text, in this chapter's own copy of the
    page - so a link missing/added on the OTHER side still points a reviewer
    at the actual words, not just the top of the topic or chapter they fall
    in. Searched over every page these elements touch, not one element's own
    box, since a link that vanished has no element of its own here to anchor
    on."""
    text = (phrase or "").strip()
    if doc is None or not text:
        return None
    for page_index in sorted({p for e in elements for p in e.pages}):
        try:
            hits = doc[page_index].search_for(text)
        except Exception:
            hits = []
        if hits:
            rect = hits[0]
            for h in hits[1:]:
                rect |= h
            return Element(kind=KIND_TEXT, text=text, key=_normalise(text),
                           boxes=[(page_index, (rect.x0, rect.y0, rect.x1, rect.y1))], section=section)
    return None


def link_changes(exp_links: list[Element], act_links: list[Element],
                 exp_elements: list[Element], act_elements: list[Element],
                 expected: fitz.Document | None = None, actual: fitz.Document | None = None) -> list[dict]:
    exp_blob = " ".join(e.key for e in exp_elements if e.key)
    act_blob = " ".join(e.key for e in act_elements if e.key)
    # A link is paired by the words it sits on; when the two documents make
    # different words clickable, the same link to the same address looked
    # missing on one side and added on the other.
    # ... within the same topic: a link in Production is still reported missing
    # when Staging only links to the same address under some other heading.
    exp_targets = {(link.section, link.label) for link in exp_links}
    act_targets = {(link.section, link.label) for link in act_links}
    # Every address Production links to anywhere in the chapter, scheme taken
    # off - a bare "www.benq.com" in Staging against Production's own
    # "https://www.benq.com" is the SAME address, not a link Staging broke,
    # even when it is not this exact link's own pair that carries the scheme.
    exp_addresses = {_uri_authority(link.label) for link in exp_links}
    out: list[dict] = []
    for a, b in _pair_links(exp_links, act_links):
        if b is None:
            # Only a LINK difference when the words are still there to click:
            # text that is gone altogether is already a content difference.
            if a.key in act_blob and (a.section, a.label) not in act_targets:
                where = _phrase_anchor(actual, act_elements, a.text, a.section)
                out.append({"type": "link-missing", "kind": KIND_LINK, "exp": a, "act": where,
                            "summary": f"Hyperlink missing in Staging — “{a.text[:80]}” is a link in Production and plain text in Staging.",
                            "detail": f"Production link goes to: {a.label}"})
        elif a is None:
            if b.detail and _uri_authority(b.label) not in exp_addresses:
                out.append({"type": "link-broken", "kind": KIND_LINK, "exp": None, "act": b,
                            "summary": f"Hyperlink not working in Staging — “{b.text[:80] or 'link'}”: {b.detail}.",
                            "detail": ""})
            elif not b.detail and b.key in exp_blob and (b.section, b.label) not in exp_targets:
                where = _phrase_anchor(expected, exp_elements, b.text, b.section)
                out.append({"type": "link-added", "kind": KIND_LINK, "exp": where, "act": b,
                            "summary": f"Hyperlink added in Staging — “{b.text[:80]}” is plain text in Production.",
                            "detail": f"Staging link goes to: {b.label}"})
        elif b.detail and not a.detail and _uri_authority(b.label) not in exp_addresses:
            out.append({"type": "link-broken", "kind": KIND_LINK, "exp": a, "act": b,
                        "summary": f"Hyperlink not working in Staging — “{b.text[:80]}”: {b.detail}.",
                        "detail": f"Production link goes to: {a.label}"})
        elif (_same_address(a.label, b.label)):
            pass
        elif (a.label != b.label and _uri_authority(a.label) != _uri_authority(b.label)
              and not (a.label.startswith(("page ", "unresolved")) or b.label.startswith("page "))):
            # "https://www.benq.com" against the bare "www.benq.com", or against
            # "http://www.benq.com" - the same address, spelled with or without
            # a scheme, or with a different one, is not a link that "goes
            # somewhere else"; a viewer opens both the same way.
            out.append({"type": "link-target", "kind": KIND_LINK, "exp": a, "act": _on_words(actual, b, a.text),
                        "summary": f"Hyperlink goes somewhere else — “{a.text[:80]}”.",
                        "detail": f"Production: {a.label} · Staging: {b.label}"})
    return out


_SIBLING_LINK_GAP = 20.0  # pt: consecutive Staging links this close together read as one list


def page_ref_dropped_changes(exp_group: list[Element], act_group: list[Element]) -> list[dict]:
    """A cross-reference's own trailing "on page N" is left out of the WORDING
    comparison elsewhere - the number itself cannot survive two different
    paginations - but Staging dropping the whole clause where Production
    states one is still a real, visible difference: the reader is told to go
    look elsewhere for more detail and is no longer told where. Reported on
    every matched pair it happens on, not once for the whole document - a
    reader meets each one individually, walking through the manual."""
    a_text = " ".join(e.text for e in exp_group if e.text)
    b_text = " ".join(e.text for e in act_group if e.text)
    a_ref = _PAGE_REF_RE.search(a_text)
    if not a_ref or _PAGE_REF_RE.search(b_text):
        return []
    return [{
        "type": "link-page-ref-dropped", "kind": KIND_TEXT,
        "summary": (f"Hyperlink cross-reference drops “on page” in Staging — Production says "
                    f"“{a_ref.group(0).strip()}”, Staging does not name a page there."),
        # The exact phrase ("on page 32"), for the short box comment - never
        # blank: a reader clicking this box needs to see AT A GLANCE which of
        # the two sides is which, not re-read the whole summary sentence.
        "detail": a_ref.group(0).strip(),
    }]


_LABEL_LINE_TOLERANCE = 2  # extra words (a list number, a bullet) still read as "the label's own line"
_SHORT_LABEL_MAX_WORDS = 4  # "Step 12:", "Important:" - longer than this is a sentence, not a lead-in label
# A lead-in label with no bold styling of its own to mark it out - "Step 5:",
# or one of the callout words this codebase already recognises (NOTE/TIP/
# WARNING/…, `i18n._CALLOUT_WORDS`) - still reads as a distinct label a
# reader's eye jumps to, purely from the short phrase-plus-colon shape.
_SHORT_LABEL_LEAD_RE = re.compile(
    r"^\s*(?:step\s*\d{1,3}|" + "|".join(sorted({re.escape(w) for w in i18n._CALLOUT_WORDS}, key=len, reverse=True))
    + r")\s*[:：]",
    re.IGNORECASE,
)


def _label_layout(doc: fitz.Document, group: list[Element]) -> str | None:
    """"inline" when a list item's own leading label ("**Reset**", "Step 5:")
    shares its printed line with the description that follows it, "own_line"
    when the label prints alone and the description starts on the NEXT line
    down - the two house styles a "Label: description" item can use, whether
    the label is set apart by bold or is plain text that merely reads as a
    label from its short, colon-terminated shape. None when this is not that
    kind of item at all.

    The label and its description are sometimes ONE element (a label run
    followed by plain continuation, when they wrap close enough to merge)
    and sometimes TWO (the label's own short line, kept separate from the
    paragraph after it) - `group` is whichever `pair_elements` produced, and
    both shapes are read the same way."""
    if len(group) >= 2:
        first = group[0]
        words = [w for w in _TOKEN_RE.findall((first.text or "").casefold()) if w.isalpha()]
        if not words:
            return None
        if first.bold_words and all(w in first.bold_words for w in words):
            return "own_line"
        if len(words) <= _SHORT_LABEL_MAX_WORDS and _SHORT_LABEL_LEAD_RE.match(first.text or ""):
            return "own_line"
        return None
    if len(group) != 1:
        return None
    el = group[0]
    if not el.text or not el.boxes:
        return None
    words = [w for w in _TOKEN_RE.findall(el.text.casefold()) if w.isalpha()]
    label_words: list[str] = []
    for w in words:
        if w not in el.bold_words:
            break
        label_words.append(w)
    if not label_words:
        m = _SHORT_LABEL_LEAD_RE.match(el.text)
        if m:
            label_words = [w for w in _TOKEN_RE.findall(m.group(0).casefold()) if w.isalpha()]
    if not label_words or len(label_words) >= len(words):
        return None  # no label lead-in, or the whole element is just that label - not this kind of item
    hits = _phrase_hits(doc, el, " ".join(label_words))
    if not hits:
        return None
    page_index, rect = hits[0]
    line_words = _line_words(doc, page_index, rect)
    return "inline" if len(line_words) > len(label_words) + _LABEL_LINE_TOLERANCE else "own_line"


def list_label_layout_changes(exp_group: list[Element], act_group: list[Element],
                              expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A list item's own bold label running inline with its description in one
    document ("**Reset** Poke the reset hole...") but printing on its own line
    in the other ("**Reset**", then the description starts on the next line) -
    the same words, only laid out differently, still a real style change a
    reader notices across the whole list, not just the one item happening to
    also carry an unrelated wording difference."""
    if not exp_group or not act_group or any(e.kind != KIND_TEXT for e in exp_group + act_group):
        return []
    a_layout, b_layout = _label_layout(expected, exp_group), _label_layout(actual, act_group)
    if not a_layout or not b_layout or a_layout == b_layout:
        return []
    return [{
        "type": "list-label-layout", "kind": KIND_TEXT,
        "summary": (
            "List item label runs inline with its description in Staging — Production prints the "
            "label on its own line, with the description starting on the next line down."
            if b_layout == "inline" else
            "List item label prints on its own line in Staging — Production runs it inline with its "
            "description on the same line."
        ),
        "detail": "",
    }]


_STRAY_SPACE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?])")


def _stray_space(group: list[Element]) -> str | None:
    text = " ".join(e.text for e in group if e.text)
    m = _STRAY_SPACE_PUNCT_RE.search(text)
    if not m:
        return None
    lead = " ".join(text[: m.start()].split()[-3:])  # whole words, not a mid-word cut
    return (lead + m.group(0)).strip()


def stray_space_changes(exp_group: list[Element], act_group: list[Element]) -> list[dict]:
    """A stray space directly before a sentence's own closing punctuation -
    "the base ." instead of "the base." - left behind when words in the
    middle of a sentence (often a page-number cross-reference the wording
    comparison already treats as legitimately dropped) were deleted without
    closing up the gap. Invisible to the wording comparison itself, which
    collapses any run of whitespace to one space before two texts are
    compared; checked directly on the printed text instead, and only when it
    is a genuine difference between the two documents - not a stray space
    either one happens to share, which is its own document's styling, not a
    defect this comparison should flag."""
    a_stray, b_stray = _stray_space(exp_group), _stray_space(act_group)
    if b_stray and not a_stray:
        return [{"type": "text-space", "kind": KIND_TEXT,
                 "summary": f"Extra space in Staging — “{b_stray}” has a stray space Production does not.",
                 "detail": ""}]
    if a_stray and not b_stray:
        return [{"type": "text-space", "kind": KIND_TEXT,
                 "summary": f"Extra space in Production — “{a_stray}” has a stray space Staging does not.",
                 "detail": ""}]
    return []


# A picture below this is an icon, not a figure: checked for a broken STREAM
# (which breaks at any size) but not with the render heuristics, which need
# room to tell a corrupt paint from an icon that is legitimately flat.
_ICON_FIGURE_SIDE = 8.0
# Below this there is nothing to show either way - a hairline rule, a bullet
# drawn as a one-pixel image.
_MIN_BROKEN_SIDE = 3.0


def broken_image_changes(actual: fitz.Document, act_pages: list[int], section_at) -> list[dict]:
    """Every picture on the chapter's Staging pages that will not show: its
    image stream is empty or will not decode, or it renders blank/corrupt.

    Icons count. The size floor here used to be the figure floor, so an ICON
    whose PNG was empty or would not decode - the one thing that is broken
    beyond doubt whatever its size - was skipped for being small, and a note
    badge or a button glyph that would not display was never reported.
    """
    from types import SimpleNamespace
    from pdfval.validators.image import _broken_reason
    out: list[dict] = []
    for page in sorted(set(act_pages)):
        try:
            infos = actual[page].get_image_info(xrefs=True)
        except Exception:
            continue
        for info in infos:
            bbox = tuple(float(v) for v in info.get("bbox") or ())
            if len(bbox) != 4:
                continue
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if width < _MIN_BROKEN_SIDE or height < _MIN_BROKEN_SIDE:
                continue  # a hairline or a one-pixel glyph image
            icon = width < _ICON_FIGURE_SIDE or height < _ICON_FIGURE_SIDE
            reason = _broken_reason(actual, page, SimpleNamespace(kind="raster", xref=int(info.get("xref") or 0),
                                                                   bbox=bbox), stream_only=icon)
            if not reason:
                continue
            place = Element(kind=KIND_FIGURE, text="", boxes=[(page, bbox)], section=section_at(page, bbox[1]),
                            width=bbox[2] - bbox[0], height=bbox[3] - bbox[1])
            what = "Icon" if icon else "Image"
            out.append({
                "type": "figure-broken", "kind": KIND_FIGURE, "exp": None, "act": place,
                "section": place.section,
                "summary": f"{what} broken in Staging — {reason}; it will not display.",
                "detail": reason,
            })
    return out


def page_ref_in_staging_changes(act_elements: list[Element], actual: fitz.Document) -> list[dict]:
    """Staging turns every printed "on page N" into a hyperlink with no page
    number - so one it drops is expected (never reported), and one it still
    prints is an issue: boxed on the phrase itself."""
    out: list[dict] = []
    for el in act_elements:
        if el.kind not in (KIND_TEXT, KIND_NOTE, KIND_TABLE, KIND_HEADING) or not el.text:
            continue
        for m in _PAGE_REF_RE.finditer(el.text):
            phrase = " ".join(m.group(0).split())
            boxes = []
            for page, bbox in el.boxes:
                try:
                    hits = actual[page].search_for(phrase, clip=fitz.Rect(bbox) + (-2, -2, 2, 2))
                except Exception:
                    hits = []
                boxes += [(page, tuple(float(v) for v in h)) for h in hits]
            place = Element(kind=KIND_TEXT, text=phrase, key=_normalise(phrase),
                            boxes=boxes[:1] or el.boxes[:1], section=el.section)
            out.append({
                "type": "page-ref-in-staging", "kind": KIND_TEXT, "exp": None, "act": place,
                "section": el.section,
                "summary": (f"Page reference present in Staging — “{phrase}” is printed in Staging; "
                            f"Staging's cross-references are links and must not name a page number."),
                "detail": el.text[:300],
            })
    return out


def page_ref_consistency_changes(act_links: list[Element]) -> list[dict]:
    """Sibling cross-reference links stacked as one list ("How to detach the
    stand (for models with stand)" / "...(for models with ergo arm stand) on
    page 36") are expected to follow one house style for a trailing "on page
    N": if Staging strips it from one sibling it should strip it from all of
    them, or the reader sees an unexplained double standard inside the very
    list meant to look uniform. Judged within Staging alone - no Production
    counterpart is needed, since a page reference is normally legitimate
    content and only its INCONSISTENCY within one visual list is the defect.
    Links are grouped as siblings by proximity: consecutive, left-aligned,
    on the same page, with only a small gap between them - a real list, not
    two unrelated links that merely happen to sit near each other.
    """
    ordered = sorted((el for el in act_links if el.boxes), key=lambda el: (el.boxes[0][0], el.boxes[0][1][1]))
    out: list[dict] = []
    group: list[Element] = []

    def flush() -> None:
        if len(group) < 2:
            return
        has_ref = [bool(_PAGE_REF_RE.search(el.text or "")) for el in group]
        if not any(has_ref) or all(has_ref):
            return
        for el, ref in zip(group, has_ref):
            if not ref:
                continue
            out.append({
                "type": "link-page-ref-inconsistent", "kind": KIND_LINK, "exp": None, "act": el,
                "summary": (f"Hyperlink cross-reference keeps “on page” inconsistently — “{el.text[:80]}” "
                            "still names a page number while a sibling item in the same list has it removed."),
                "detail": "Expected: not rendered in Staging, matching the other item(s) in this list.",
            })

    for el in ordered:
        page_index, bbox = el.boxes[0]
        if group:
            prev_page, prev_last_bbox = group[-1].boxes[-1]
            prev_x0 = group[-1].boxes[0][1][0]
            same_list = (
                page_index == prev_page
                and 0 <= bbox[1] - prev_last_bbox[3] <= _SIBLING_LINK_GAP
                and abs(bbox[0] - prev_x0) <= 15.0
            )
            if not same_list:
                flush()
                group = []
        group.append(el)
    flush()
    return out


# --- figures ----------------------------------------------------------------
#
# Figures are NOT aligned in reading order with the text. Order is exactly what
# breaks when one document drops a picture or moves it: every figure after it
# was paired with its neighbour's counterpart, and a "WiFi cover" drawing was
# reported as "replaced" by the "User documents" booklet printed beside it.
#
# Instead every figure in the chapter is matched against every figure on the
# other side, on two pieces of evidence: how it LOOKS, and the label printed
# beside it ("User documents", "WiFi cover with screws") - the label is what
# a reader uses to tell two similar line drawings apart. A figure left without
# a partner is then looked for across the WHOLE other document before it is
# called missing, because a figure moved into another section is not missing -
# it is in the wrong place, and the report says where.

_CAPTION_REACH = 110.0       # points: how far from a figure's detected box its label may be printed
_CAPTION_MAX_CHARS = 120
_FIGURE_MATCH = 0.60         # looks alike enough to be the same figure on its own
_FIGURE_MATCH_WITH_LABEL = 0.20  # ... when the label beside it matches too
_FIGURE_MATCH_DRAWN = 0.30   # a drawn figure against an embedded one, within the topic
_SIDE_BY_SIDE_Y = 12.0      # points: figures whose tops are this close sit in one row
_SIDE_BY_SIDE_GAP = 40.0    # points: ... and this close across are one group of artwork
_FIGURE_ELSEWHERE = 0.72     # a figure found elsewhere must look this alike
_LABEL_BONUS = 0.35
# A UI mock-up screen (a setup wizard's "Welcome" / "Language" / "Setup the
# board" pages, say) shares almost all of its chrome - the same background,
# button placement, panel style - with every OTHER step's screen in the same
# wizard, so two DIFFERENT steps' screenshots routinely score past
# `_FIGURE_SAME` on template similarity alone, even though their captions
# ("2. On the Welcome page..." vs "3. On the Language page...") plainly
# disagree. Overriding an explicit caption disagreement - as opposed to
# allowing a cross-topic match, or the "no caption" case where two figures'
# only evidence is how they look - needs a much higher bar than "same
# picture": close enough to near-identical that it really is the one image
# repeated, not two different screens of the same template.
_FIGURE_SAME_DESPITE_LABEL = 0.95
_FIG_INDEX: dict[int, list] = {}
# The smallest a figure can be in each document, scaled to its type size (set by
# `compare_chapters`): the same logo prints at 13pt in a document set in 5pt type
# and at 26pt in one set in 12pt, and one absolute minimum dropped it from the
# first while keeping it in the second.
_FIGURE_MIN_BY_DOC: dict[int, float] = {}


def _figure_min_side(doc: fitz.Document) -> float:
    return _FIGURE_MIN_BY_DOC.get(doc_key(doc), _FIGURE_MIN_SIDE)


_NEXT_STEP_OPENER_RE = re.compile(r"^\s*(?:\d{1,2}|[a-z])[.)]\s", re.IGNORECASE)
# pt: a short label reads about this wide at most. A run of body prose sitting
# above a row of icon+label pairs (each icon on the left, its own label in a
# column well to the right) can touch a figure's very edge at zero gap while
# running on for hundreds of points past it - the same zero gap a genuine
# label earns by sitting flush against the figure, but for an unrelated
# reason. Ranked down by how far a candidate's width runs past what a label
# ever needs, so the real label - a little farther away, but short - still
# wins over prose that merely happens to graze the figure first.
_CAPTION_LABEL_WIDTH = 170.0


def _caption(fig: Element, texts: list[Element]) -> str:
    """The label printed nearest a figure, on its own page, if any."""
    x0, y0, x1, y1 = fig.raw_bbox or fig.bbox
    best, best_distance = None, None
    fallback, fallback_distance = None, None
    for t in texts:
        if t.page != fig.page or not t.text or len(t.text) > _CAPTION_MAX_CHARS:
            continue
        if t.kind == KIND_HEADING:
            continue  # a heading names the section, not the picture under it
        tx0, ty0, tx1, ty1 = t.bbox
        dx = max(tx0 - x1, x0 - tx1, 0.0)
        dy = max(ty0 - y1, y0 - ty1, 0.0)
        gap = (dx * dx + dy * dy) ** 0.5
        if gap > _CAPTION_REACH:
            continue
        distance = gap + max(0.0, (tx1 - tx0) - _CAPTION_LABEL_WIDTH)
        # A numbered/lettered step opener ("3. On the Language page...")
        # printed just BELOW a figure is the instruction for the NEXT step,
        # not this one's label - a tutorial screenshot sits between its own
        # step's text above it and the next step's text below, and the gap
        # to that next step's opener is very often the smaller of the two.
        # Read as this figure's caption on raw distance alone, every
        # screenshot in the sequence was captioned one step late. Held back
        # to a fallback tier instead: still used if nothing else is in reach.
        if ty0 >= y1 - 0.5 and _NEXT_STEP_OPENER_RE.match(t.text):
            if fallback_distance is None or distance < fallback_distance:
                fallback, fallback_distance = t, distance
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = t, distance
    if best is not None:
        return best.text
    if fallback is not None:
        return fallback.text
    # A picture printed as one cell of a checklist/spec table (an "item name |
    # item picture" row) has no free-standing text beside it at all - its own
    # label lives in the other cell of that same row, inside the one table
    # Element the whole grid is read as.
    for t in texts:
        if t.kind != KIND_TABLE or not t.row_boxes:
            continue
        best_row, best_overlap = None, 0.0
        for (page, bbox), row_cells in zip(t.row_boxes, t.cells):
            if page != fig.page:
                continue
            # The row with the MOST of the figure's own height inside it, not
            # the first row touched at all: a figure's box is rarely cut at
            # exactly its row's boundary, and returning on the first overlap
            # found grabbed the row above's label from the one or two points
            # a figure grazed into its tail end - every icon in a stacked
            # checklist captioned one row early.
            overlap = min(y1, bbox[3]) - max(y0, bbox[1])
            if overlap > best_overlap:
                best_row, best_overlap = row_cells, overlap
        label = next((c for c in (best_row or ()) if (c or "").strip()), "")
        if label:
            return label[:_CAPTION_MAX_CHARS]
    # ... or the same row's item name set as ordinary text, with no ruling at
    # all to read as a table - a short, left-aligned label facing a picture
    # held in a fixed right-hand column sits far past an ordinary caption's
    # reach, but is still the only text this row has.
    row_label, row_gap = None, None
    for t in texts:
        if (t.page != fig.page or not t.text or len(t.text) > _CAPTION_MAX_CHARS
                or t.kind in (KIND_HEADING, KIND_TABLE, KIND_FIGURE)):
            continue
        tx0, ty0, tx1, ty1 = t.bbox
        if tx1 > x0 or min(y1, ty1) - max(y0, ty0) <= 0:
            continue  # only text set to the LEFT of the whole row, in its band
        gap = x0 - tx1
        if row_gap is None or gap < row_gap:
            row_label, row_gap = t, gap
    return row_label.text if row_label is not None else ""


def _labels_agree(a: str, b: str) -> bool | None:
    """True / False when both figures carry a label, None when either has none."""
    ka, kb = _normalise(a), _normalise(b)
    if not ka or not kb:
        return None
    return ka == kb or ka in kb or kb in ka


def _describe(fig: Element, caption: str) -> str:
    return f"the figure beside “{caption[:60]}”" if caption else f"a {fig.width:.0f}×{fig.height:.0f}pt figure"


def _figure_index(doc: fitz.Document) -> list[tuple]:
    """Every figure in the whole document: (page, bbox, fingerprint)."""
    key = doc_key(doc)
    if key not in _FIG_INDEX:
        items = []
        for page_index in range(doc.page_count):
            try:
                figures = get_figures(doc, page_index, print_ready=True)
            except Exception:
                continue
            for f in figures:
                if f.display_width < _figure_min_side(doc) or f.display_height < _figure_min_side(doc):
                    continue
                if _is_page_scale_content(doc, page_index, f):
                    continue
                raw = tuple(float(v) for v in f.bbox)
                bbox = _ink_bbox(doc, page_index, raw)
                items.append((page_index, bbox, imagefp.fingerprint(doc, page_index, bbox),
                              imagefp.fingerprint(doc, page_index, raw) if raw != bbox else None))
        _FIG_INDEX[key] = items
    return _FIG_INDEX[key]


def _find_elsewhere(fig: Element, doc: fitz.Document, own_pages: list[int]) -> Element | None:
    """The best-looking match for this figure anywhere in `doc` OUTSIDE this
    chapter's own pages, as an element that can be boxed and screenshotted."""
    if fig.fp is None:
        return None
    best, best_score = None, 0.0
    pages = set(own_pages)
    for page_index, bbox, fp, raw_fp in _figure_index(doc):
        if page_index in pages:
            continue
        score = _figure_similarity(fig, Element(kind=KIND_FIGURE, fp=fp, raw_fp=raw_fp))
        if score > best_score:
            best, best_score = (page_index, bbox, fp), score
    if best is None or best_score < _FIGURE_ELSEWHERE:
        return None
    page_index, bbox, fp = best
    return Element(kind=KIND_FIGURE, boxes=[(page_index, bbox)], width=bbox[2] - bbox[0],
                   height=bbox[3] - bbox[1], fp=fp)


_OVERSIZE_STEP = 0.35       # a figure this much larger in Staging is oversized (was 0.20, increased to reduce false positives) ...
_OVERSIZE_MIN_PT = 18.0     # ... and by at least this many points (was 12.0, increased for significance)
_ALIGN_CENTRE = 0.12        # centre within this share of the text column = centred
_ALIGN_SHIFT = 0.12         # and it must move at least this far to count
_MARGIN_CACHE: dict[int, tuple] = {}


def _text_column(doc: fitz.Document, page_index: int) -> tuple[float, float]:
    key = doc_key(doc)
    if key not in _MARGIN_CACHE:
        try:
            _MARGIN_CACHE[key] = _document_margins(doc)
        except Exception:
            _MARGIN_CACHE[key] = (None, None)
    left, right = _MARGIN_CACHE[key]
    rect = doc[page_index].rect
    if left is None or right is None or right - left < 50:
        return rect.x0, rect.x1
    return left, right


def _alignment(fig: Element, doc: fitz.Document) -> tuple[float, str]:
    left, right = _text_column(doc, fig.page)
    offset = ((fig.bbox[0] + fig.bbox[2]) / 2 - (left + right) / 2) / max(1.0, right - left)
    if abs(offset) <= _ALIGN_CENTRE:
        return offset, "centred"
    return offset, "left-aligned" if offset < 0 else "right-aligned"


_LABEL_MIN_CHARS = 3
_LABEL_WORDS: dict[int, object] = {}
# A callout number ( "1", "2" pointing at "Speakers", "Power LED indicator") is
# NOT included here even though it is a genuine label too: OCR reads a small
# numeral inside a coloured circle far less reliably than a word, and reading
# it as missing when it is printed on both sides (confirmed by eye) is worse
# than not checking it at all.


def _page_words(doc: fitz.Document):
    from pdfval import ocr
    key = doc_key(doc)
    if key not in _LABEL_WORDS:
        _LABEL_WORDS[key] = ocr.PageWords(doc)
    return _LABEL_WORDS[key]


def _label_keys(words: list[str]) -> set[str]:
    from pdfval import ocr
    keys = {ocr.normalize_word(w) for w in words}
    return {k for k in keys if len(k) >= _LABEL_MIN_CHARS and any(ch.isalpha() for ch in k)}


def figure_label_missing(a: Element, b: Element, name: str,
                         expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """Words printed ON the Production figure that the Staging figure does not
    carry. Staging very often bakes labels into the picture as pixels, so the
    Staging figure is read with OCR before a word is called missing - and when
    OCR cannot read it at all, nothing is asserted."""
    from pdfval import ocr
    exp_box = a.raw_bbox or a.bbox
    act_box = b.raw_bbox or b.bbox
    wanted = _label_keys(ocr.words_in(_page_words(expected).native_words(a.page), exp_box))
    if not wanted:
        return []
    have = _label_keys(ocr.words_in(_page_words(actual).native_words(b.page), act_box, margin=4))
    missing = wanted - have
    if not missing:
        return []
    read = ocr.words_in(_page_words(actual).ocr_words(b.page), act_box, margin=4)
    if not read:
        return []  # could not read the Staging figure - cannot tell dropped from rasterised
    missing -= _label_keys(read)
    if not missing:
        return []
    words = ", ".join(f"“{w}”" for w in sorted(missing)[:8])
    return [{"type": "figure-label-missing", "kind": KIND_FIGURE,
             "summary": f"Image label missing in Staging — {name}: {words} printed on the Production figure, not on the Staging one.",
             "detail": f"{len(missing)} label word(s) missing"}]


def visual_figure_changes(a: Element, b: Element, name: str,
                          expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """Spots inside a matched figure that render differently (see
    `visual_diff`), boxed on both sides - shown for review, never a failure."""
    # Icons are compared by `icon_changes`; callout icons are styling, never spots.
    if not a.boxes or not b.boxes or min(a.width, a.height, b.width, b.height) <= _ICON_MAX_SIDE:
        return []
    # The picture is under the same heading on both sides, which is all that
    # is asked of it (the user's rule): the two documents draw and scale their
    # artwork their own way, so spots that "look different" inside a picture
    # printed where it belongs are not reported. A picture that moved to
    # another heading still is - by `figure-elsewhere`.
    if a.section and a.section == b.section:
        return []
    found = visual_diff.differing_regions(expected, a.boxes[0], actual, b.boxes[0])
    if not found:
        return []
    count = len(found["act_marks"])
    return [{
        "type": "figure-visual", "kind": KIND_FIGURE,
        "summary": (f"Figure renders differently in Staging — {name} matches overall, but "
                    f"{count} spot{'s' if count > 1 else ''} inside it look different; check by eye."),
        "detail": "",
        "exp_marks": found["exp_marks"],
        "marks": found["act_marks"],
        "review_only": True,
    }]


_TOPIC_SAMPLES = 24            # heights per page band probed for which topic prints there
_WHOLE_COVERAGE = 0.6          # the smaller figure found filling this much of the larger: the same picture
_SUSPECT_AREA_RATIO = 4.0      # a pair this far apart in size may be a piece paired with a whole
_SUSPECT_SHAPE_GAP = 0.35
_SCREENSHOT_MIN_SIDE = 80.0    # pt
_SCREENSHOT_MIN_WORDS = 6
_SCREENSHOT_WORD_SHARE = 0.8
_FRAMING_TYPES = {"figure-different", "figure-size", "figure-alignment", "figure-visual"}


_TOPIC_REGION_PAD = 260.0     # pt: added around a suspiciously thin sampled slice
_TOPIC_REGION_THIN = 120.0    # pt: a topic sampled to less height than this may be a boundary miss,
                              # not a genuinely short topic - most figures alone stand taller than this


def _topic_regions(doc: fitz.Document, bands: list | None, section: str,
                   section_at) -> list[tuple[int, "fitz.Rect"]]:
    """The stretches of this document's chapter printed under `section`.

    Sampled at `_TOPIC_SAMPLES` points per page band, which can miss a picture
    that sits just past where the last sample happened to land - the two
    documents' headings rarely fall at exactly the same point on the page, so
    a topic's true bottom edge is somewhere between two samples, not AT one of
    them. A slice thinner than a real figure usually stands (`_TOPIC_REGION_THIN`)
    is padded outward before it is trusted as the whole of a short topic -
    the caller still confirms whatever is found there is a pixel match AND
    lands back in this same section, so padding only widens where to look,
    never what counts as found.
    """
    if not section or section_at is None or not bands:
        return []
    out: list[tuple[int, fitz.Rect]] = []
    for page_index, top, bottom in bands:
        step = (bottom - top) / (_TOPIC_SAMPLES - 1) if bottom > top else 0.0
        hits = [top + n * step for n in range(_TOPIC_SAMPLES) if section_at(page_index, top + n * step) == section]
        if hits:
            page = doc[page_index].rect
            y0, y1 = hits[0] - step, hits[-1] + step
            if y1 - y0 < _TOPIC_REGION_THIN:
                y0, y1 = y0 - _TOPIC_REGION_PAD, y1 + _TOPIC_REGION_PAD
            out.append((page_index, fitz.Rect(page.x0, max(top, y0), page.x1, min(bottom, y1))))
    return out


def _words_in(doc: fitz.Document, regions: list[tuple[int, "fitz.Rect"]], with_ocr: bool) -> set[str]:
    from pdfval import ocr

    out: set[str] = set()
    for page_index, rect in regions:
        def inside(w) -> bool:
            return rect.contains(fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
        out |= {ocr.normalize_word(w[4]) for w in doc[page_index].get_text("words") if inside(w)}
        if with_ocr and ocr.available():
            out |= {ocr.normalize_word(w[4]) for w in _page_words(doc).ocr_words(page_index) if inside(w)}
    return {w for w in out if len(w) >= 3 and not w.isdigit()}


def _screenshot_words_printed(fig: Element, own_doc: fitz.Document, other_doc: fitz.Document,
                              regions: list[tuple[int, "fitz.Rect"]]) -> bool:
    """A screenshot one side embeds as an image and the other draws as vector
    text: pixel matching cannot tell two menus of one UI apart, but its words
    can - nearly all of them printed in the other side's copy of the topic."""
    if min(fig.width, fig.height) < _SCREENSHOT_MIN_SIDE or not fig.boxes:
        return False
    page_index, bbox = fig.boxes[0]
    own = [(page_index, fitz.Rect(bbox))]
    words = _words_in(own_doc, own, with_ocr=False)
    if len(words) < _SCREENSHOT_MIN_WORDS:
        words = _words_in(own_doc, own, with_ocr=True)
    if len(words) < _SCREENSHOT_MIN_WORDS:
        return False
    need = _SCREENSHOT_WORD_SHARE * len(words)
    return (len(words & _words_in(other_doc, regions, with_ocr=False)) >= need
            or len(words & _words_in(other_doc, regions, with_ocr=True)) >= need)


def _seen_in_topic(fig: Element, own_doc: fitz.Document, other_doc: fitz.Document,
                   other_bands: list | None, other_section_at) -> Element | None:
    """This figure printed in the other document's copy of its own topic, found
    on the rendered pages (see `visual_diff.find_artwork`): one piece of a
    bigger picture there, drawn where this side embeds it, resized - or a
    screenshot whose words the other side prints."""
    if not fig.boxes:
        return None
    regions = _topic_regions(other_doc, other_bands, fig.section, other_section_at)
    if not regions:
        return None
    page_index, bbox = fig.boxes[0]
    # No section gate on the match itself: the two documents' headings rarely
    # sit at exactly the same relative position, so the true picture can fall
    # just the other side of where THIS side's own topic boundary lands
    # (confirmed on a real manual: the same illustration, same pose, one
    # heading later). Safe without it because the SEARCH is already narrow -
    # this topic's own band, padded a little past a thin slice - not the
    # whole chapter or document, and the pixel/edge verification below is
    # what actually decides "the same picture", the same bar `_printed_elsewhere`
    # holds a whole-document search to.
    hit = visual_diff.find_artwork(own_doc, page_index, bbox, other_doc, regions)
    if hit is None and _screenshot_words_printed(fig, own_doc, other_doc, regions):
        page, rect = regions[0]
        hit = (page, (rect.x0, rect.y0, rect.x1, rect.y1))
    if hit is None:
        return None
    x0, y0, x1, y1 = hit[1]
    return Element(kind=KIND_FIGURE, boxes=[hit], width=x1 - x0, height=y1 - y0, section=fig.section)


def _pair_framing(a: Element, b: Element, expected: fitz.Document, actual: fitz.Document) -> str | None:
    """"whole" when the smaller of two paired figures is found filling the
    larger, "piece" when it is found as only part of it, None otherwise."""
    if not a.boxes or not b.boxes:
        return None
    if a.width * a.height <= b.width * b.height:
        small, small_doc, large, large_doc = a, expected, b, actual
    else:
        small, small_doc, large, large_doc = b, actual, a, expected
    large_page, large_box = large.boxes[0]
    hit = visual_diff.find_artwork(small_doc, small.boxes[0][0], small.boxes[0][1], large_doc,
                                   [(large_page, fitz.Rect(large_box) + (-4, -4, 4, 4))])
    if hit is None:
        return None
    x0, y0, x1, y1 = hit[1]
    return "whole" if (x1 - x0) * (y1 - y0) >= _WHOLE_COVERAGE * large.width * large.height else "piece"


def _settle_pair(diffs: list[dict], a: Element, b: Element,
                 expected: fitz.Document, actual: fitz.Document,
                 exp_bands: list | None, act_bands: list | None,
                 exp_section_at, act_section_at) -> list[dict]:
    """A paired figure's picture, size and alignment findings, checked against
    the rendered pages: the same picture drawn on one side and embedded on the
    other is not "a different picture", and a piece paired with the whole
    picture it belongs to has no size or alignment of its own to compare."""
    if not any(d.get("type") in _FRAMING_TYPES for d in diffs):
        return diffs
    framing = _pair_framing(a, b, expected, actual)
    if framing == "whole":
        return [d for d in diffs if d.get("type") != "figure-different"]
    if framing == "piece":
        return [d for d in diffs if d.get("type") not in _FRAMING_TYPES]
    area_a, area_b = a.width * a.height, b.width * b.height
    shape_a, shape_b = a.width / max(1.0, a.height), b.width / max(1.0, b.height)
    suspect = (max(area_a, area_b) / max(1.0, min(area_a, area_b)) >= _SUSPECT_AREA_RATIO
               or abs(shape_a - shape_b) / max(1e-6, min(shape_a, shape_b)) > _SUSPECT_SHAPE_GAP)
    if suspect:
        seen = (_seen_in_topic(a, expected, actual, act_bands, act_section_at) if area_a <= area_b
                else _seen_in_topic(b, actual, expected, exp_bands, exp_section_at))
        if seen is not None:
            return [d for d in diffs if d.get("type") not in _FRAMING_TYPES]
    return diffs


def _stands_alone(fig: Element, others: list[Element] | None) -> bool:
    """Nothing printed beside the figure on its line - no label, no second
    picture. A figure in a picture-and-label grid has no meaningful alignment
    of its own."""
    for o in others or []:
        if o is fig or o.page != fig.page:
            continue
        overlap = min(fig.bbox[3], o.bbox[3]) - max(fig.bbox[1], o.bbox[1])
        if overlap > 0.3 * min(fig.bbox[3] - fig.bbox[1], o.bbox[3] - o.bbox[1]) and (
            o.bbox[2] <= fig.bbox[0] + 2 or o.bbox[0] >= fig.bbox[2] - 2
        ):
            return False
    return True


def _figure_layout(a: Element, b: Element, name: str,
                   expected: fitz.Document, actual: fitz.Document,
                   exp_others: list[Element] | None = None,
                   act_others: list[Element] | None = None,
                   scale: float = 1.0) -> list[dict]:
    """Oversized in Staging, or aligned differently - the two layout problems a
    paired figure can have. A figure drawn SMALLER is not reported."""
    out: list[dict] = []
    # Measured against Production's size scaled to Staging's type: a document
    # set in 12pt against one set in 5pt prints every logo bigger, by design.
    grew = [axis for axis, va, vb in (("width", a.width, b.width), ("height", a.height, b.height))
            if va and (vb - va * scale) / (va * scale) >= _OVERSIZE_STEP and vb - va * scale >= _OVERSIZE_MIN_PT]
    if grew:
        out.append({"type": "figure-size", "kind": KIND_FIGURE,
                    "summary": (f"Figure oversized in Staging — {name}: {a.width:.0f}×{a.height:.0f}pt in "
                                f"Production, {b.width:.0f}×{b.height:.0f}pt in Staging."),
                    "detail": " and ".join(grew) + " larger"})
    if not (_stands_alone(a, exp_others) and _stands_alone(b, act_others)):
        return out
    off_a, cls_a = _alignment(a, expected)
    off_b, cls_b = _alignment(b, actual)
    if cls_a != cls_b and abs(off_a - off_b) >= _ALIGN_SHIFT and "centred" in (cls_a, cls_b):
        out.append({"type": "figure-alignment", "kind": KIND_FIGURE,
                    "summary": f"Figure alignment changed — {name} is {cls_a} in Production and {cls_b} in Staging.",
                    "detail": ""})
    return out


_ART_MIN_SIDE = 6.0     # points: smaller than this is a glyph, not artwork
_ART_SHAPE_GAP = 0.35   # a counterpart's width/height ratio may differ by this share


def _chapter_bands(doc: fitz.Document, pages: list[int], span) -> list[tuple[int, float, float]]:
    """(page, top, bottom) for each stretch of page the chapter occupies."""
    (start_page, start_y), end = span
    bands = []
    for page_index in pages:
        rect = doc[page_index].rect
        top = start_y - 2 if page_index == start_page else rect.y0
        bottom = end[1] + 2 if (end and page_index == end[0]) else rect.y1
        bands.append((page_index, top, bottom))
    return bands


def _artwork_boxes(doc: fitz.Document, page_index: int, top: float, bottom: float) -> list["fitz.Rect"]:
    """Everything picture-like printed in a band of the page, at any size above a
    glyph: embedded images, and clusters of vector drawing."""
    page = doc[page_index]
    boxes: list[fitz.Rect] = []
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"])
            if top <= (r.y0 + r.y1) / 2 <= bottom and min(r.width, r.height) >= _ART_MIN_SIDE:
                boxes.append(r)
    except Exception:
        pass
    try:
        drawn = [fitz.Rect(d["rect"]) for d in page.get_drawings()
                 if top <= (d["rect"].y0 + d["rect"].y1) / 2 <= bottom
                 and d["rect"].width < page.rect.width * 0.6]  # not a rule or a table border
    except Exception:
        drawn = []
    clusters: list[fitz.Rect] = []
    for r in drawn:
        clusters.append(fitz.Rect(r))
        merged = True
        while merged:  # grow the newest cluster through everything it touches
            merged = False
            for i in range(len(clusters) - 1):
                if (clusters[i] + (-2, -2, 2, 2)).intersects(clusters[-1]):
                    clusters[-1] |= clusters.pop(i)
                    merged = True
                    break
    boxes += [c for c in clusters if min(c.width, c.height) >= _ART_MIN_SIDE]
    return boxes


def _printed_in_band(fig: Element, doc: fitz.Document, bands: list[tuple[int, float, float]],
                     taken: list[tuple[int, tuple]], section_at=None) -> Element | None:
    """This picture printed in the other document's copy of the chapter - at
    whatever size, embedded or drawn as vectors - or None.

    The figure detector keeps to pictures of a useful size, and a document set
    in small type prints its logos under it: Production's TCO logo is a 22pt
    vector drawing, Staging's the same logo as a 45pt image. Without a second
    look, every such logo was "only in Staging". A place already taken by a
    matched figure is not offered again."""
    if fig.fp is None or not fig.height:
        return None
    shape = fig.width / fig.height
    best, best_score = None, 0.0
    for page_index, top, bottom in bands:
        for r in _artwork_boxes(doc, page_index, top, bottom):
            if abs(r.width / max(1.0, r.height) - shape) / shape > _ART_SHAPE_GAP:
                continue
            if any(p == page_index and _rect_area(fitz.Rect(t) & r) >= 0.5 * _rect_area(r) for p, t in taken):
                continue
            if section_at is not None and fig.section and section_at(page_index, r.y0) != fig.section:
                continue  # printed under another topic: not this figure's counterpart
            try:
                fp = imagefp.fingerprint(doc, page_index, tuple(r))
            except Exception:
                continue
            score = _figure_similarity(fig, Element(kind=KIND_FIGURE, fp=fp))
            if score > best_score:
                best, best_score = (page_index, r, fp), score
    if best is None or best_score < _FIGURE_MATCH:
        return None
    page_index, r, fp = best
    return Element(kind=KIND_FIGURE, boxes=[(page_index, tuple(r))], width=r.width, height=r.height, fp=fp)


_PAGE_ART_CACHE: dict[tuple, list] = {}
_ART_FP_CACHE: dict[tuple, object] = {}
_ELSEWHERE_AREA_RATIO = 4.0  # a counterpart elsewhere this much larger or smaller is other artwork


def _page_artwork_boxes(doc: fitz.Document, page_index: int) -> list["fitz.Rect"]:
    key = (doc_key(doc), page_index)
    if key not in _PAGE_ART_CACHE:
        rect = doc[page_index].rect
        _PAGE_ART_CACHE[key] = _artwork_boxes(doc, page_index, rect.y0, rect.y1)
    return _PAGE_ART_CACHE[key]


def _cached_fingerprint(doc: fitz.Document, page_index: int, bbox: tuple):
    # A whole-document crawl checks the same page's candidates against every
    # still-unmatched figure - fingerprinted once here, not once per figure.
    key = (doc_key(doc), page_index, tuple(round(v, 1) for v in bbox))
    if key not in _ART_FP_CACHE:
        try:
            _ART_FP_CACHE[key] = imagefp.fingerprint(doc, page_index, bbox)
        except Exception:
            _ART_FP_CACHE[key] = None
    return _ART_FP_CACHE[key]


def _printed_elsewhere(fig: Element, doc: fitz.Document, taken: list[tuple[int, tuple]]) -> Element | None:
    """This picture printed anywhere else in the document - under a different
    TOC heading, not just this chapter's own copy of it, and not just this
    chapter's own narrower topic band (a chapter can nest a whole OTHER
    heading - "How to detach the stand" inside "How to assemble..." - whose
    pages the topic-band check above never looks at) - so a figure is only
    "missing"/"added" once the whole document has been checked. Held to a
    near-certain match: crawling the whole document at the same bar as within
    a chapter would pair unrelated look-alike figures from two different
    topics that merely share a shape.

    Never attempted for an icon-sized figure. A directional arrow ("Move to
    the right", "Move up") beside a 5-way controller table is simple enough,
    and printed often enough throughout a manual, that a DIFFERENT arrow
    pointing a different way in a totally different topic still scores past
    the near-certain bar - confirmed: a single such icon matched five
    unrelated topics across the document in one run. A figure this small is
    exactly the shape class the near-certain bar cannot actually tell apart;
    only the same-topic, position-scoped check above is safe for it."""
    if fig.fp is None or not fig.height or min(fig.width, fig.height) <= _ICON_MAX_SIDE:
        return None
    shape = fig.width / fig.height
    best, best_score = None, 0.0
    area = fig.width * fig.height
    for page_index in range(doc.page_count):
        for r in _page_artwork_boxes(doc, page_index):
            if abs(r.width / max(1.0, r.height) - shape) / shape > _ART_SHAPE_GAP:
                continue
            # An icon-sized box, or one far off this figure's size, is another
            # kind of artwork that only happens to share a shape.
            r_area = _rect_area(r)
            if min(r.width, r.height) <= _ICON_MAX_SIDE or \
                    max(area, r_area) / max(1.0, min(area, r_area)) > _ELSEWHERE_AREA_RATIO:
                continue
            if any(p == page_index and _rect_area(fitz.Rect(t) & r) >= 0.5 * r_area for p, t in taken):
                continue
            fp = _cached_fingerprint(doc, page_index, tuple(r))
            if fp is None:
                continue
            score = _figure_similarity(fig, Element(kind=KIND_FIGURE, fp=fp))
            if score > best_score:
                best, best_score = (page_index, r, fp), score
    if best is None or best_score < _FIGURE_SAME:
        return None
    page_index, r, fp = best
    return Element(kind=KIND_FIGURE, boxes=[(page_index, tuple(r))], width=r.width, height=r.height, fp=fp)


def _figure_repeats_elsewhere(fig: Element, others: list[Element]) -> bool:
    """Whether this same picture is ALSO used, at this same near-identical
    fingerprint, somewhere else in the same document - a note icon or a menu
    screenshot deliberately reused once per item. Only figures that do NOT
    repeat are one-off illustrations, so a one-off pairing across two
    different sections is a real misplacement, not a reused icon landing
    beside whichever item it happens to sit closest to."""
    return any(o is not fig and _figure_similarity(fig, o) >= _FIGURE_SAME for o in others)


def _wrong_section_diff(a: Element, b: Element, name: str,
                        exp_entries: list["TocEntry"], act_entries: list["TocEntry"],
                        exp_figs: list[Element], act_figs: list[Element]) -> dict | None:
    """A figure paired across two different sections is normally a picture
    reused per topic (allowed through on looks alone) - but when it is a
    one-off on BOTH sides, landing under a different heading in Staging than
    Production had it under is a genuine misplacement, not a reuse."""
    if not (a.section and b.section and a.section != b.section):
        return None
    if _figure_repeats_elsewhere(a, exp_figs) or _figure_repeats_elsewhere(b, act_figs):
        return None
    where_exp = heading_at(exp_entries, a.page, a.bbox[1]) or a.section
    where_act = heading_at(act_entries, b.page, b.bbox[1]) or b.section
    return {"type": "figure-wrong-section", "kind": KIND_FIGURE,
            "summary": (f"Figure wrongly placed in Staging — {name} is printed under “{where_exp}” "
                        f"in Production, but under “{where_act}” in Staging."),
            "detail": f"Production: {where_exp} · Staging: {where_act}"}


def figure_changes(
    exp_figs: list[Element], act_figs: list[Element],
    exp_texts: list[Element], act_texts: list[Element],
    expected: fitz.Document, actual: fitz.Document,
    exp_entries: list[TocEntry], act_entries: list[TocEntry],
    exp_pages: list[int], act_pages: list[int],
    exp_bands: list | None = None, act_bands: list | None = None,
    scale: float = 1.0,
    exp_section_at=None, act_section_at=None,
) -> list[dict]:
    """Rows for every figure in the chapter: `{exp, act, differences}`."""
    exp_caps = [_caption(f, exp_texts) for f in exp_figs]
    act_caps = [_caption(f, act_texts) for f in act_figs]

    used_exp: set[int] = set()
    used_act: set[int] = set()
    rows: list[dict] = []

    _COMPOSITE_WIDTH_RATIO = 1.8  # a same-row figure at least this much wider holds several icons, not one

    def _inside_wider_figure(fig: Element, others: list[Element]) -> int | None:
        """Index in `others` of a much wider picture on the other side that this
        whole figure sits inside - Staging embeds a strip of icons
        (temperature/humidity/altitude, power states) as one combined image
        where Production draws each as its own figure. Not missing, added, or
        size-mismatched: it is inside that one image, boxed against it rather
        than compared to it figure-for-figure. Run before any similarity/size
        matching so a small icon is never scored (and reported oversized/
        undersized) against the composite it merely sits inside.
        """
        for k, g in enumerate(others):
            if g.page != fig.page or g.width < fig.width * _COMPOSITE_WIDTH_RATIO:
                continue
            if fig.section and g.section and fig.section != g.section:
                continue
            gx0, gy0, gx1, gy1 = g.bbox
            fx0, fy0, fx1, fy1 = fig.bbox
            if gx0 - 2 <= fx0 and fx1 <= gx1 + 2 and gy0 - 4 <= fy0 and fy1 <= gy1 + 4:
                return k
        return None

    for i, a in enumerate(exp_figs):
        k = _inside_wider_figure(a, act_figs)
        if k is not None:
            used_exp.add(i)
            used_act.add(k)
            rows.append({"exp": [a], "act": [act_figs[k]], "differences": []})
    for j, b in enumerate(act_figs):
        if j in used_act:
            continue
        k = _inside_wider_figure(b, exp_figs)
        if k is not None:
            used_exp.add(k)
            used_act.add(j)
            rows.append({"exp": [exp_figs[k]], "act": [b], "differences": []})

    candidates = []
    for i, a in enumerate(exp_figs):
        if i in used_exp:
            continue
        for j, b in enumerate(act_figs):
            if j in used_act:
                continue
            look = _figure_similarity(a, b)
            # A figure is only ever the same figure within its own topic -
            # unless it looks near-certainly identical: a screenshot of the
            # same on-screen menu, deliberately repeated once per menu item,
            # reflows to sit closest to a different item's heading on the
            # other side - still the same picture, not a different one.
            if a.section and b.section and a.section != b.section and look < _FIGURE_SAME:
                continue
            labels = _labels_agree(exp_caps[i], act_caps[j])
            if not (look >= _FIGURE_MATCH or (labels and look >= _FIGURE_MATCH_WITH_LABEL)):
                continue
            if labels is False and look < _FIGURE_SAME_DESPITE_LABEL:
                continue  # different captions - a numbered step's own screenshot,
                          # not the same picture merely repeated with a new label
            candidates.append((look + (_LABEL_BONUS if labels else 0.0), i, j, look, labels))
    candidates.sort(reverse=True)
    for _, i, j, look, labels in candidates:
        if i in used_exp or j in used_act:
            continue
        used_exp.add(i)
        used_act.add(j)
        a, b = exp_figs[i], act_figs[j]
        diffs: list[dict] = []
        name = _describe(a, exp_caps[i] or act_caps[j])
        if look < _FIGURE_DIFFERENT:
            diffs.append({"type": "figure-different", "kind": KIND_FIGURE,
                          "summary": f"Figure is a different picture — {name} is printed in both documents, but the pictures do not match.",
                          "detail": f"appearance match {look:.0%}"})
        diffs.extend(_figure_layout(a, b, name, expected, actual, exp_texts + exp_figs, act_texts + act_figs, scale=scale))
        if look >= _FIGURE_DIFFERENT:
            # A drawing against an embedded image differs in anti-aliasing
            # specks, not ink, so this used to be skipped whenever `a.drawn !=
            # b.drawn` - but that silently dropped every diagram Staging bakes
            # to a raster while Production keeps as live vector art (a very
            # common real case), leaving genuine missing labels/callouts
            # completely unchecked instead of just noisy. Both callees already
            # guard against exactly the noise this was meant to avoid:
            # `figure_label_missing` only asserts a word missing after OCR
            # itself fails to read it off the rendered picture, and
            # `differing_regions` bails out (returns None) when the two
            # renders cannot be aligned or when more than `MAX_CHANGED_SHARE`
            # of the crop differs - a rendering-style mismatch too large to
            # trust shows nothing, exactly as before, but a real, compact,
            # local difference is no longer thrown away just because one side
            # rasterised the artwork.
            diffs.extend(figure_label_missing(a, b, name, expected, actual))
            diffs.extend(visual_figure_changes(a, b, name, expected, actual))
        diffs = _settle_pair(diffs, a, b, expected, actual, exp_bands, act_bands, exp_section_at, act_section_at)
        wrong_section = _wrong_section_diff(a, b, name, exp_entries, act_entries, exp_figs, act_figs)
        if wrong_section:
            diffs.append(wrong_section)
        rows.append({"exp": [a], "act": [b], "differences": diffs})

    # Second chance: figures still without a partner on both sides pair on how
    # they look alone. A label beside a figure is good evidence when it agrees,
    # but two documents often label the same drawing differently - a heading
    # above it on one side, an "(a) Display (b) Wall" legend on the other.
    mixed_pairs: list[tuple[Element, Element]] = []
    leftovers = sorted(
        (
            (look, i, j)
            for i in range(len(exp_figs)) if i not in used_exp
            for j in range(len(act_figs)) if j not in used_act
            for look in [_figure_similarity(exp_figs[i], act_figs[j])]
            if look >= _FIGURE_SAME or not (
                exp_figs[i].section and act_figs[j].section and exp_figs[i].section != act_figs[j].section
            )
        ),
        reverse=True,
    )
    for look, i, j in leftovers:
        # A figure one document DRAWS and the other EMBEDS does not fingerprint
        # alike even when it is the same symbol (Production's vector WEEE and
        # battery marks against Staging's images: 0.36) - pair those, still
        # within the topic, on a lower bar, and judge only their size and place.
        mixed = exp_figs[i].drawn != act_figs[j].drawn
        if look < (_FIGURE_MATCH_DRAWN if mixed else _FIGURE_MATCH) or i in used_exp or j in used_act:
            continue
        used_exp.add(i)
        used_act.add(j)
        a, b = exp_figs[i], act_figs[j]
        name = _describe(a, exp_caps[i] or act_caps[j])
        diffs = list(_figure_layout(a, b, name, expected, actual, exp_texts + exp_figs, act_texts + act_figs, scale=scale))
        if mixed:
            mixed_pairs.append((a, b))
            # Same reasoning as the main candidates loop above: a drawn-vs-
            # embedded pair used to stop at size/alignment only, but
            # `figure_label_missing`/`differing_regions` already guard their
            # own noise (OCR-confirmed-missing only, and a bail-out past
            # `MAX_CHANGED_SHARE`), so skipping them here just hid real
            # content differences for exactly the pairs most likely to have
            # them - a diagram redrawn in a different tool.
            diffs.extend(figure_label_missing(a, b, name, expected, actual))
            diffs.extend(visual_figure_changes(a, b, name, expected, actual))
            diffs = _settle_pair(diffs, a, b, expected, actual, exp_bands, act_bands, exp_section_at, act_section_at)
            wrong_section = _wrong_section_diff(a, b, name, exp_entries, act_entries, exp_figs, act_figs)
            if wrong_section:
                diffs.append(wrong_section)
            rows.append({"exp": [a], "act": [b], "differences": diffs})
            continue
        if look < _FIGURE_DIFFERENT:
            diffs.append({"type": "figure-different", "kind": KIND_FIGURE,
                          "summary": f"Figure is a different picture — {name} is labelled differently in the two documents and the pictures do not match.",
                          "detail": f"appearance match {look:.0%} · Production label: “{exp_caps[i] or '—'}” · Staging label: “{act_caps[j] or '—'}”"})
        diffs = _settle_pair(diffs, a, b, expected, actual, exp_bands, act_bands, exp_section_at, act_section_at)
        wrong_section = _wrong_section_diff(a, b, name, exp_entries, act_entries, exp_figs, act_figs)
        if wrong_section:
            diffs.append(wrong_section)
        rows.append({"exp": [a], "act": [b], "differences": diffs})

    # Last resort: image validation only asks whether a picture is there, not
    # what it looks like, so appearance is no longer the gate once a topic's
    # own figures are down to their last, unmatched few - a diagram Production
    # DRAWS as vectors and Staging EMBEDS as one big screenshot, resized well
    # past what the appearance/shape checks above tolerate, still fingerprints
    # nothing alike, but it is still a picture in that topic's slot. Paired in
    # reading order within the shared topic, not each called missing on one
    # side and added on the other - and only when neither side is wildly
    # bigger than the other, so an icon is never quietly credited as standing
    # in for a full illustration.
    _LAST_RESORT_AREA_RATIO = 4.0
    remaining_act_by_section: dict[str, list[int]] = {}
    for j in range(len(act_figs)):
        if j not in used_act:
            remaining_act_by_section.setdefault(act_figs[j].section, []).append(j)
    for group in remaining_act_by_section.values():
        group.sort(key=lambda j: (act_figs[j].page, act_figs[j].bbox[1]))
    for i in sorted((i for i in range(len(exp_figs)) if i not in used_exp),
                     key=lambda i: (exp_figs[i].page, exp_figs[i].bbox[1])):
        a = exp_figs[i]
        if not a.section:
            continue
        candidates = remaining_act_by_section.get(a.section) or []
        area_a = a.width * a.height
        if area_a <= 0 or not candidates:
            continue
        j = next((j for j in candidates
                  if act_figs[j].width * act_figs[j].height > 0
                  and max(area_a, act_figs[j].width * act_figs[j].height)
                      / min(area_a, act_figs[j].width * act_figs[j].height) <= _LAST_RESORT_AREA_RATIO), None)
        if j is None:
            continue
        candidates.remove(j)
        used_exp.add(i)
        used_act.add(j)
        rows.append({"exp": [a], "act": [act_figs[j]], "differences": []})

    taken_exp = [(exp_figs[i].page, exp_figs[i].bbox) for i in used_exp]
    taken_act = [(act_figs[j].page, act_figs[j].bbox) for j in used_act]

    def beside_mixed_partner(fig: Element, side: str) -> bool:
        """One drawing on one side, several images side by side on the other:
        Production draws its WEEE bin and battery marks 4pt apart, so they are
        one vector cluster, where Staging embeds each as its own image. The
        image beside the one already paired with that drawing is the same
        artwork, not an extra figure."""
        for a, b in mixed_pairs:
            partner = b if side == "act" else a
            if partner.page != fig.page:
                continue
            same_row = abs(partner.bbox[1] - fig.bbox[1]) <= _SIDE_BY_SIDE_Y
            gap = max(partner.bbox[0], fig.bbox[0]) - min(partner.bbox[2], fig.bbox[2])
            if same_row and gap <= _SIDE_BY_SIDE_GAP:
                return True
        return False

    def caption_anchor(caption: str, other_texts: list[Element], section: str) -> tuple | None:
        """Where this figure's own caption/label text prints on the OTHER side,
        even though the picture itself does not - a reader clicking "figure
        missing/added" lands beside the words the picture illustrated, not at
        the top of the whole heading it happens to fall under."""
        if not caption or len(caption) < 8:
            return None
        phrase = _normalise(caption)
        best, best_score = None, 0.55
        for el in other_texts:
            if not el.key or (section and el.section and el.section != section):
                continue
            score = _ratio(phrase, el.key)
            if score > best_score:
                best, best_score = el, score
        if not best or not best.boxes:
            return None
        page_index, bbox = best.boxes[0]
        return (page_index, bbox[1])

    def proportional_anchor(page: int, own_pages: list[int], other_pages: list[int]) -> tuple | None:
        """Last resort, when there is no caption to place a figure by at all -
        a bare logo, a decorative graphic: roughly where this page's position
        IN ITS OWN chapter falls on the other side. The two documents paginate
        differently, but a figure a third of the way through one side's copy
        of a chapter is still roughly a third of the way through the
        other's - closer than the chapter's own top, which is where a figure
        on page 1 of a chapter with no matched topic (front matter, before any
        heading) otherwise lands."""
        if not own_pages or not other_pages or page not in own_pages:
            return None
        frac = own_pages.index(page) / max(1, len(own_pages) - 1)
        return (other_pages[round(frac * (len(other_pages) - 1))], 0.0)

    for i, a in enumerate(exp_figs):
        if i in used_exp or beside_mixed_partner(a, "exp"):
            continue
        if _note_badges(expected, a):
            continue  # a column of note icons, not a picture - see `_note_badges`
        if _piece_of_matched(a, exp_figs, used_exp):
            continue  # a slice of a picture already matched whole on the other side
        name = _describe(a, exp_caps[i])
        here = _printed_in_band(a, actual, act_bands, taken_act, act_section_at) if act_bands else None
        if here is None:
            here = _seen_in_topic(a, expected, actual, act_bands, act_section_at)
        if here is not None:
            rows.append({"exp": [a], "act": [here], "differences": []})
            continue
        found = _printed_elsewhere(a, actual, taken_act)
        if found is not None:
            where = heading_at(act_entries, found.page, found.bbox[1]) or "another section"
            diff = {"type": "figure-elsewhere", "kind": KIND_FIGURE, "exp": a, "act": found,
                    "summary": f"Figure is in a different section in Staging — {name} is here in Production, but Staging prints it under “{where}” (p.{found.page + 1}).",
                    "detail": ""}
        else:
            diff = {"type": "figure-missing", "kind": KIND_FIGURE, "exp": a, "act": None,
                    "summary": f"Figure missing in Staging — {name} is not printed anywhere in Staging.",
                    "detail": f"{a.width:.0f}×{a.height:.0f}pt"}
            anchor = caption_anchor(exp_caps[i], act_texts, a.section) or proportional_anchor(a.page, exp_pages, act_pages)
            if anchor:
                diff["act_anchor"] = anchor
        rows.append({"exp": [a], "act": [], "differences": [diff]})

    for j, b in enumerate(act_figs):
        if j in used_act or beside_mixed_partner(b, "act"):
            continue
        name = _describe(b, act_caps[j])
        here = _printed_in_band(b, expected, exp_bands, taken_exp, exp_section_at) if exp_bands else None
        if here is None:
            here = _seen_in_topic(b, actual, expected, exp_bands, exp_section_at)
        if here is not None:
            rows.append({"exp": [here], "act": [b], "differences": []})
            continue
        found = _printed_elsewhere(b, expected, taken_exp)
        if found is not None:
            where = heading_at(exp_entries, found.page, found.bbox[1]) or "another section"
            diff = {"type": "figure-elsewhere", "kind": KIND_FIGURE, "exp": found, "act": b,
                    "summary": f"Figure is in a different section in Production — Staging prints {name} here, but Production has it under “{where}” (p.{found.page + 1}).",
                    "detail": ""}
        else:
            diff = {"type": "figure-added", "kind": KIND_FIGURE, "exp": None, "act": b,
                    "summary": f"Figure only in Staging — {name} is not printed anywhere in Production.",
                    "detail": f"{b.width:.0f}×{b.height:.0f}pt"}
            anchor = caption_anchor(act_caps[j], exp_texts, b.section) or proportional_anchor(b.page, act_pages, exp_pages)
            if anchor:
                diff["exp_anchor"] = anchor
        rows.append({"exp": [], "act": [b], "differences": [diff]})
    return rows


# --- the same content, laid out differently --------------------------------

_LAYOUT_MIN_CHARS = 4
_LAYOUT_SHORT_CHARS = 20          # below this, only a container that STARTS with it counts
# The last-resort excuse when no verbatim phrase match was found anywhere -
# a bag-of-words share, not a real match on meaning - so it is asked to be
# almost total: two documents that reworded a passage while reusing most of
# its vocabulary must still be told apart from one that only moved it.
_LAYOUT_COVERAGE = 0.97           # share of a long element's words present on the other side
_LAYOUT_COVERAGE_MIN_TOKENS = 12


def _flat(text: str) -> str:
    return _WS_RE.sub(" ", _normalise(text).replace("|", " ")).strip()


def _has_phrase(haystack: str, needle: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack))


def _present_elsewhere(el: Element, others: list[Element], flats: list[str],
                       blob: str, tokens: "Counter[str]") -> tuple[bool, Element | None]:
    """Is what this one-sided element says printed on the other side anyway?

    `(found, container)`: `container` is the single element that carries it
    when there is one (a table the rows were read into, a paragraph the heading
    was merged into), None when the words are there but spread over several.

    Headings and short text are held to a strict rule - the other side must
    have an element that STARTS with them, or a table - because a deleted
    "Settings" heading must not be excused just because the word "settings"
    appears in some paragraph. Longer text may be found anywhere, as a phrase
    or, failing that, as nearly all of its words.
    """
    if el.kind == KIND_FIGURE or not el.text:
        return False, None
    needle = _flat(el.text)
    if len(needle) < _LAYOUT_MIN_CHARS:
        return False, None
    strict = el.kind == KIND_HEADING or len(needle) < _LAYOUT_SHORT_CHARS
    for other, flat in zip(others, flats):
        if not flat:
            continue
        if strict:
            if flat.startswith(needle) or (other.kind == KIND_TABLE and _has_phrase(flat, needle)):
                return True, other
        elif _has_phrase(flat, needle):
            return True, other
    if strict:
        return False, None
    if _has_phrase(blob, needle):
        return True, None
    wanted = _tokens(el.key)
    total = sum(wanted.values())
    if total >= _LAYOUT_COVERAGE_MIN_TOKENS and sum((wanted & tokens).values()) / total >= _LAYOUT_COVERAGE:
        return True, None
    return False, None


def reconcile_layout(
    pairs: list[tuple], exp: list[Element], act: list[Element],
    exp_figures: list[Element] | None = None, act_figures: list[Element] | None = None,
) -> list[tuple]:
    """Excuse one-sided content that the other document prints anyway.

    The two documents do not agree on what is a table: Production p.28 prints a
    UI-element table as rows of prose ("Pop-up messages Enable/Disable pop-up
    notifications.") that Staging extracts as one table, and a table never pairs
    with a paragraph - so every row was reported missing and the table added,
    85 findings in one chapter, none of them true. That is the same content
    divided up differently, which is the rule this check is built on (the wrap,
    the page break), so each such element becomes a `layout` row: shown in the
    side-by-side list with where its words are, never counted as a difference.

    Returns `(exp_group, act_group, note, container)` for every pair.
    """
    # Only content with NOTHING of its own to answer to may excuse a one-sided
    # element - a legitimate "same content, divided up differently" case (a
    # table's rows against the paragraph they were read from, both otherwise
    # unmatched). Content a normal, both-sided pair ALREADY accounts for is
    # spoken for: crediting it a second time is what let an entire numbered
    # procedure Staging printed TWICE - once as the real, matching copy, once
    # as a genuinely extra duplicate - excuse its own duplicate by pointing at
    # the very same Production text its legitimate twin was already paired
    # with, so the duplicate silently vanished instead of being reported.
    matched_exp_ids = {id(e) for eg, ag, _ in pairs if eg and ag for e in eg}
    matched_act_ids = {id(e) for eg, ag, _ in pairs if eg and ag for e in ag}
    exp_spare = [e for e in exp if id(e) not in matched_exp_ids]
    act_spare = [e for e in act if id(e) not in matched_act_ids]
    exp_spare_flats = [_flat(e.text) if e.kind != KIND_FIGURE else "" for e in exp_spare]
    act_spare_flats = [_flat(e.text) if e.kind != KIND_FIGURE else "" for e in act_spare]
    exp_spare_blob = " ".join(f for f in exp_spare_flats if f)
    act_spare_blob = " ".join(f for f in act_spare_flats if f)
    exp_spare_tokens, act_spare_tokens = _tokens(exp_spare_blob), _tokens(act_spare_blob)
    # `exp`/`act` here are the topic's TEXT elements only (figures are matched
    # separately, by `figure_changes`) - a caption's own figure is passed in
    # from there so a one-sided caption can still be told from ordinary text.
    if exp_figures is None:
        exp_figures = [e for e in exp if e.kind == KIND_FIGURE]
    if act_figures is None:
        act_figures = [e for e in act if e.kind == KIND_FIGURE]

    def is_figure_caption(el: Element, own_figures: list[Element]) -> bool:
        # A short label captioning a nearby icon ("WEEE", "Battery") is held
        # to no leniency at all: the recycling symbol's own caption starting
        # with the same word as an unrelated "WEEE directive..." paragraph
        # elsewhere on the page is a coincidence of the acronym, not evidence
        # the icon still carries its label - losing it is a real, reportable
        # difference, never waved through as "found elsewhere".
        return bool(el.text) and len(el.text) <= _LAYOUT_SHORT_CHARS and _near_a_figure(
            el, own_figures, reach=_HEADING_CAPTION_REACH
        )

    out: list[tuple] = []
    for exp_group, act_group, note in pairs:
        container = None
        if exp_group and not act_group and len(exp_group) == 1:
            if not is_figure_caption(exp_group[0], exp_figures):
                found, container = _present_elsewhere(exp_group[0], act_spare, act_spare_flats, act_spare_blob, act_spare_tokens)
                if found:
                    note = "layout"
        elif act_group and not exp_group and len(act_group) == 1:
            if not is_figure_caption(act_group[0], act_figures):
                found, container = _present_elsewhere(act_group[0], exp_spare, exp_spare_flats, exp_spare_blob, exp_spare_tokens)
                if found:
                    note = "layout"
        out.append((exp_group, act_group, note, container))
    return out


# --- what differs ----------------------------------------------------------


def _colour_name(color: int) -> str:
    return f"#{color & 0xFFFFFF:06x}"


def _colour_distance(a: int, b: int) -> int:
    """How far apart two text colours are, per channel. A document re-exported
    through another tool routinely shifts pure black by a point or two, and
    reporting that as "the colour changed" on every paragraph is worthless."""
    return max(
        abs(((a >> shift) & 0xFF) - ((b >> shift) & 0xFF))
        for shift in (16, 8, 0)
    )


_EDGE_TOLERANCE = 4.0   # points: a cell edge this close to a column rule is on it


def _column_edges(t: Element) -> list[float]:
    """The table's interior column rules: cell edges shared by most rows."""
    counts: dict[int, int] = {}
    rows = 0
    for row in t.grid:
        cells = [c for c in row if c]
        if not cells:
            continue
        rows += 1
        for x in {round(v) for c in cells for v in c}:
            counts[x] = counts.get(x, 0) + 1
    if not rows:
        return []
    edges = sorted(x for x, n in counts.items() if n >= max(2, rows // 2))
    return edges[1:-1] if len(edges) > 2 else []


def _row_merges(t: Element, row: tuple, edges: list[float]) -> int | None:
    """How many cells in this row span across a column rule; None when the row's
    cells could not be read (a row printed without rules)."""
    if not row or any(c is None for c in row):
        return None
    return sum(
        1 for x0, x1 in row
        if any(x0 + _EDGE_TOLERANCE < e < x1 - _EDGE_TOLERANCE for e in edges)
    )


def _merged_rows(a: Element, b: Element) -> list[int]:
    """Rows where a cell spans a column rule on one side and not on the other.

    Judged on the cells' own geometry against the table's column rules - never
    on the reader's empty-cell markers or on how many cells the words fill,
    both of which a table printed without vertical rules gets wrong in every
    row while nothing in it is merged. A row whose cells cannot be read on
    either side is not judged at all."""
    edges_a, edges_b = _column_edges(a), _column_edges(b)
    if not edges_a or not edges_b:
        return []

    def by_label(t: Element, edges: list[float]) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for i, (texts, row) in enumerate(zip(t.cells, t.grid)):
            label = next((c for c in texts if c), "")
            merges = _row_merges(t, row, edges)
            if label and merges is not None:
                out.setdefault(label, (i, merges))
        return out

    ea, eb = by_label(a, edges_a), by_label(b, edges_b)
    return sorted(ea[k][0] + 1 for k in ea.keys() & eb.keys() if (ea[k][1] > 0) != (eb[k][1] > 0))


def _same_characters(a: Element, b: Element) -> bool:
    """Both tables hold exactly the same characters, all cells pooled: the grid
    was only divided up differently by the two extractions. Production's Taiwan
    RoHS table read its two-row header as one row, which shifted every cell below
    it and reported twenty-two "changed" cells in a table identical in print."""
    def characters(el: Element) -> Counter:
        return Counter(ch for row in el.cells for cell in row for ch in (cell or "") if not ch.isspace())
    mine = characters(a)
    return bool(mine) and mine == characters(b)


def _table_cell_changes(a: Element, b: Element) -> list[str]:
    """Which cells say something different, row by row. Compared by position
    within the row, but only for rows that still line up - a table that gained
    a row is reported as a shape change, not as every cell below it changing."""
    out: list[str] = []
    for r, (row_a, row_b) in enumerate(zip(a.cells, b.cells), start=1):
        for c, (cell_a, cell_b) in enumerate(zip(row_a, row_b), start=1):
            if cell_a != cell_b:
                out.append(f"row {r}, column {c}: “{cell_a or '—'}” → “{cell_b or '—'}”")
    return out


def compare_group(
    exp: list[Element], act: list[Element], size_scale: float, moved: bool = False
) -> list[dict]:
    """Every way this one piece of the chapter differs between the documents.

    A group is what one side prints against what the other prints in its place:
    usually one element each, sometimes two against one where the documents
    divide the content differently. Merging each side before comparing is what
    makes a differently-divided paragraph a non-event.

    `size_scale` is Staging's body size over Production's: a document rebuilt a
    point larger throughout is not "every paragraph resized", so sizes are
    compared after dividing that out.
    """
    a, b = merge_elements(exp), merge_elements(act)
    out: list[dict] = []
    if moved and a is not None and b is not None:
        out.append({
            "type": "moved", "kind": a.kind,
            "summary": f"{KIND_LABEL[a.kind]} out of sequence — it is printed in a different place in Staging, "
                       f"not where Production has it in the reading order.",
            "detail": (a.text or "")[:200], "minor": True,
        })
    found = _compare_merged(a, b, size_scale)
    found += paragraph_gap_changes(exp, act)
    sides = [n for n, el in (("Production", a), ("Staging", b)) if el is not None and el.ocr]
    for d in found:
        if sides:
            d["ocr_sides"] = sides
    out.extend(found)
    return reportable(out)


_PARAGRAPH_GAP_MIN = 0.5   # a gap this share of a line's height between two blocks is a paragraph space


def paragraph_gap_changes(exp: list[Element], act: list[Element]) -> list[dict]:
    """Production prints two paragraphs with a space between them where
    Staging prints the same words as one block with no space: the paragraph
    break is lost. Only same-page neighbours with a visible gap count, so a
    paragraph Production merely continues on the next page is not one."""
    texts_a = [e for e in exp if e.kind == KIND_TEXT and e.text]
    texts_b = [e for e in act if e.kind == KIND_TEXT and e.text]
    if len(texts_a) < 2 or len(texts_b) != 1:
        return []
    out: list[dict] = []
    for first, second in zip(texts_a, texts_a[1:]):
        (p1, b1), (p2, b2) = first.boxes[-1], second.boxes[0]
        if p1 != p2:
            continue  # continued over a page break: not a paragraph space
        line = max(1.0, first.size or (b1[3] - b1[1]))
        gap = b2[1] - b1[3]
        if gap < _PARAGRAPH_GAP_MIN * line:
            continue
        joined = texts_b[0]
        if _normalise(second.text)[:40] not in joined.key:
            continue
        out.append({
            "type": "paragraph-gap", "kind": KIND_TEXT, "exp": second, "act": joined,
            "summary": (f"Paragraph space missing in Staging — Production starts a new paragraph at "
                        f"“{second.text[:60]}” with a space above it; Staging runs it on in the same "
                        f"paragraph with no space."),
            "detail": second.text[:200],
        })
    return out


_BARE_MARKER_RE = re.compile(
    r"^(?:\d{1,2}[.)]"
    r"|(?:[a-z]|ii|iii|iv|vi|vii|viii|ix|xi|xii)[.)]"
    r"|[•◦▪▸‣⁃·∙])$",
    re.IGNORECASE,
)


def _looks_like_extraction_garbage(text: str) -> bool:
    """Mostly PyMuPDF's own "(cid:N)" placeholder for a glyph with no
    ToUnicode mapping at all - a font subset whose text layer could not be
    read, not prose the other side genuinely lacks. A whole element made of
    this (a table built from such a font, most often) has nothing reliable
    to compare, so reporting it "missing"/"added" would claim a content
    difference the extraction itself never actually read. A stray "(cid:N)"
    inside an otherwise normal sentence is left alone - that is one bad
    glyph in real prose, not an unreadable element."""
    text = text or ""
    cid_chars = sum(len(m.group()) for m in _CID_TOKEN_RE.finditer(text))
    return cid_chars > 0 and cid_chars >= 0.5 * len(text.strip())


def _looks_like_bare_marker(text: str) -> bool:
    """A lone list-number, letter/roman-numeral sub-step marker, or bullet
    glyph with nothing else to it - split off from its own sentence because
    reading order put it there, or a step-number badge drawn apart from its
    heading rather than inline with it. Never meaningful content on its own,
    whichever side prints it alone, so it is not worth a "missing"/"added"
    finding - the sentence it belongs to is compared (and reported on) as its
    own element regardless. A letter/numeral needs its own trailing "." or ")"
    to count: a bare "a" with no punctuation is left alone, since that is also
    how a genuinely doubled word ("without a a grommet ring") reads out."""
    return bool(_BARE_MARKER_RE.match((text or "").strip()))


# The Private Use Area is deliberately NOT here on its own - manuals
# legitimately map callout/UI glyphs into it, and a PUA character already
# present in Production's own copy is the source document's own font, not a
# regression. It only counts once it is NEW in Staging - see `_artifact_set`.
_ENCODING_ARTIFACT_RE = re.compile(
    r"[�￾￿\U0001fffe\U0001ffff\U0002fffe\U0002ffff\U0010fffe\U0010ffff]"
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f]"
)
_CID_TOKEN_RE = re.compile(r"\(cid:\s?\d+\)")  # PyMuPDF's literal text for a glyph with no ToUnicode mapping
_PUA_RE = re.compile(r"[-\U000f0000-\U000ffffd\U00100000-\U0010fffd]")


def _artifact_set(text: str) -> set[str]:
    """The distinct un-decodable markers in `text`: the replacement character
    and other noncharacters, raw control characters, unmapped Private-Use
    glyphs, and the literal "(cid:N)" token PyMuPDF prints for a glyph with no
    ToUnicode mapping at all - every one of them a place the text layer could
    not read what is actually printed there."""
    out = {ch for ch in (text or "") if _ENCODING_ARTIFACT_RE.match(ch) or _PUA_RE.match(ch)}
    if _CID_TOKEN_RE.search(text or ""):
        out.add("(cid:N)")
    return out


def _opposite_text(bad_text: str, a_text: str, b_text: str, from_a: bool) -> str:
    """What the OTHER document prints where `bad_text`'s undecodable run sits.

    An unreadable run says nothing on its own - the reader needs to see what
    stands in its place on the other side, which is the whole point of the
    comparison. The word diff of the two texts pairs them up: the deleted run
    holding the artifact against the inserted run opposite it.
    """
    ops = _word_diff(a_text or "", b_text or "")
    mine, theirs = ("del", "ins") if from_a else ("ins", "del")
    found: list[str] = []
    i = 0
    while i < len(ops):
        if ops[i]["type"] == "equal":
            i += 1
            continue
        block = []
        while i < len(ops) and ops[i]["type"] != "equal":
            block.append(ops[i])
            i += 1
        # As many words as the unreadable run itself is long, not the whole
        # opposite block: an unreadable entry sitting next to a genuinely
        # missing one merges into one block, and quoting all of it names
        # content that is not what stands in the unreadable run's place.
        bad = [w for o in block if o["type"] == mine for w in o["text"].split() if _artifact_set(w)]
        if not bad:
            continue
        opposite = [w for o in block if o["type"] == theirs for w in o["text"].split()]
        if opposite:
            found.append(" ".join(opposite[: len(bad)]))
    return " ".join(found)[:80]


def _encoding_regression(kind: str, a_text: str, b_text: str) -> dict | None:
    """Characters one side's text layer cannot decode at all - a replacement
    character, an unmapped Private-Use glyph, a bare "(cid:N)" - where the
    other side's matching text decodes cleanly.

    What this does NOT establish is which document is at fault, and the old
    wording ("Text encoding issue in Production") asserted exactly that. An
    unreadable text layer is not the same as a wrongly printed page: this
    manual's Production copy cannot decode its Simplified Chinese menu entry
    yet prints it perfectly, while Staging decodes to four real-but-wrong
    codepoints and prints four wrong glyphs - so the side the old message
    blamed was the side that was right. All the text layers can honestly say
    is that the two documents disagree HERE and neither reading can be
    trusted, so the finding now shows both readings and asks for the row to be
    checked by eye, rather than failing one document on the strength of a
    text layer that by its own admission could not be read.
    """
    for lost, x_text, y_text in ((False, a_text, b_text), (True, b_text, a_text)):
        new = _artifact_set(y_text) - _artifact_set(x_text)
        if new:
            shown = ", ".join(sorted(f"“{ch}”" if ch != "(cid:N)" else ch for ch in new)[:6])
            side = "Staging" if not lost else "Production"
            other = "Production" if not lost else "Staging"
            opposite = _opposite_text(y_text, a_text, b_text, from_a=lost)
            return {
                "type": "text-encoding", "kind": kind,
                "summary": (
                    f"Characters cannot be read in {side} — {len(new)} character(s) have no usable "
                    f"encoding ({shown}), so what {side} prints here cannot be compared by text"
                    + (f", while {other} prints “{opposite}” in their place. Check this by eye: the two "
                       f"documents disagree here and neither text layer can settle it."
                       if opposite else ". Check by eye what each document prints here.")
                ),
                "detail": (y_text or "")[:200],
                # Shown for a reviewer, never failed on: the evidence is a text
                # layer that could not be read, which cannot prove a defect.
                "review_only": True,
            }
    return None


# --- a glyph that decodes right but does not DRAW right ---------------------
#
# `_encoding_regression` catches a character the text layer cannot decode at
# all - a replacement character, an unmapped Private-Use glyph, a bare
# "(cid:N)". It has nothing to say about a character that decodes PERFECTLY
# FINE (every text-based check above trusts it, correctly) but whose font
# draws the wrong OUTLINE for that glyph ID - a font-subsetting bug where the
# ToUnicode map and the glyph table disagree. A reader sees one character;
# the text layer says another. The only way to catch that is to look at what
# is actually drawn, which is what OCR is for.

_GLYPH_STRIP_RE = re.compile(r"[\s，,。.、；;：:！!？?（）()“”‘’–—\-~*]")
_GLYPH_MIN_LEN = 8       # characters, after stripping - too little context to trust a single substitution
_GLYPH_MIN_FLANK = 3     # characters required to read back correctly on EACH side of the suspect one
_GLYPH_MAX_BAD = 2       # at most this many characters may actually differ


def glyph_render_changes(act_elements: list[Element], actual: fitz.Document) -> list[dict]:
    """A single character (or two) inside a Staging element that Tesseract,
    reading the page exactly as it is drawn, does not see - while every
    character before and after it, in the same element, reads back exactly
    as the text layer says. That specific shape - one clean substitution deep
    inside an otherwise perfect OCR read - is what a broken embedded-font
    glyph looks like. Scattered, page-wide OCR noise (Tesseract's ordinary
    error rate on dense non-Latin text, worse at small sizes and inside
    NOTE boxes) does not take this shape - a genuine misread disagrees with
    its own neighbours too, not just the one word in between - so requiring
    a long, clean run of agreement on both sides is what keeps this from
    firing on ordinary OCR imprecision.

    Reported as a REVIEW finding, never a confident assertion: OCR itself is
    never perfect enough to be the last word on what is printed, only reason
    enough to point a reviewer's eye at the exact spot.

    Production is not checked here - not because it cannot have the same
    bug, but because a PRODUCTION defect visually identical on both sides is
    not something Staging did wrong, and every other absolute check in this
    file (a broken image, an unstyled link) is scoped to Staging alone for
    the same reason."""
    if not act_elements:
        return []
    from pdfval import ocr

    if not ocr.available():
        return []
    out: list[dict] = []
    for el in act_elements:
        if el.kind not in (KIND_TEXT, KIND_NOTE) or not el.text or not el.boxes:
            continue
        text_clean = _GLYPH_STRIP_RE.sub("", el.text)
        if len(text_clean) < _GLYPH_MIN_LEN or not _CJK_RE.search(text_clean):
            continue
        page_index, bbox = el.boxes[0]
        words = ocr.words_in(_page_words(actual).ocr_words(page_index), bbox, margin=4)
        if not words:
            continue
        read_clean = _GLYPH_STRIP_RE.sub("", "".join(words))
        if not read_clean:
            continue
        ops = [op for op in difflib.SequenceMatcher(None, text_clean, read_clean).get_opcodes()
               if op[0] != "equal"]
        if len(ops) != 1 or ops[0][0] != "replace":
            continue
        _, i1, i2, j1, j2 = ops[0]
        if (i2 - i1) > _GLYPH_MAX_BAD or (j2 - j1) > _GLYPH_MAX_BAD:
            continue
        if i1 < _GLYPH_MIN_FLANK or len(text_clean) - i2 < _GLYPH_MIN_FLANK:
            continue
        out.append({
            "type": "glyph-render", "kind": el.kind, "exp": None, "act": el,
            "summary": (f"Possible broken glyph in Staging — the text says “{text_clean[i1:i2]}”, but the "
                        f"page itself reads as “{read_clean[j1:j2]}” there; worth a look by eye."),
            "detail": text_clean[:200],
            "review_only": True,
        })
    return out


def _compare_merged(a: Element | None, b: Element | None, size_scale: float) -> list[dict]:
    if a is None and b is None:
        return []
    if b is None:
        if (a.kind == KIND_TEXT and _looks_like_bare_marker(a.text)) or _looks_like_extraction_garbage(a.text):
            return []
        return [{
            "type": "missing",
            "kind": a.kind,
            "summary": f"{KIND_LABEL[a.kind]} in Production only — nothing answers to it in Staging.",
            "detail": a.text[:400] if a.text else f"{int(a.width)}×{int(a.height)}pt figure",
        }]
    if a is None:
        if (b.kind == KIND_TEXT and _looks_like_bare_marker(b.text)) or _looks_like_extraction_garbage(b.text):
            return []
        return [{
            "type": "added",
            "kind": b.kind,
            "summary": f"{KIND_LABEL[b.kind]} in Staging only — Production has nothing here.",
            "detail": b.text[:400] if b.text else f"{int(b.width)}×{int(b.height)}pt figure",
        }]

    out: list[dict] = []

    if a.kind == KIND_FIGURE and b.kind == KIND_FIGURE:
        score = _figure_similarity(a, b)
        if score < _FIGURE_DIFFERENT:
            out.append({
                "type": "figure-content", "kind": KIND_FIGURE,
                "summary": "Figure replaced — the artwork in this place is a different picture.",
                "detail": f"appearance match {score:.0%}",
            })
        elif score < _FIGURE_SAME:
            out.append({
                "type": "figure-review", "kind": KIND_FIGURE,
                "summary": "Figure could not be confirmed identical — compare the two crops.",
                "detail": f"appearance match {score:.0%}",
                "minor": True,
            })
        for axis, va, vb in (("width", a.width, b.width), ("height", a.height, b.height)):
            if va and abs(vb - va) / va >= _FIGURE_SIZE_STEP and abs(vb - va) >= _FIGURE_SIZE_MIN_PT:
                out.append({
                    "type": "figure-size", "kind": KIND_FIGURE,
                    "summary": f"Figure {axis} changed — {va:.0f}pt in Production, {vb:.0f}pt in Staging.",
                    "detail": f"{(vb - va) / va:+.0%}",
                    "minor": True,
                })
        return out

    if a.kind == KIND_TABLE and b.kind == KIND_TABLE:
        if (a.header_fill is None) != (b.header_fill is None):
            out.append({
                "type": "table-header-fill", "kind": KIND_TABLE,
                "summary": (
                    "Table header has no background colour in Staging — Production highlights the "
                    f"header row ({_colour_name(a.header_fill)}), Staging leaves it plain."
                    if b.header_fill is None else
                    "Table header has a background colour only in Staging — Production leaves the "
                    f"header row plain, Staging fills it ({_colour_name(b.header_fill)})."
                ),
                "detail": "",
            })
        elif a.header_fill is not None and _colour_distance(a.header_fill, b.header_fill) >= _COLOUR_STEP:
            out.append({
                "type": "table-header-fill", "kind": KIND_TABLE,
                "summary": (
                    f"Table header background colour changed — {_colour_name(a.header_fill)} in "
                    f"Production, {_colour_name(b.header_fill)} in Staging."
                ),
                "detail": "",
            })
        rows = _merged_rows(a, b) if a.cols == b.cols else []
        if rows:
            out.append({
                "type": "table-merge", "kind": KIND_TABLE,
                "summary": (
                    "Table cells merged or split differently — row(s) "
                    f"{', '.join(map(str, rows[:8]))}{' …' if len(rows) > 8 else ''} are divided "
                    "into cells differently in Staging."
                ),
                "detail": "",
            })
        if (a.rows, a.cols) != (b.rows, b.cols):
            # One side's grid read with more (or fewer) cells than the other's
            # while saying the same thing is how the two PDFs are ruled, not a
            # change to the table. Said plainly, and left for the eye.
            subset = _containment(a.key, b.key) >= 0.9
            out.append({
                "type": "table-extract" if subset else "table-shape",
                "kind": KIND_TABLE,
                "summary": (
                    "Table is laid out differently — it reads as "
                    f"{a.rows}×{a.cols} in Production against {b.rows}×{b.cols} in Staging "
                    "(rows × columns), with the same wording; compare the two crops."
                    if subset else
                    f"Table shape changed — {a.rows}×{a.cols} in Production, "
                    f"{b.rows}×{b.cols} in Staging (rows × columns)."
                ),
                "detail": "",
                "minor": subset,
            })
            return out  # cell-by-cell is meaningless across different shapes
        cells = max(1, a.rows * a.cols)
        empty_a = sum(1 for row in a.cells for c in row if not c)
        empty_b = sum(1 for row in b.cells for c in row if not c)
        if abs(empty_a - empty_b) / cells >= _TABLE_EMPTY_ASYMMETRY:
            # Same grid, but a third of the cells are blank on one side only:
            # one document's table extracted its values and the other's did
            # not. Listing every blank cell as "changed" would be a storm of
            # findings about the extractor, not about the table.
            out.append({
                "type": "table-extract", "kind": KIND_TABLE,
                "summary": (
                    "Table cells read empty on one side and filled on the other — "
                    f"{empty_a} blank in Production, {empty_b} in Staging; compare the two crops."
                ),
                "detail": "", "minor": True,
            })
            return out
        if _same_characters(a, b):
            return out  # the same content, divided into cells differently by the two extractions
        changes = _table_cell_changes(a, b)
        if changes:
            out.append({
                "type": "table-cell", "kind": KIND_TABLE,
                "summary": f"{len(changes)} table cell(s) say something different.",
                "detail": "; ".join(changes[:6]) + (" …" if len(changes) > 6 else ""),
            })
        return out

    if a.kind != KIND_FIGURE:
        encoding = _encoding_regression(a.kind, a.text, b.text)
        if encoding:
            out.append(encoding)

    if a.key != b.key:
        gone, extra = _content_runs(a.key, b.key)
        if gone or extra:
            out.append({
                "type": "text", "kind": a.kind,
                "gone": gone, "extra": extra,
                "summary": _content_summary(a.kind, gone, extra),
                "detail": "",
                "word_diff": _word_diff(a.text, b.text),
            })
        else:
            # `_content_runs` compares WORDS only (`_TOKEN_RE` strips every
            # punctuation mark before it looks) - deliberately, so a real
            # content loss isn't lost among reflow noise. That also made a
            # genuine punctuation-only edit invisible: the same words, but a
            # sentence that lost its period, or a comma where Production has
            # none. `a.key != b.key` already proved something besides
            # whitespace differs once the word-level check comes back empty;
            # `_word_diff` (word-tokenised, not the character key) confirms
            # it is real punctuation and not just quote-style folding.
            wd = _word_diff(_PAGE_REF_RE.sub("", a.text), _PAGE_REF_RE.sub("", b.text))
            changed = [op for op in wd if op["type"] != "equal"]
            if changed:
                changed_text = " ".join(op["text"] for op in changed)
                if _only_bullet_glyphs(changed_text):
                    pass  # the list's own marker change - see _only_bullet_glyphs
                elif _has_symbol(changed_text):
                    out.append({
                        "type": "symbol", "kind": a.kind,
                        "summary": "Symbol changed — the wording is the same, but a symbol or mark differs.",
                        "detail": "",
                        "word_diff": wd,
                    })
                elif mark_changes(a.text, b.text, a.wraps, b.wraps):
                    # named mark by mark by the punctuation pass just below
                    pass

    # Punctuation, quotes and spaces between the words both sides print -
    # checked whatever else differs, so a sentence that changed a word AND lost
    # its period reports both.
    if a.kind in (KIND_TEXT, KIND_NOTE, KIND_HEADING) and a.text and b.text \
            and not any(d.get("type") in ("punctuation", "symbol") for d in out):
        marks = mark_changes(a.text, b.text, a.wraps, b.wraps)
        if marks:
            out.append({
                "type": "punctuation", "kind": a.kind, "mark_notes": marks,
                "summary": "Punctuation or spacing changed — " + "; ".join(marks[:6])
                           + (f"; and {len(marks) - 6} more" if len(marks) > 6 else "") + ".",
                "detail": "",
                "word_diff": _word_diff(_PAGE_REF_RE.sub("", a.text), _PAGE_REF_RE.sub("", b.text)),
            })

    if a.kind == KIND_NOTE and a.label != b.label:
        out.append({
            "type": "note-label", "kind": KIND_NOTE,
            "summary": f"Callout type changed — “{a.label}” in Production, “{b.label or 'none'}” in Staging.",
            "detail": "",
        })

    # Neither direction is reported on headings or heading-like labels: their
    # styling is Staging's design (see `bold-added` below).
    # A callout's own label ("Tip", "NOTE:") is the callout's styling - Staging
    # sets every label bold in its note box - never a bold change in the text.
    lost = [] if KIND_HEADING in (a.kind, b.kind) else [
        p for p in _bold_missing(a, b) if not _BARE_CALLOUT_RE.match(p)]
    if lost and _is_heading_style_label(a, lost):
        lost = []
    if lost:
        shown = ", ".join(f"“{w}”" for w in lost[:8]) + (" …" if len(lost) > 8 else "")
        out.append({
            "type": "bold-missing", "kind": a.kind,
            "summary": f"Bold missing in Staging — {shown} is bold in Production and regular in Staging.",
            "detail": "",
            # Only a candidate until `confirm_bold` has looked at the pages.
            "phrases": lost,
        })
    # Bold the other way: set bold in Staging where Production prints the words
    # regular. Not on headings - Staging sets every heading bold where
    # Production sets them large and regular, one decision that would bury the
    # rest; a heading's styling is its own question. A short standalone label
    # ("Panel", "System", "Connectivity") that Staging bolds in FULL - not a
    # word or two inside a longer sentence - is the same styling decision even
    # when it wasn't picked up as a heading (too small, or not on the TOC).
    gained = (
        [] if KIND_HEADING in (a.kind, b.kind)
        else [p for p in _bold_missing(b, a) if not _BARE_CALLOUT_RE.match(p)]
    )
    if gained and _is_heading_style_label(b, gained):
        gained = []
    if gained:
        shown = ", ".join(f"“{w}”" for w in gained[:8]) + (" …" if len(gained) > 8 else "")
        out.append({
            "type": "bold-added", "kind": b.kind,
            "summary": f"Bold added in Staging — {shown} is bold in Staging and regular in Production.",
            "detail": "",
            "phrases": gained,
        })
    # Underline the same way bold is checked: a PDF has no font flag for it, so
    # `underline_words` already comes from a drawn rule (see `_line_underlined`)
    # rather than a candidate needing confirmation on the page.
    lost_underline = [] if KIND_HEADING in (a.kind, b.kind) else _underline_missing(a, b)
    if lost_underline:
        shown = ", ".join(f"“{w}”" for w in lost_underline[:8]) + (" …" if len(lost_underline) > 8 else "")
        out.append({
            "type": "underline-missing", "kind": a.kind,
            "summary": f"Underline missing in Staging — {shown} is underlined in Production and plain in Staging.",
            "detail": "",
        })
    gained_underline = [] if KIND_HEADING in (a.kind, b.kind) else _underline_missing(b, a)
    if gained_underline:
        shown = ", ".join(f"“{w}”" for w in gained_underline[:8]) + (" …" if len(gained_underline) > 8 else "")
        out.append({
            "type": "underline-added", "kind": b.kind,
            "summary": f"Underline added in Staging — {shown} is underlined in Staging and plain in Production.",
            "detail": "",
        })
    # Italic the same way, from the font itself: an italic or oblique face
    # (flag or font name), both directions, headings left to their own styling.
    for own, other, kind, verb in ((a, b, "italic-missing", "italic in Production and upright in Staging"),
                                   (b, a, "italic-added", "italic in Staging and upright in Production")):
        slanted = [] if KIND_HEADING in (a.kind, b.kind) else _italic_missing(own, other)
        if slanted:
            shown = ", ".join(f"“{w}”" for w in slanted[:8]) + (" …" if len(slanted) > 8 else "")
            is_are = "is" if len(slanted) == 1 else "are"
            label = "Italic missing in Staging" if kind == "italic-missing" else "Italic added in Staging"
            out.append({
                "type": kind, "kind": own.kind, "phrases": slanted,
                "summary": f"{label} — {shown} {is_are} {verb}.",
                "detail": "",
            })
    return out


def _italic_missing(a: Element, b: Element) -> list[str]:
    """Phrases set italic in `a` whose words `b` prints, but not in italic -
    the same walk as `_bold_missing`, over `italic_words`."""
    a_italic, b_italic = Counter(a.italic_words), Counter(b.italic_words)
    b_words = _tokens(b.key)
    lost: Counter = Counter()
    for word, count in a_italic.items():
        if not b_words.get(word):
            continue
        shortfall = min(count, b_words[word]) - b_italic.get(word, 0)
        if shortfall > 0:
            lost[word] = shortfall
    if not lost:
        return []
    out: list[str] = []
    run: list[str] = []
    for word in a.italic_words:
        if lost.get(word, 0) > 0:
            lost[word] -= 1
            run.append(word)
        elif run:
            out.append(" ".join(run))
            run = []
    if run:
        out.append(" ".join(run))
    return out


def _is_heading_style_label(b: Element, gained: list[str]) -> bool:
    """True when `gained` bolds an entire short, punctuation-free label
    ("Panel", "System", "Connectivity") rather than a word or two inside a
    longer sentence - the same styling choice already excused for headings."""
    if b.kind == KIND_TABLE or _MARKER_TAIL_RE.match((b.text or "").strip()):
        return False  # table cells and list items keep their bold checks
    words = _tokens(b.key)
    total = sum(words.values())
    if not total or total > 4:
        return False
    if re.search(r"[.!?,;:]\s*$", (b.text or "").strip()):
        return False
    bolded = sum(len(_TOKEN_RE.findall(phrase)) for phrase in gained)
    return bolded >= total


def _content_runs(a_key: str, b_key: str) -> tuple[list[str], list[str]]:
    """What Staging lost and gained, as runs of CONSECUTIVE words.

    Aligned word by word, so a run is a phrase that really was printed together
    - a plain count of words the two sides disagree on glued scattered words
    from all over a paragraph into "the the the menu select network". A run
    that is only moved within the element, and a hyphenated word rejoined on
    one side ("thirdparty" / "third party"), cancel out."""
    ta, tb = _TOKEN_RE.findall(a_key), _TOKEN_RE.findall(b_key)
    gone: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if tag in ("delete", "replace") and i1 < i2:
            gone.append(" ".join(ta[i1:i2]))
        if tag in ("insert", "replace") and j1 < j2:
            extra.append(" ".join(tb[j1:j2]))
    for run in list(gone):
        if run in extra:
            gone.remove(run)
            extra.remove(run)
    for run in list(gone):
        joined = run.replace(" ", "")
        match = next((e for e in extra if e.replace(" ", "") == joined), None)
        if match is not None:
            gone.remove(run)
            extra.remove(match)
    return gone, extra


_SPELLING_RATIO = 0.8   # two single words this alike are one word spelled differently


def _spelling_pairs(gone: list[str], extra: list[str]) -> dict[str, str]:
    """Single words Staging spells differently: Production's word -> Staging's."""
    pairs: dict[str, str] = {}
    free = [e for e in extra if " " not in e]
    for g in gone:
        if " " in g or len(g) < 4:
            continue
        best = max(free, key=lambda e: difflib.SequenceMatcher(a=g, b=e).ratio(), default=None)
        if best is not None and difflib.SequenceMatcher(a=g, b=best).ratio() >= _SPELLING_RATIO:
            pairs[g] = best
            free.remove(best)
    return pairs


def _content_summary(kind: str, gone: list[str], extra: list[str]) -> str:
    spelled = _spelling_pairs(gone, extra)
    if spelled:
        gone = [g for g in gone if g not in spelled]
        extra = [e for e in extra if e not in spelled.values()]
    parts = []
    if spelled:
        parts.append("spelled differently in Staging: " + "; ".join(f"“{g}” → “{e}”" for g, e in list(spelled.items())[:4]))
    if gone:
        parts.append("missing in Staging: " + "; ".join(f"“{r[:90]}”" for r in gone[:4]) + (" …" if len(gone) > 4 else ""))
    if extra:
        parts.append("extra in Staging: " + "; ".join(f"“{r[:90]}”" for r in extra[:4]) + (" …" if len(extra) > 4 else ""))
    return f"{KIND_LABEL.get(kind, 'Text')} content changed — " + " · ".join(parts) + "."


_ELSEWHERE_MIN_WORDS = 3


_SQUASHED_MIN_CHARS = 12   # below this, text found with its spaces ignored proves too little


_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_SHORT_LABEL_TOKENS = 6   # a label this short may be printed inside a longer line or a table
# `_printed_in`'s own fallback has no phrase or order requirement at all - it
# just asks whether every one of the element's words, as a bag, exists
# somewhere in the topic's remaining words - so it is trusted only for an
# actual one-or-two-word LABEL ("Deutsch", "WEEE"), never for a short
# SENTENCE: "Restart the device now." losing its own meaning while every one
# of its four common words happens to recur elsewhere in the topic is exactly
# the kind of miss a bag-of-words match cannot tell apart from a real one.
_SHORT_LABEL_BAG_TOKENS = 2


def _cells_text(el: Element) -> str:
    return " ".join(cell for row in el.cells for cell in row if cell)


def _text_units(text: str) -> Counter:
    """What text is compared by when the way it was divided into cells or blocks
    cannot be trusted: its characters for Chinese, Japanese and Korean - whose
    words have no spaces to split on, so each extraction cuts them differently -
    and its words otherwise."""
    text = text or ""
    solid = [ch for ch in text if not ch.isspace() and ch != "|"]
    if solid and len(_CJK_RE.findall(text)) >= 0.3 * len(solid):
        return Counter(solid)
    return Counter(_TOKEN_RE.findall(text))


def _chapter_units(elements: list[Element]) -> Counter:
    total: Counter = Counter()
    for el in elements:
        if el.kind != KIND_FIGURE:
            total += _text_units(_cells_text(el) if el.kind == KIND_TABLE else el.key)
    return total


def _topic_words(exp: list[Element], act: list[Element],
                 exp_doc: fitz.Document | None = None, act_doc: fitz.Document | None = None,
                 exp_figs: list[Element] | None = None, act_figs: list[Element] | None = None) -> tuple:
    """Everything each side prints in one topic, in the forms the settle checks
    read: (exp words, act words, exp units, act units, exp line keys, act line keys).

    Includes words OCR reads off this topic's own figures, not only the text
    layer: an OSD menu Production draws as real, extractable text ("DualView",
    "CAD / CAM") is often a single screenshot on the other side, the same
    words baked into it as pixels. Without this, every one of those labels
    settle_content cannot find in the text layer reports as content Staging
    dropped - the same false "missing" for a whole class of vector-text-against-
    embedded-screenshot pages, not one page's quirk.
    """
    def words(elements: list[Element], doc: fitz.Document | None, figs: list[Element] | None) -> str:
        out = " " + " ".join(w for e in elements for w in _TOKEN_RE.findall(e.key)) + " "
        if doc is None or not figs:
            return out
        from pdfval import ocr
        for f in figs:
            if not f.boxes:
                continue
            page_index, bbox = f.boxes[0]
            try:
                read = ocr.words_in(_page_words(doc).ocr_words(page_index), bbox, margin=4)
            except Exception:
                continue
            out += " " + " ".join(ocr.normalize_word(w) for w in read) + " "
        return out

    def keys(elements: list[Element]) -> set[str]:
        return {" ".join(_TOKEN_RE.findall(e.key)) for e in elements if e.key}

    return (words(exp, exp_doc, exp_figs), words(act, act_doc, act_figs),
            _chapter_units(exp), _chapter_units(act), keys(exp), keys(act))


def _printed_in(el: Element, other_words: str, other_units: Counter | None = None) -> bool:
    """Does the other document print this element's content in its copy of the
    chapter? Text in the same order (spacing ignored, since "47052070 or" and
    "47052070or" are the same print); a short label anywhere, since one side
    prints "Deutsch" beside its paragraph and the other inside it; every word or
    character of a table.

    A table split off a paired table onto its own (a continuation fragment
    that duplicates the header/footnote text of a table already accounted for
    elsewhere) will not hit 100% overlap - it shares that boilerplate with the
    OTHER split fragment too, so it's double-counted against a total that only
    has it once. Nor will a legend Production sets as plain paragraphs and
    Staging's extractor reads as a table grid: cutting the same words into
    cells moves word-boundaries a katakana long-vowel mark ("マーク" vs "マ" +
    "ク") sits on, so a handful of units come out split differently on each
    side even though every character is the same. 0.7 still rejects a
    genuinely new table, which shares essentially nothing with the other
    side's content."""
    tokens = _TOKEN_RE.findall(el.key)
    if not tokens:
        return False
    if el.kind == KIND_TABLE:
        mine = _text_units(_cells_text(el))
        theirs = other_units if other_units is not None else Counter(other_words.split())
        return bool(mine) and sum((mine & theirs).values()) >= 0.7 * sum(mine.values())
    run = " ".join(tokens)
    if f" {run} " in other_words:
        return True
    squashed = run.replace(" ", "")
    minimum = 4 if _CJK_RE.search(run) else _SQUASHED_MIN_CHARS
    if len(squashed) >= minimum and squashed in other_words.replace(" ", ""):
        return True
    # A heading's own words scattered separately elsewhere in the topic's body
    # prose ("copyright" in one sentence, "disclaimer" in another) is not the
    # heading printed elsewhere - unlike a short caption/label, a heading is
    # never legitimately folded into running text, so an extra/missing heading
    # (e.g. a "Copyright and Disclaimer" section only one side wraps as its own
    # heading) must not be silently settled just because its words are common.
    if el.kind == KIND_HEADING:
        return False
    return len(tokens) <= _SHORT_LABEL_BAG_TOKENS and not (Counter(tokens) - Counter(other_words.split()))


_TABLE_ROW_MATCH = 0.8  # a cell or row this alike is the same row, not a coincidence


def _matches_table_row(text: str, tables: list[Element] | None, min_ratio: float = _TABLE_ROW_MATCH) -> bool:
    """Whether `text` is (near enough to) one whole cell, or one whole row, of
    a table on the OTHER side - a stronger, structural match than "somewhere
    in this topic's running prose": a label-and-description table one side
    grids and the other prints as plain paragraphs still holds the very same
    row, one cell at a time, even when a single glyph the two extractions
    read differently (a subset font's ToUnicode map missing one character)
    keeps it from lining up as one exact, contiguous run of characters."""
    if not text or not tables:
        return False
    key = _normalise(text)
    if not key:
        return False
    for t in tables:
        for row in t.cells or ():
            for cell in row:
                if cell and _ratio(key, _normalise(cell)) >= min_ratio:
                    return True
            row_text = _normalise(" ".join(c for c in row if c))
            if row_text and _ratio(key, row_text) >= min_ratio:
                return True
    return False


def _elsewhere(run: str, other_words: str) -> bool:
    """Is a run of words one side lost printed elsewhere in the other side's
    chapter? Three words or more, verbatim; or - for a run the two extractions
    spaced differently, and for Chinese or Japanese, which has no spaces to
    trust - the same characters with spacing ignored.

    A heading-length label ("Directive WEEE") is too short to trust as a
    squashed-character match, but is exactly the case this settling exists
    for: Production glues it onto the paragraph under it as running text
    where Staging sets it apart as its own heading (or the reverse). Requires
    the exact phrase, not a word-count floor - "WEEE" and "Directive" out of
    order would not count."""
    words = run.split()
    if len(words) >= _ELSEWHERE_MIN_WORDS and f" {run} " in other_words:
        return True
    squashed = run.replace(" ", "")
    if _CJK_RE.search(run):
        return len(squashed) >= 4 and squashed in other_words.replace(" ", "")
    if len(words) >= _ELSEWHERE_MIN_WORDS and len(squashed) >= _SQUASHED_MIN_CHARS \
            and squashed in other_words.replace(" ", ""):
        return True
    return len(words) <= _SHORT_LABEL_TOKENS and len(squashed) >= 6 and f" {run} " in other_words


def _elsewhere_counted(run: str, own_words: str, other_words: str) -> bool:
    """`_elsewhere`, copy for copy: a phrase the other side prints elsewhere
    in the topic still counts as lost when that side prints it fewer times
    than this one - Production's "Power & Energy > Power & Energy" losing its
    repeat in Staging is a real edit, though Staging still prints the phrase
    once."""
    if not _elsewhere(run, other_words):
        return False
    copies = re.compile(rf"(?<= ){re.escape(run)}(?= )")
    own = len(copies.findall(own_words))
    return not own or len(copies.findall(other_words)) >= own


# --- tables, row by row ------------------------------------------------------
#
# A table is compared the way a reviewer reads it: row against row. Rows pair
# on what they say, never on their position, so a row inserted near the top
# does not make every row below it "changed". Then:
#   * a Production row with no counterpart is a missing row, a Staging row with
#     none an added one - unless its words are printed outside the other
#     document's table in the same section (Production sets the body of its
#     language table as plain text, which the table detector never reads);
#   * one row printed as two, or two as one, is a merge;
#   * a paired row divided into fewer or more cells is a merge;
#   * a paired cell whose words differ is a changed cell;
#   * a different number of columns is a column change.
_TABLE_GRID_TYPES = ("table-shape", "table-cell", "table-merge", "table-extract")
_ROW_MATCH = 0.85              # rows sharing this share of their words are the same row ...
_CELL_MARKERS = {"-", "–", "—", "•", "·", "*", "▪", "◦"}  # list markers inside a table cell
# A callout label printed in a cell ("NOTE:", "TIP:") is styling, never content.
_CALLOUT_LABEL_RE = re.compile(
    r"\b(?:" + "|".join(sorted({re.escape(w) for w in i18n._CALLOUT_WORDS}, key=len, reverse=True)) + r")\s*[:：]",
    re.IGNORECASE,
)


_CELL_HYPHEN_RE = re.compile(r"(\w)-(\w)")


def _cell_units(text: str) -> Counter:
    """A cell's words, a wrap hyphen joined back and a callout label taken out
    - the same two things `_normalise` takes out of running text, which plain
    `_text_units` does not: a cell wrapped at a hyphen ("factory pre-\nset")
    is read back by one table extractor with the hyphen kept, right up
    against "set" with no space to show where the line broke, and by the
    other with the hyphen simply dropped - "factory pre-set" against "factory
    preset", entirely from how each extractor rejoined the line, never a real
    difference. `_SOFT_HYPHEN_RE` cannot catch this: it requires the space a
    line break leaves in running text, which is exactly what a cell's own
    line-join already ate. Every hyphen between two letters is joined here,
    a plain cell being no place to tell a wrap from a genuinely hyphenated
    word like "USB-C" apart - and a real difference elsewhere in the cell's
    words still stands, this only forgives the hyphen itself."""
    text = _CELL_HYPHEN_RE.sub(r"\1\2", text or "")
    return _text_units(_CALLOUT_LABEL_RE.sub(" ", text))


_CELL_LINE_RE = re.compile(r"[\n\r]+")


def _cell_items(text: str) -> list[str]:
    """A cell's own printed lines - "English", "Français", "Deutsch" ... - the
    granularity a reordered list is judged at, one option per printed line."""
    return [_normalise(p) for p in _CELL_LINE_RE.split(text or "") if _normalise(p)]


def _reordered_items(x: str, y: str) -> bool:
    """Same lines, printed in a different order: Staging re-sequenced the same
    options rather than changing, adding or dropping any of them."""
    items_x, items_y = _cell_items(x), _cell_items(y)
    return len(items_x) >= 2 and items_x != items_y and Counter(items_x) == Counter(items_y)


_ROW_MATCH_LABELLED = 0.5      # ... or this share, when their first cell (the row's label) is the same
_CELL_SHIFT_SHARE = 0.4        # more paired rows than this with changed cells: the grid was read shifted
_TABLE_CELL_CONSOLIDATE_MIN = 3  # this many rows changed in one table: one layout issue, not one per row
_ROW_MERGE = 0.85              # a row this covered by two rows on the other side was split
_ROW_PRINTED_ELSEWHERE = 0.9   # a row whose words are this printed outside the other table is not lost
_ROWS_RELIABLE = 0.5           # below this share of rows paired, the grids were read too differently


def _printed_columns(table: Element) -> int:
    """The columns a reader sees, not the columns the ruling draws.

    A header cell ruled out of line with the values under it - Staging's
    accessories table puts "ST4304" over x 191-254 and its values over
    x 131-254 - comes out as two columns: one holding only the header text, the
    next holding only the values. Joined back, the table has the columns it
    prints, and a different count means the tables really differ."""
    rows = [r for r in table.cells if r]
    if len(rows) < 2:
        return table.cols
    width = max(len(r) for r in rows)
    head = [(rows[0][k] if k < len(rows[0]) else "") or "" for k in range(width)]
    body = rows[1:]

    def share(k: int) -> float:
        return sum(1 for r in body if k < len(r) and (r[k] or "").strip()) / max(1, len(body))

    count, k = 0, 0
    while k < width:
        if k + 1 < width:
            a_head, b_head = bool(head[k].strip()), bool(head[k + 1].strip())
            if a_head and not b_head and share(k) < _HEADER_ONLY_SHARE and share(k + 1) >= _BODY_ONLY_SHARE:
                count, k = count + 1, k + 2
                continue
            if b_head and not a_head and share(k + 1) < _HEADER_ONLY_SHARE and share(k) >= _BODY_ONLY_SHARE:
                count, k = count + 1, k + 2
                continue
        count, k = count + 1, k + 1
    return count


_HEADER_ONLY_SHARE = 0.1   # a column this empty under its header holds only the header
_BODY_ONLY_SHARE = 0.5     # a header-less column this full holds that header's values


def _row_text(row) -> str:
    return " ".join(c for c in row if c)


def _clip_row(row, limit: int = 60) -> str:
    text = _row_text(row)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _similar(u: Counter, v: Counter) -> float:
    total = max(sum(u.values()), sum(v.values()))
    return sum((u & v).values()) / total if total else 1.0


def _row_element(table: Element, indexes: list[int]) -> Element:
    """Some rows of a table, as an element boxed on exactly those rows."""
    rows = tuple(table.cells[i] for i in indexes)
    text = " | ".join(_row_text(r) for r in rows)
    boxes = [table.row_boxes[i] for i in indexes if i < len(table.row_boxes)] or list(table.boxes[:1])
    return Element(kind=KIND_TABLE, text=text, key=text, boxes=boxes, rows=len(rows),
                   cols=max((len(r) for r in rows), default=0), cells=rows)


def _row_label(row) -> str:
    return next((re.sub(r"\s+", "", c) for c in row if c and c.strip()), "")


def _pair_rows(a: Element, b: Element, rows_a: list[int], rows_b: list[int],
               units_a: list[Counter], units_b: list[Counter]) -> dict[int, int]:
    """Rows paired in reading order: identical rows first, then - inside each
    stretch that differs - a row with the same label (its first cell) and much
    the same content, or one that says nearly the same. Pairing on similarity
    alone matched "塑料外框 ○ ○ ○" with "后壳 ○ ○ ○": rows that share every
    circle and nothing else."""
    def signature(table: Element, index: int) -> str:
        return re.sub(r"\s+", "", _row_text(table.cells[index]))

    matcher = difflib.SequenceMatcher(a=[signature(a, i) for i in rows_a],
                                      b=[signature(b, j) for j in rows_b], autojunk=False)
    pairs: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.update({rows_a[i1 + k]: rows_b[j1 + k] for k in range(i2 - i1)})
        elif tag == "replace":
            floor = j1 - 1
            for x in range(i1, i2):
                i = rows_a[x]
                best, best_score = None, 0.0
                for y in range(floor + 1, j2):
                    j = rows_b[y]
                    score = _similar(units_a[i], units_b[j])
                    labelled = bool(_row_label(a.cells[i])) and _row_label(a.cells[i]) == _row_label(b.cells[j])
                    if not (score >= _ROW_MATCH or (labelled and score >= _ROW_MATCH_LABELLED)):
                        continue
                    score += 1.0 if labelled else 0.0
                    if score > best_score:
                        best, best_score = (y, j), score
                if best is not None:
                    pairs[i] = best[1]
                    floor = best[0]
    return pairs


def _spilled(left: Counter, other_rows: Counter) -> bool:
    """Words a cell holds on one side and not the other that are really header
    or neighbouring text the extraction spilled into it: real words (a Chinese
    character, or two letters or more), every one printed in the other table's
    other rows. A changed value - "o" to "x", "○" to "×" - is never spill."""
    return bool(left) and all(
        (_CJK_RE.match(u) or (len(u) >= 2 and u.isalpha())) for u in left
    ) and not (left - other_rows)


def _cells_in(table: Element, index: int) -> int:
    """How many real cells a row is divided into - a cell spanning two columns is one."""
    spanned = table.spans[index] if index < len(table.spans) else 0
    return len(table.cells[index]) - spanned


def _same_weight_in_topic(phrases: list[str], topic: list[Element], doc: fitz.Document) -> bool:
    """True when every phrase is printed bold somewhere in the other side's
    copy of the topic - "Brightness" bold in Production's table column as well
    as in Staging's label line. The bold check paired Staging's label with
    the regular word inside Production's description; the phrase itself
    has the same weight on both sides, so it is not a bold change."""
    if not phrases:
        return False
    bold_seq = [" " + " ".join(e.bold_words) + " " for e in topic if e.bold_words]
    table_bold: list[str] = []
    for el in topic:
        if el.kind != KIND_TABLE:
            continue
        for page_index, bbox in el.boxes:
            try:
                data = doc[page_index].get_text("dict", clip=fitz.Rect(*bbox))
            except Exception:
                continue
            words = [w for b in data.get("blocks", []) for ln in b.get("lines", []) for sp in ln.get("spans", [])
                     if sp.get("flags", 0) & _BOLD_FLAG for w in _TOKEN_RE.findall((sp.get("text") or "").casefold())]
            if words:
                table_bold.append(" " + " ".join(words) + " ")
    haystack = bold_seq + table_bold
    return all(
        any(" " + " ".join(_TOKEN_RE.findall(p.casefold())) + " " in h for h in haystack)
        for p in phrases
    )


_LABEL_LEAD_RE = r"\s*[—–:\-]\s+"   # "Input source — Switch to …" / "Brightness: Adjust …"


def table_fill_changes(a: Element, b: Element, expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A header or row Production prints on a background that Staging prints
    on plain paper. The COLOUR is Staging's design - any background counts -
    only a background gone altogether is reported.

    The header answers to the header; a body row to the Staging row with the
    same printed label (its first cell), in order when a label repeats."""
    if not a.cells or not b.cells or not a.row_boxes or not b.row_boxes:
        return []

    def shaded(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
        """True when coloured fills - however many shapes, one per cell or one
        per row - cover most of `bbox`. White and near-white are paper."""
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        try:
            drawings = doc[page_index].get_drawings()
        except Exception:
            return False
        covered = sum(
            _intersection(bbox, tuple(d["rect"])) for d in drawings
            if d.get("fill") and d.get("rect") is not None and min(d["fill"][:3]) < 0.95
            and d["rect"].height > 2   # a rule line, not a background
        )
        return covered / area >= _FILL_COVER

    def cell_boxes(el: Element, i: int) -> list[tuple[int, tuple] | None]:
        if i >= len(el.row_boxes):
            return []
        page_index, (_, y0, _, y1) = el.row_boxes[i]
        spans = el.grid[i] if i < len(el.grid) else ()
        return [(page_index, (x[0], y0, x[1], y1)) if x else None for x in spans]

    def row_fill(doc: fitz.Document, el: Element, i: int) -> list[bool]:
        return [bool(c) and shaded(doc, c[0], c[1]) for c in cell_boxes(el, i)]

    def fill(doc: fitz.Document, el: Element, i: int) -> int | None:
        return 1 if any(row_fill(doc, el, i)) else None

    labels_b: dict[str, list[int]] = {}
    for j in range(1, len(b.cells)):
        labels_b.setdefault((b.cells[j][0] if b.cells[j] else "") or "", []).append(j)
    pairs = [(0, 0)]
    used: dict[str, int] = {}
    for i in range(1, len(a.cells)):
        label = (a.cells[i][0] if a.cells[i] else "") or ""
        options = labels_b.get(label) if label else None
        if not options:
            continue
        k = used.get(label, 0)
        if k < len(options):
            pairs.append((i, options[k]))
            used[label] = k + 1

    # Cell by cell: the header row, a shaded label column, a highlighted row -
    # every cell Production prints on a background must have one in Staging.
    header_lost = False
    lost_columns: dict[int, list[str]] = {}
    boxes_a: list[tuple] = []
    boxes_b: list[tuple] = []
    for i, j in pairs:
        fa, fb = row_fill(expected, a, i), row_fill(actual, b, j)
        if not fa or not any(fa):
            continue
        ca, cb = cell_boxes(a, i), cell_boxes(b, j)
        same_shape = len(fa) == len(fb)
        for k, on in enumerate(fa):
            if not on:
                continue
            staging_has = fb[k] if same_shape and k < len(fb) else any(fb)
            if staging_has:
                continue
            if i == 0:
                header_lost = True
            else:
                lost_columns.setdefault(k, []).append((a.cells[i][0] if a.cells[i] else "") or "")
            if k < len(ca) and ca[k]:
                boxes_a.append(ca[k])
            if same_shape and k < len(cb) and cb[k]:
                boxes_b.append(cb[k])
            elif not same_shape and j < len(b.row_boxes):
                boxes_b.append(b.row_boxes[j])
    if not header_lost and not lost_columns:
        return []
    header_names = [c for c in (a.cells[0] if a.cells else ()) if c]
    parts = []
    if header_lost:
        parts.append("the header row")
    for k, labels in sorted(lost_columns.items()):
        column = header_names[k] if k < len(header_names) else f"column {k + 1}"
        shown = ", ".join(f"“{label[:30]}”" for label in labels[:4] if label) + (" …" if len(labels) > 4 else "")
        parts.append(f"the “{column}” column" + (f" ({shown})" if shown else ""))
    return [{
        "type": "table-fill-missing", "kind": KIND_TABLE,
        "summary": (f"Table background missing in Staging — {' and '.join(parts)} "
                    f"{'is' if len(parts) == 1 else 'are'} printed on a background in Production "
                    f"and on plain paper in Staging."),
        "detail": "",
        "exp": Element(kind=KIND_TABLE, text=a.text, key=a.key, boxes=boxes_a or [a.row_boxes[0]],
                       section=a.section),
        "act": Element(kind=KIND_TABLE, text=b.text, key=b.key, boxes=boxes_b or [b.row_boxes[0]],
                       section=b.section),
    }]


def _printed_as_line(doc: fitz.Document, pages: list[int], key: str) -> bool:
    """True when one of `pages` prints `key` as a line standing on its own."""
    if not key:
        return False
    for page_index in pages:
        try:
            data = doc[page_index].get_text("dict")
        except Exception:
            continue
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                if _normalise(text) == key:
                    return True
    return False


def table_as_text(table: Element, act_topic: list[Element]) -> dict | None:
    """A Production table Staging no longer prints as a table, but as one line
    per row - the row's label, a dash or colon, then its description.

    Only the label-then-description form counts: that is a table turned into
    text. Rows merely printed somewhere as loose words are more often a table
    one extraction failed to read as a grid, and are left to the other checks."""
    rows = [r for r in (table.cells or ())[1:] if r and r[0] and len(r) >= 2]
    if len(rows) < 2:
        return None
    lines, labels = [], []
    for row in rows:
        label = re.escape(row[0])
        for el in act_topic:
            if el.kind != KIND_TABLE and re.match(label + _LABEL_LEAD_RE, (el.text or "").strip(), re.IGNORECASE):
                lines.append(el)
                labels.append(row[0])
                break
    if len(lines) < max(2, (len(rows) + 1) // 2):
        return None
    header = " / ".join(c for c in table.cells[0] if c)
    shown = ", ".join(f"“{w}”" for w in labels[:6]) + (" …" if len(labels) > 6 else "")
    return {
        "type": "table-as-text", "kind": KIND_TABLE,
        "summary": (f"Table missing in Staging — the “{header}” table ({len(rows)} rows: {shown}) "
                    f"is printed as plain text lines in Staging."),
        "detail": "", "exp": table, "act": merge_elements(lines),
    }


def table_row_changes(a: Element, b: Element, exp_units: Counter, act_units: Counter) -> list[dict]:
    """Rows missing or added, rows and cells merged or split, changed cells and a
    changed column count, between Production's table `a` and Staging's `b`."""
    units_a = [_cell_units(_row_text(r)) for r in a.cells]
    units_b = [_cell_units(_row_text(r)) for r in b.cells]
    rows_a = [i for i, u in enumerate(units_a) if u]
    rows_b = [j for j, u in enumerate(units_b) if u]
    if not rows_a or not rows_b:
        return []
    pairs = _pair_rows(a, b, rows_a, rows_b, units_a, units_b)
    shorter = min(len(rows_a), len(rows_b))
    if shorter >= 2 and len(pairs) < _ROWS_RELIABLE * shorter:
        # The two grids were read too differently to trust a row-by-row
        # match - but that is exactly the situation a genuine row loss/gain
        # tends to cause (a table missing several rows in Staging often also
        # reads badly row-by-row), and returning nothing here left the reader
        # with no signal at all that anything was wrong. A plain row COUNT
        # still holds even when pairing does not, so it is reported as a
        # fallback rather than staying silent - Production's own row count is
        # the one Staging must not fall short of, and any extra beyond it is
        # itself worth flagging, not just a shortfall.
        if len(rows_a) != len(rows_b):
            return [{
                "type": "table-row-count", "kind": KIND_TABLE,
                "summary": (
                    f"Table row count differs — Production has {len(rows_a)} rows, "
                    f"Staging has {len(rows_b)} (rows could not be matched one-to-one to say which)."
                ),
                "detail": "", "exp": a, "act": b,
            }]
        return []

    outside_a = exp_units - sum((units_a[i] for i in rows_a), Counter())
    outside_b = act_units - sum((units_b[j] for j in rows_b), Counter())
    free_a = [i for i in rows_a if i not in pairs]
    free_b = [j for j in rows_b if j not in set(pairs.values())]
    out: list[dict] = []

    def page_of(el: Element, row: int) -> int | None:
        return el.row_boxes[row][0] if row < len(el.row_boxes) else None

    def header_repeat(el: Element, units: list, r: int) -> bool:
        return 0 < r < len(units) and units[r] == units[0]

    def next_piece(el: Element, units: list, free: list[int], r: int) -> int | None:
        """The next row that can continue row `r`: the very next one, or the
        first after the header rows a new page repeats in between."""
        k = r + 1
        while k < len(units) and header_repeat(el, units, k) and page_of(el, k) != page_of(el, r):
            k += 1
        return k if k in free else None

    def at_page_break(el: Element, units: list, r: int) -> bool:
        """Row `r` is the last row before, or the first after, a page break -
        where a row cut by the break reads as a split or merged row."""
        page = page_of(el, r)
        if page is None:
            return False
        before = r - 1
        while before > 0 and header_repeat(el, units, before):
            before -= 1
        after = r + 1
        while after < len(units) and header_repeat(el, units, after):
            after += 1
        return (0 < before and page_of(el, before) not in (None, page)) \
            or (after < len(units) and page_of(el, after) not in (None, page))

    for i in list(free_a):
        for j in free_b:
            k = next_piece(b, units_b, free_b, j)
            # A split needs BOTH pieces to make up Production's row: when one
            # piece alone already is that row, the other is a separate row.
            if k is not None and _similar(units_a[i], units_b[j] + units_b[k]) >= _ROW_MERGE \
                    and _similar(units_a[i], units_b[j]) < _ROW_MERGE and _similar(units_a[i], units_b[k]) < _ROW_MERGE:
                if page_of(b, j) != page_of(b, k) or at_page_break(a, units_a, i) \
                        or at_page_break(b, units_b, j) or at_page_break(b, units_b, k):
                    # One row continued over a page break (its first cell left
                    # empty under the repeated header) is still one row - and
                    # a row beside a break on either side is cut by where the
                    # page ends (a wrap), not split in the table itself.
                    free_a.remove(i)
                    free_b.remove(j)
                    free_b.remove(k)
                    break
                out.append({
                    "type": "table-merge", "kind": KIND_TABLE,
                    "summary": (f"Table merge issue — Staging splits Production's row {i + 1} "
                                f"(“{_clip_row(a.cells[i])}”) into two rows."),
                    "detail": "", "exp": _row_element(a, [i]), "act": _row_element(b, [j, k]),
                })
                free_a.remove(i)
                free_b.remove(j)
                free_b.remove(k)
                break
    for j in list(free_b):
        for i in free_a:
            k = next_piece(a, units_a, free_a, i)
            if k is not None and page_of(a, i) != page_of(a, k) \
                    and _similar(units_a[i] + units_a[k], units_b[j]) >= _ROW_MERGE:
                # Production's row continued over its own page break: one row.
                free_b.remove(j)
                free_a.remove(i)
                free_a.remove(k)
                break
            if i + 1 in free_a and _similar(units_a[i] + units_a[i + 1], units_b[j]) >= _ROW_MERGE \
                    and _similar(units_a[i], units_b[j]) < _ROW_MERGE and _similar(units_a[i + 1], units_b[j]) < _ROW_MERGE:
                if at_page_break(a, units_a, i) or at_page_break(a, units_a, i + 1) \
                        or at_page_break(b, units_b, j):
                    # Rows cut apart by a page break on one side: a wrap.
                    free_b.remove(j)
                    free_a.remove(i)
                    free_a.remove(i + 1)
                    break
                out.append({
                    "type": "table-merge", "kind": KIND_TABLE,
                    "summary": (f"Table merge issue — Staging merges Production's rows {i + 1} and {i + 2} "
                                f"(“{_clip_row(a.cells[i])}”) into one row."),
                    "detail": "", "exp": _row_element(a, [i, i + 1]), "act": _row_element(b, [j]),
                })
                free_b.remove(j)
                free_a.remove(i)
                free_a.remove(i + 1)
                break

    def printed_outside(units: Counter, outside: Counter) -> bool:
        return sum((units & outside).values()) >= _ROW_PRINTED_ELSEWHERE * sum(units.values())

    # A row one extraction divided differently is printed across the other
    # table's unpaired rows; a row really lost is printed nowhere.
    pool_a = outside_a + sum((units_a[i] for i in free_a), Counter())
    pool_b = outside_b + sum((units_b[j] for j in free_b), Counter())
    for i in free_a:
        if not printed_outside(units_a[i], pool_b):
            out.append({
                "type": "table-row-missing", "kind": KIND_TABLE,
                "summary": f"Table row missing in Staging — row {i + 1}: “{_clip_row(a.cells[i])}”.",
                "detail": "", "exp": _row_element(a, [i]), "act": b,
            })
    for j in free_b:
        if not printed_outside(units_b[j], pool_a):
            out.append({
                "type": "table-row-added", "kind": KIND_TABLE,
                "summary": f"Table row added in Staging — row {j + 1}: “{_clip_row(b.cells[j])}”.",
                "detail": "", "exp": a, "act": _row_element(b, [j]),
            })

    cell_changes: list[tuple] = []
    sequence_changes: list[tuple] = []
    merged_rows: list[tuple[int, int, int, int]] = []
    for i, j in sorted(pairs.items()):
        cells_a, cells_b = _cells_in(a, i), _cells_in(b, j)
        if cells_a != cells_b:
            # A raw grid can flag one side's cell as a colspan placeholder
            # (`spans`) that the other side's extraction of the SAME row did
            # not, purely as a quirk of how each page's rules were traced -
            # not a real merge or split, which always changes what a cell
            # actually holds. The row's own cells, exactly as printed, are the
            # only test that cannot be fooled by a `spans` count alone.
            if a.cells[i] == b.cells[j]:
                continue
            merged_rows.append((i, j, cells_a, cells_b))
            continue
        if len(a.cells[i]) != len(b.cells[j]):
            continue
        if units_a[i] == units_b[j]:
            # The row's words match overall - unless a cell re-sequenced them,
            # which counting words alone can never see (same multiset either way).
            if not any(_reordered_items(x, y) for x, y in zip(a.cells[i], b.cells[j])):
                continue
        other_a = sum((units_a[x] for x in rows_a if x != i), Counter())
        other_b = sum((units_b[y] for y in rows_b if y != j), Counter())
        left_a, left_b = units_a[i] - units_b[j], units_b[j] - units_a[i]
        reordered_row = any(_reordered_items(x, y) for x, y in zip(a.cells[i], b.cells[j]))
        if not reordered_row and (not left_a or _spilled(left_a, other_b)) and (not left_b or _spilled(left_b, other_a)):
            continue  # header or neighbouring text the extraction put in this row
        # A word straddling a column boundary ("LED" read as "L" | "ED status")
        # reads as one cell missing a word and its neighbour gaining one, even
        # though the row prints the same characters either way - compare the
        # row with cell boundaries erased before trusting a per-cell split.
        if "".join(a.cells[i]).replace(" ", "") == "".join(b.cells[j]).replace(" ", ""):
            continue
        changed = [(k, x, y) for k, (x, y) in enumerate(zip(a.cells[i], b.cells[j]))
                   # A pure reordering keeps the same words, so the same word
                   # COUNTS either way - `_cell_units` alone never flags it.
                   if _cell_units(_bare_xref(x)) != _cell_units(_bare_xref(y)) or _reordered_items(x, y)]
        # Same items, printed in a different order - not a missing/added/changed
        # word, so kept out of the ordinary "table-cell" diff and reported as
        # its own "the rows/options are sequenced differently" issue instead.
        reordered = [(k, x, y) for k, x, y in changed if _reordered_items(x, y)]
        changed = [c for c in changed if c not in reordered]
        if changed:
            cell_changes.append((i, j, changed))
        if reordered:
            sequence_changes.append((i, j, reordered))
    # A cell's own un-decodable character (a CJK/Arabic OSD language list read
    # back as raw "(cid:N)" tokens is where this turns up most) is worth
    # flagging on its own, independently of whatever `cell_changes` above made
    # of the same cell - a table cell comparison can fold a garbled cell into
    # "changed" or, if the two sides otherwise still look equal enough, drop
    # it entirely, and either way loses the fact that it does not decode.
    seen_encoding: set[str] = set()
    for i, j in sorted(pairs.items()):
        if i >= len(a.cells) or j >= len(b.cells):
            continue
        for cell_a, cell_b in zip(a.cells[i], b.cells[j]):
            enc = _encoding_regression(KIND_TABLE, cell_a, cell_b)
            if enc and enc["summary"] not in seen_encoding:
                seen_encoding.add(enc["summary"])
                enc["exp"], enc["act"] = _row_element(a, [i]), _row_element(b, [j])
                out.append(enc)
    if merged_rows:
        # One issue for the table: the same merge decision printed on many rows
        # (a spanned Extension column, a status column split in two) reads as
        # one change, boxed on the whole table on both sides.
        merged = sum(1 for _, _, ca, cb in merged_rows if cb < ca)
        split = len(merged_rows) - merged
        rows_shown = ", ".join(str(i + 1) for i, _, _, _ in merged_rows[:8]) + (" …" if len(merged_rows) > 8 else "")
        how = (f"{merged} row{'s' if merged != 1 else ''} with cells merged" if merged else "") + \
              (" and " if merged and split else "") + \
              (f"{split} row{'s' if split != 1 else ''} with cells split" if split else "")
        out.append({
            "type": "table-merge", "kind": KIND_TABLE,
            "summary": (f"Table merge issue — {how} in Staging compared with Production "
                        f"({'row' if len(merged_rows) == 1 else 'rows'} {rows_shown})."),
            "detail": "", "exp": a, "act": b,
        })
    # A grid read one row out of step shows a "changed" cell in nearly every
    # row - deliberately NOT consolidated into one "table layout issue" and
    # dropped from the per-row report: row/cell content missing or changed
    # matters more than the risk of it being a read/layout artefact, so every
    # row still gets its own "table-cell" finding below.
    def _column_name(k: int) -> str:
        if a.headers and len(a.headers) == len(a.cells[0] if a.cells else ()) and k < len(a.headers):
            return f"“{a.headers[k]}”"
        return f"column {k + 1}"

    def _cell_change(k: int, x: str, y: str) -> str:
        """What differs in one cell, quoted: the differing words, never the
        first 40 characters of each side - two long cells that start alike
        read as "“Confirm if there was a power outage. • Con” →" the same text
        while the real change, a sub-item Staging dropped, sat past the clip."""
        x, y = x or "", y or ""
        where = _column_name(k)
        if x and y and x != y and x.replace(" ", "") == y.replace(" ", ""):
            side = "Production" if x.count(" ") > y.count(" ") else "Staging"
            other = "Staging" if side == "Production" else "Production"
            return (f"{where}: “{x[:60]}” changed to “{y[:60]}” — {side} has a space "
                    f"{other} does not")
        wa, wb = x.split(), y.split()
        bare = lambda w: _PUNCT_RE.sub("", w)  # noqa: E731
        is_marker = lambda w: w in _CELL_MARKERS or bool(_CALLOUT_LABEL_RE.fullmatch(w))  # noqa: E731
        pieces: list[str] = []
        swaps: set[tuple[str, str]] = set()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=[bare(w) or w for w in wa],
                                                          b=[bare(w) or w for w in wb],
                                                          autojunk=False).get_opcodes():
            if op == "equal":
                continue
            ra, rb = wa[i1:i2], wb[j1:j2]
            ma, mb = [w for w in ra if is_marker(w)], [w for w in rb if is_marker(w)]
            if ma and mb and ma[0] != mb[0]:
                swaps.add((ma[0], mb[0]))
            ra, rb = [w for w in ra if not is_marker(w)], [w for w in rb if not is_marker(w)]
            if not ra and not rb:
                continue
            if ra and rb:
                pieces.append(f"“{' '.join(ra)[:60]}” changed to “{' '.join(rb)[:60]}”")
            elif ra:
                pieces.append(f"“{' '.join(ra)[:60]}” missing in Staging")
            else:
                pieces.append(f"“{' '.join(rb)[:60]}” added in Staging")
        for pm, sm in sorted(swaps):
            pieces.append(f"list markers “{pm}” in Production, “{sm}” in Staging")
        if not pieces:
            # Only punctuation or spacing differs: name the marks, not two
            # clips of the same words.
            marks = mark_changes(x, y)
            if marks:
                return f"{where}: " + "; ".join(marks[:4])
            return f"{where}: “{(x or '—')[:60]}” → “{(y or '—')[:60]}”"
        return f"{where}: " + "; ".join(pieces[:4]) + (" …" if len(pieces) > 4 else "")

    for i, j, changed in cell_changes:
        shown = "; ".join(_cell_change(k, x, y) for k, x, y in changed[:3])
        # The row's own printed words, not just its first filled cell: in a
        # table whose first column is merged over several rows that cell is the
        # codec ("4K"), which named the row by a fragment.
        label = _clip_row(a.cells[i], 50) if i < len(a.cells) else ""
        row_name = f"the “{label}” row" if label else f"row {i + 1}"
        out.append({
            "type": "table-cell", "kind": KIND_TABLE,
            "summary": f"Table cell changed in {row_name} — {shown}.",
            "detail": "", "exp": _row_element(a, [i]), "act": _row_element(b, [j]),
            "cells": [(i, j, k, x, y) for k, x, y in changed],
        })

    for i, j, changed in sequence_changes:
        label = _clip_row(a.cells[i], 50) if i < len(a.cells) else ""
        row_name = f"the “{label}” row" if label else f"row {i + 1}"
        cols = ", ".join(_column_name(k) for k, x, y in changed[:3])
        out.append({
            "type": "table-cell-sequence", "kind": KIND_TABLE,
            "summary": (f"Sequence changed in {row_name} — {cols}: the same items are "
                        f"printed in a different order in Staging."),
            "detail": "", "exp": _row_element(a, [i]), "act": _row_element(b, [j]),
            "cells": [(i, j, k, x, y) for k, x, y in changed],
        })

    # Printed columns: a column of icons is a column the reader sees; a column
    # the ruling split in two is not two (see `_printed_columns`).
    cols_a = _printed_columns(a) + (1 if a.icon_column else 0)
    cols_b = _printed_columns(b) + (1 if b.icon_column else 0)
    if cols_a != cols_b:
        def named(t: Element) -> str:
            shown = [f"“{h}”" for h in t.headers[:6]]
            if t.icon_column:
                shown.insert(0, "an icon column")
            return ", ".join(shown) or "no header"
        where = ""
        if a.icon_column != b.icon_column:
            own, inside = ("Production", "Staging") if a.icon_column else ("Staging", "Production")
            where = f" {own} prints the icons in a column of their own; {inside} prints them inside the first column's cells."
        out.append({
            "type": "table-columns", "kind": KIND_TABLE,
            "summary": (f"Table cell issue — Production has {cols_a} columns ({named(a)}); "
                        f"Staging has {cols_b} columns ({named(b)}).{where}"),
            "detail": "", "exp": a, "act": b,
        })
    return out


# A cross-reference in brackets: "(See page 51)" - "( )" once its page number
# is taken out - or Staging's "(See “Quick access to setting menu”)", the same
# reference turned into a link naming its section instead of its page.
_XREF_RE = re.compile(r"\(\s*(?:see\s*)?(?:[\"“”'‘’][^\"“”'‘’]*[\"“”'‘’])?\s*\)", re.IGNORECASE)


_XREF_HEAD_RE = re.compile(r"\(\s*see\s*[\"“][^\"“”)]*$", re.IGNORECASE)   # an element ending inside one
_XREF_TAIL_RE = re.compile(r"^[^\"“”()]*[\"”]\s*\)")                   # an element starting inside one


def _bare_xref(cell: str) -> str:
    """A cell with its bracketed cross-reference reduced to "()": Staging
    naming the section where Production named the page is the expected
    change of a page reference into a link, not new wording. A callout
    label's own colon ("Note" / "Note:") is its styling and goes too."""
    return _CALLOUT_COLON_RE.sub(r"\1", _XREF_RE.sub("()", cell or ""))


_CALLOUT_COLON_RE = re.compile(
    r"\b(" + "|".join(sorted({re.escape(w) for w in i18n._CALLOUT_WORDS}, key=len, reverse=True)) + r")\s*[:：]",
    re.IGNORECASE,
)


_ICON_LABEL_MIN_SIDE = 10.0  # pt: a cell's printed artwork at least this big is an icon, not a bare glyph


def _drop_labels_drawn_as_icons(diffs: list[dict], a: Element, b: Element,
                                expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A changed cell whose only difference is a short label ("1", "2") that
    Staging prints inside an icon image - the same icon, its number baked into
    the picture - is not a changed cell. Confirmed by finding Production's
    printed cell artwork in Staging's row on the rendered page."""
    return [d for d in diffs
            if not (d.get("type") == "table-cell" and d.get("cells")
                    and all(_icon_label_cell(a, b, *cell, expected, actual) for cell in d["cells"]))]


def _icon_label_cell(a: Element, b: Element, i: int, j: int, k: int, x: str, y: str,
                     expected: fitz.Document, actual: fitz.Document) -> bool:
    words_a, words_b = (x or "").split(), (y or "").split()
    gone = [w for w in words_a if w not in words_b]
    if not gone or any(w not in words_a for w in words_b) or any(len(_PUNCT_RE.sub("", w)) > 2 for w in gone):
        return False
    if i >= len(a.row_boxes) or j >= len(b.row_boxes) or i >= len(a.grid) or k >= len(a.grid[i]) or not a.grid[i][k]:
        return False
    page_a, row_a = a.row_boxes[i]
    page_b, row_b = b.row_boxes[j]
    cx0, cx1 = a.grid[i][k]
    return visual_diff.find_artwork(expected, page_a, (cx0, row_a[1], cx1, row_a[3]),
                                    actual, [(page_b, fitz.Rect(row_b) + (-4, -4, 4, 4))],
                                    min_side=_ICON_LABEL_MIN_SIDE) is not None


def settle_tables(diffs: list[dict], exp_el: Element | None, act_el: Element | None,
                  exp_units: Counter, act_units: Counter) -> list[dict]:
    """Drop a table difference that is only the two extractions dividing the
    same content differently.

    Whatever one table holds and the other lacks must be printed elsewhere in
    the other document's copy of the chapter: Staging's Taiwan RoHS table
    carries its title and its notes inside the grid, where Production prints
    them as text above and below it. A row really lost from Staging's table is
    printed nowhere else, and is still reported.
    
    When tables are restructured (Production: 4 tables, Staging: 2 tables merged),
    drop "row added/missing" errors if the row's content exists anywhere in the
    other document's complete table pool, not just within paired tables."""
    out: list[dict] = []
    for d in diffs:
        dtype = d.get("type")
        
        # Handle "table-cell" and "table-shape" as before
        if dtype in ("table-cell", "table-shape"):
            if exp_el is None or act_el is None or exp_el.kind != KIND_TABLE or act_el.kind != KIND_TABLE:
                out.append(d)
                continue
            mine, theirs = _text_units(_cells_text(exp_el)), _text_units(_cells_text(act_el))
            if (mine - theirs) - (act_units - theirs) or (theirs - mine) - (exp_units - mine):
                out.append(d)
            continue
        
        # Handle "table-row-missing" and "table-row-added": check if row content exists
        # anywhere in the other document's entire table pool (not just paired table)
        if dtype in ("table-row-missing", "table-row-added"):
            if dtype == "table-row-missing":
                # Production row missing in Staging: check if it's in act_units (entire Staging pool)
                el = d.get("exp")
                if el is not None:
                    row_units = _text_units(_row_text(el.cells[0]) if el.cells else "")
                    # A Counter has no ordering (`<=` between two Counters raises
                    # TypeError) - subset is "nothing left over once act_units'
                    # own counts are subtracted out", the same idiom used above.
                    if row_units and not (row_units - act_units):
                        # Row's content found in Staging's tables, skip this false positive
                        continue
            else:  # dtype == "table-row-added"
                # Staging row added: check if it's in exp_units (entire Production pool)
                el = d.get("act")
                if el is not None:
                    row_units = _text_units(_row_text(el.cells[0]) if el.cells else "")
                    if row_units and not (row_units - exp_units):
                        # Row's content found in Production's tables, skip this false positive
                        continue
        
        out.append(d)
    return out


def settle_content(diffs: list[dict], exp_words: str, act_words: str,
                   exp_el: Element | None = None, act_el: Element | None = None,
                   exp_units: Counter | None = None, act_units: Counter | None = None,
                   exp_keys: set[str] | None = None, act_keys: set[str] | None = None,
                   exp_figures: list[Element] | None = None, act_figures: list[Element] | None = None,
                   exp_tables: list[Element] | None = None, act_tables: list[Element] | None = None,
                   exp_spare_words: str | None = None, act_spare_words: str | None = None,
                   exp_spare_units: Counter | None = None, act_spare_units: Counter | None = None) -> list[dict]:
    """Drop missing/extra content the other document prints elsewhere in the SAME
    heading's own topic - never a neighbouring or wider chapter's content.

    A run of three or more words that Staging prints in its next paragraph is
    not missing - the paragraphs were only divided differently. One- and
    two-word runs are kept: they are the real edits ("to the any" -> "to any"),
    and common words are found everywhere.

    The same goes for a whole paragraph, label or table that one side prints and
    the other seems not to: when the other document's copy of THIS topic prints
    it too, it was only divided up differently.

    Checked only against the element's own TOPIC (this heading's own words) -
    a wider, chapter-wide or cross-heading search was tried and dropped: it can
    settle a finding using content that belongs to a different heading, which is
    never correct even when it happens to share wording."""
    out: list[dict] = []
    for d in diffs:
        kind = d.get("type")
        if kind in ("missing", "added") and d.get("kind") != KIND_FIGURE:
            el = exp_el if kind == "missing" else act_el
            # Content a normal, both-sided pair already accounts for is spoken
            # for: settling THIS finding with it too is what let a genuine
            # duplicate (Staging printing an entire numbered procedure twice)
            # excuse itself by pointing at the very Production text its
            # legitimately-matched twin was already paired with. Falls back to
            # the full topic words when the caller has not computed the
            # spare-only ones (keeps this callable exactly as before).
            if kind == "missing":
                words = act_spare_words if act_spare_words is not None else act_words
                units = act_spare_units if act_spare_units is not None else act_units
            else:
                words = exp_spare_words if exp_spare_words is not None else exp_words
                units = exp_spare_units if exp_spare_units is not None else exp_units
            own_figures = (exp_figures if kind == "missing" else act_figures) or []
            other_tables = (act_tables if kind == "missing" else exp_tables) or []
            # A short label captioning a nearby icon ("WEEE", "Battery") is
            # never settled this way: the word merely appearing SOMEWHERE in
            # the topic's own running prose ("...equipment and/or Battery...")
            # is not evidence the icon still carries its own caption - unlike
            # a genuinely relocated label ("Deutsch" beside its paragraph on
            # one side, inside it on the other), losing an icon's caption is
            # a real difference a reader would notice. But a whole cell of an
            # actual TABLE on the other side is not a coincidence the way a
            # stray word in running prose is: a label-icon-description row one
            # side grids and the other reflows into plain paragraphs still
            # opens with the very same label, so it is the icon that was
            # dropped, not the caption.
            caption = (el is not None and el.text and len(el.text) <= _LAYOUT_SHORT_CHARS
                       and _near_a_figure(el, own_figures, reach=_HEADING_CAPTION_REACH)
                       and not _matches_table_row(el.text, other_tables, min_ratio=0.85))
            if el is not None and not caption and (
                _printed_in(el, words, units) or _matches_table_row(el.text, other_tables)
            ):
                continue
            out.append(d)
            continue
        if kind != "text":
            out.append(d)
            continue
        # A run the other side prints as a line of its own is not lost: Production
        # glues "KONFORMITÄTSERKLÄRUNG" onto the paragraph under it, Staging sets
        # it apart as a heading.
        # A callout's label ("Warning", "WARNING:", "Note") is its styling, not
        # its content: Staging sets it in its own callout box, often as a label
        # the text extraction keeps apart. Bullet glyphs are never words.
        # A misspelling on one side ("SmartRemoote" / "SmartRemote") is one
        # changed word: both halves stay together, even when the correct
        # spelling is printed elsewhere on the other side - filtering one half
        # alone reported the typo as a word "missing".
        spelled = _spelling_pairs(d.get("gone", []), d.get("extra", []))
        gone = [r for r in d.get("gone", []) if r in spelled or (
                not _elsewhere_counted(r, exp_words, act_words) and r not in (act_keys or ())
                and not _BARE_CALLOUT_RE.match(r))]
        extra = [r for r in d.get("extra", []) if r in spelled.values() or (
                 not _elsewhere_counted(r, act_words, exp_words) and r not in (exp_keys or ())
                 and not _BARE_CALLOUT_RE.match(r))]
        if not gone and not extra:
            # Nothing of the sentence's own WORDING is genuinely lost once its
            # callout label (never content on its own, just above) is taken
            # out of the count - but the same sentence can still differ by
            # more than wording: a cross-reference dropping the quote marks
            # that set its target apart when it becomes a hyperlink, say.
            # Neither `_content_runs`'s word-level view (quotes are not
            # words) nor the label filter just above can see that, so it is
            # worth one more look here - the label and any page-number cross-
            # reference stripped from BOTH sides first, so what triggered
            # this whole finding is not what gets re-reported by it - before
            # the difference is lost for good along with the label word that
            # brought it to this function in the first place.
            if exp_el is not None and act_el is not None and exp_el.text and act_el.text:
                exp_bare = _LEADING_CALLOUT_RE.sub("", _PAGE_REF_RE.sub(" ", exp_el.text))
                act_bare = _LEADING_CALLOUT_RE.sub("", _PAGE_REF_RE.sub(" ", act_el.text))
                wd2 = _word_diff(exp_bare, act_bare)
                changed_text = " ".join(op["text"] for op in wd2 if op["type"] != "equal")
                # Whole words in the change mean the sentence was split or moved
                # into a neighbouring block - not a punctuation change; any word
                # truly gone is the word checks' finding.
                if changed_text and not _is_trivial_punct(changed_text) \
                        and not _only_bullet_glyphs(changed_text) \
                        and mark_changes(exp_bare, act_bare, exp_el.wraps, act_el.wraps):
                    out.append({
                        "type": "symbol" if _has_symbol(changed_text) else "punctuation",
                        "kind": d.get("kind"),
                        "summary": (
                            "Symbol changed — the wording is the same, but a symbol or mark differs."
                            if _has_symbol(changed_text) else
                            "Punctuation changed — the wording is the same, only punctuation differs."
                        ),
                        "detail": "",
                        "word_diff": wd2,
                    })
            continue
        d["gone"], d["extra"] = gone, extra
        d["summary"] = _content_summary(d.get("kind"), gone, extra)
        if d.get("word_diff"):
            # The boxes follow the summary: a run settled as printed further on
            # (a wrapped last bullet set as its own lines) is not boxed either.
            keep_del = [_normalise(r) for r in gone]
            keep_ins = [_normalise(r) for r in extra]

            def kept(op: dict) -> bool:
                runs = keep_del if op["type"] == "del" else keep_ins
                text = _normalise(op["text"])
                size = len(text.split()) or 1
                # Whole words, covering most of the op: "the" must not keep a
                # whole sentence that happens to contain "the".
                return any(
                    r and (f" {r} " in f" {text} " or f" {text} " in f" {r} ")
                    and len(r.split()) >= 0.5 * size
                    for r in runs
                )

            d["word_diff"] = [
                op if op["type"] == "equal" or kept(op) else {"type": "equal", "text": op["text"]}
                for op in d["word_diff"] if op["type"] != "ins" or kept(op)
            ]
        out.append(d)
    return out


def _unsplit(joined: "Counter[str]", parts: "Counter[str]", other_key: str) -> None:
    """Cancel a word in `joined` that is two adjacent words of `other_key` run
    together, and those two words in `parts`."""
    tokens = _TOKEN_RE.findall(other_key)
    for first, second in zip(tokens, tokens[1:]):
        word = first + second
        if joined.get(word, 0) > 0 and parts.get(first, 0) > 0 and parts.get(second, 0) > 0:
            joined[word] -= 1
            parts[first] -= 1
            parts[second] -= 1
    for counter in (joined, parts):
        for key in [k for k, v in counter.items() if v <= 0]:
            del counter[key]


def _phrases(key: str, words: "Counter[str]", limit: int = 12) -> list[str]:
    """The `words` in the order `key` prints them, consecutive ones kept together."""
    remaining = Counter(words)
    out: list[str] = []
    run: list[str] = []
    for token in _TOKEN_RE.findall(key):
        if remaining.get(token, 0) > 0:
            remaining[token] -= 1
            run.append(token)
        elif run:
            out.append(" ".join(run))
            run = []
    if run:
        out.append(" ".join(run))
    return out[:limit]


def _bold_missing(a: Element, b: Element) -> list[str]:
    """Phrases set bold in Production whose words are in Staging but not bold."""
    prod_bold = Counter(a.bold_words)
    stage_bold = Counter(b.bold_words)
    stage_words = _tokens(b.key)
    lost: Counter = Counter()
    for word, count in prod_bold.items():
        if len(word) < 2 or not stage_words.get(word):
            continue
        shortfall = min(count, stage_words[word]) - stage_bold.get(word, 0)
        if shortfall > 0:
            lost[word] = shortfall
    if not lost:
        return []
    # Walk Production's bold words in order so the report names phrases.
    out: list[str] = []
    run: list[str] = []
    for word in a.bold_words:
        if lost.get(word, 0) > 0:
            lost[word] -= 1
            run.append(word)
        elif run:
            out.append(" ".join(run))
            run = []
    if run:
        out.append(" ".join(run))
    return out


def _underline_missing(a: Element, b: Element) -> list[str]:
    """Phrases underlined in `a` whose words are in `b` but not underlined -
    the same walk as `_bold_missing`, over `underline_words` instead.

    A URL or a link's visible text is excluded: a working hyperlink is
    already judged, styling and all, by `links.py` - which treats colour OR
    underline as enough to mark it clickable - and reporting every footnote
    URL underlined in Production but merely coloured in Staging here would be
    the same finding twice, the second time without that tolerance. A long
    URL wrapped onto two lines can also leave one or two of its own words
    reading as a short run of their own (the drawn rule under a wrapped
    continuation does not always start exactly where the text does) - a
    fragment that small is that seam, not a deliberate one- or two-word
    emphasis, so only a run of three words or more is kept.
    """
    a_underlined = Counter(a.underline_words)
    b_underlined = Counter(b.underline_words)
    b_words = _tokens(b.key)
    lost: Counter = Counter()
    for word, count in a_underlined.items():
        if len(word) < 2 or not b_words.get(word):
            continue
        shortfall = min(count, b_words[word]) - b_underlined.get(word, 0)
        if shortfall > 0:
            lost[word] = shortfall
    if not lost:
        return []
    out: list[str] = []
    run: list[str] = []
    for word in a.underline_words:
        if lost.get(word, 0) > 0:
            lost[word] -= 1
            run.append(word)
        elif run:
            if len(run) >= 3 and not _URL_LIKE_WORDS & set(run):
                out.append(" ".join(run))
            run = []
    if run and len(run) >= 3 and not _URL_LIKE_WORDS & set(run):
        out.append(" ".join(run))
    return out


# --- bold, as it prints -----------------------------------------------------
#
# Whether a phrase is bold is decided by how dark it PRINTS, not by the font's
# weight. Production sets a phrase in "Bold" and Staging in "Semibold" or a
# synthetic bold, and the font flag says "not bold" about text that reads just
# as dark on the page - so a flag difference is only a candidate. Each
# candidate phrase is rendered on both pages and its strokes measured, and it
# is reported only when Staging's strokes are clearly thinner.
_BOLD_ZOOM = 4.0        # ~290 dpi: a stroke is several pixels wide
_INK_LEVEL = 150        # grey below this is ink
# Staging's strokes must be at least this share of Production's to read as
# bold too - low enough to still catch a real, if modest, weight increase
# (a regular font traded for a medium one, not just Regular-vs-Bold), high
# enough that two renders of the very same weight, which differ only by a few
# percent from anti-aliasing, are not called a difference.
_BOLD_SIMILAR = 0.90


_BOLD_MATCH_OVERLAP = 0.5  # a Staging line must share this much of the Production line's words
_BOLD_SHORT_LINE_RATIO = 0.6  # the shorter line's words, as a share of the longer's, below which
                              # the shorter line is judged on ITS OWN coverage instead


def _phrase_hits(doc: fitz.Document, el: Element, phrase: str) -> list[tuple[int, "fitz.Rect"]]:
    """Every (page, rect) where `phrase` prints inside this element. "wi fi" is
    the tokenised form of a printed "Wi-Fi", so the hyphenated form is tried too."""
    for attempt in (phrase, phrase.replace(" ", "-")):
        hits: list[tuple[int, fitz.Rect]] = []
        for page_index, bbox in el.boxes:
            try:
                found = doc[page_index].search_for(attempt, clip=fitz.Rect(bbox) + (-3, -3, 3, 3))
            except Exception:
                continue
            hits.extend((page_index, r) for r in found)
        if hits:
            return hits
    return []


def _line_words(doc: fitz.Document, page_index: int, rect) -> set[str]:
    """The words of the printed line `rect` sits on."""
    middle = (rect.y0 + rect.y1) / 2
    words: set[str] = set()
    for lb, text in _text_lines(doc, page_index):
        if lb[1] <= middle <= lb[3] and lb[0] < rect.x1 + 200 and lb[2] > rect.x0 - 200:
            words.update(_TOKEN_RE.findall(text.casefold()))
    return words


def _set_bold(doc: fitz.Document, page_index: int, rect) -> bool:
    """Is the text at `rect` set in a bold font?"""
    try:
        data = doc[page_index].get_text("dict", clip=fitz.Rect(rect) + (-1, -1, 1, 1))
    except Exception:
        return False
    return any(
        span.get("flags", 0) & _BOLD_FLAG and (span.get("text") or "").strip()
        for block in data.get("blocks", []) for line in block.get("lines", []) for span in line.get("spans", [])
    )


def _bold_places(expected: fitz.Document, exp_el: Element, actual: fitz.Document, act_el: Element,
                 phrase: str, limit: int = 3) -> list[tuple]:
    """(Production place, Staging place) for each place Production sets `phrase`
    in bold, paired with the Staging place whose printed line shares most of its
    words. Never simply the first time the word turns up on the other side: that
    measured a bold heading "Connection" against the link "USB connection"
    further down the page."""
    stage = [(at, _line_words(actual, *at)) for at in _phrase_hits(actual, act_el, phrase)]
    pairs: list[tuple] = []
    for a_at in _phrase_hits(expected, exp_el, phrase):
        if len(pairs) >= limit:
            break
        if not _set_bold(expected, *a_at):
            continue
        a_words = _line_words(expected, *a_at)
        best, best_score = None, 0.0
        for b_at, b_words in stage:
            score = len(a_words & b_words) / max(1, len(a_words | b_words))
            shorter, longer = sorted((len(a_words), len(b_words)))
            if shorter and shorter / max(1, longer) <= _BOLD_SHORT_LINE_RATIO:
                # Staging's line-wrap reflowed the phrase onto a line with far
                # fewer words of its own (a heading-length line split down to
                # "...and press Connect." on Production's side, "press Connect"
                # on Staging's) - the extra words only the LONGER line has must
                # not count against it, so it is judged on how much of the
                # SHORTER line's own words are shared, not the union of both.
                score = max(score, len(a_words & b_words) / shorter)
            if score > best_score:
                best, best_score = b_at, score
        if best is not None and best_score >= _BOLD_MATCH_OVERLAP:
            pairs.append((a_at, best))
    return pairs


def stroke_weight(doc: fitz.Document, page_index: int, rect) -> float | None:
    """How heavy the type in `rect` prints: its mean stroke thickness, as a share
    of the line's height so the type size cancels out. None when there is too
    little ink to measure."""
    try:
        import numpy as np

        pix = doc[page_index].get_pixmap(
            matrix=fitz.Matrix(_BOLD_ZOOM, _BOLD_ZOOM), clip=fitz.Rect(rect),
            colorspace=fitz.csGRAY, alpha=False,
        )
        if pix.width < 4 or pix.height < 4:
            return None
        grey = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
        ink = grey < _INK_LEVEL
        area = int(ink.sum())
        edges = int(np.count_nonzero(ink[:, 1:] != ink[:, :-1]) + np.count_nonzero(ink[1:, :] != ink[:-1, :]))
        if area < 30 or not edges:
            return None
        return (2.0 * area / edges) / pix.height
    except Exception:
        return None


def confirm_bold(diffs: list[dict], exp_el: Element | None, act_el: Element | None,
                 expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """Keep a "bold missing" candidate only for the phrases Staging prints
    visibly lighter than Production. A phrase that cannot be found on the page
    cannot be judged by how it prints, so it is not reported on the font's word
    alone."""
    out: list[dict] = []
    for diff in diffs:
        kind = diff.get("type")
        if kind not in ("bold-missing", "bold-added") or exp_el is None or act_el is None:
            out.append(diff)
            continue
        added = kind == "bold-added"
        # The side that sets the words bold, and the side measured against it.
        bold_doc, bold_el, plain_doc, plain_el = (
            (actual, act_el, expected, exp_el) if added else (expected, exp_el, actual, act_el)
        )
        kept: list[tuple[str, float]] = []
        bold_marks: list[tuple] = []
        plain_marks: list[tuple] = []
        for phrase in diff.get("phrases") or []:
            lightest = None
            for bold_at, plain_at in _bold_places(bold_doc, bold_el, plain_doc, plain_el, phrase):
                wb, wp = stroke_weight(bold_doc, *bold_at), stroke_weight(plain_doc, *plain_at)
                if not wb or not wp:
                    continue  # unmeasurable - not an issue either way
                # A pixel-measured ratio right at `_BOLD_SIMILAR`'s edge is not
                # trustworthy on its own when the two documents also set body
                # text at different point sizes overall (Production 12pt,
                # Staging 11pt, say) - a smaller size measures a few percent
                # heavier as a share of its own line height from anti-aliasing
                # alone, nothing to do with weight, and can mask a genuine bold
                # difference the printed page shows plainly. Staging's own
                # font simply not being the bold one Production's copy is is
                # already a definite answer in that case, not a rendering
                # nuance - trusted just as much as a clearly lighter pixel
                # measurement, never on its own when the pixels agree it
                # prints just as dark (a "Bold"-named font with no real bold
                # weight of its own must not be reported either).
                if wp / wb >= _BOLD_SIMILAR and _set_bold(plain_doc, *plain_at):
                    continue
                lightest = wp / wb if lightest is None else min(lightest, wp / wb)
                bold_marks.append((bold_at[0], tuple(bold_at[1])))
                plain_marks.append((plain_at[0], tuple(plain_at[1])))
            if lightest is not None:
                kept.append((phrase, lightest))
        if not kept:
            continue
        shown = ", ".join(f"“{p}”" for p, _ in kept[:8]) + (" …" if len(kept) > 8 else "")
        verb = "is" if len(kept) == 1 else "are"
        diff["phrases"] = [p for p, _ in kept]
        if added:
            diff["summary"] = f"Bold added in Staging — {shown} {verb} bold in Staging and regular in Production."
            diff["detail"] = f"Production's text prints {min(r for _, r in kept):.0%} as heavy as Staging's."
            diff["exp_marks"], diff["marks"] = plain_marks, bold_marks
        else:
            diff["summary"] = f"Bold missing in Staging — {shown} {verb} bold in Production and regular in Staging."
            diff["detail"] = f"Staging's text prints {min(r for _, r in kept):.0%} as heavy as Production's."
            # The exact words, so the viewer boxes them rather than their paragraph.
            diff["exp_marks"], diff["marks"] = bold_marks, plain_marks
        out.append(diff)
    return out


_SPACE_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def space_changes(exp_group: list[Element], act_group: list[Element]) -> list[dict]:
    """A space lost or added between two words - Staging's "Road 16Sector"
    where Production prints "16 Sector". The wording comparison rejoins such
    words as a wrap artefact, so they are looked for here, on the printed text:
    two adjacent words on one side printed as one word on the other, and never
    printed apart there. A word hyphenated at a line end is a wrap, not a space."""
    def words(group: list[Element]) -> list[str]:
        out: list[str] = []
        for el in group:
            if el.kind == KIND_FIGURE:
                continue
            for raw in (el.text or "").split():
                if _CJK_RE.search(raw):
                    continue
                out.append(raw)
        return out

    def core(token: str) -> str:
        return "".join(_SPACE_WORD_RE.findall(token)).casefold()

    wa, wb = words(exp_group), words(act_group)
    if not wa or not wb:
        return []
    joined_b = {core(t): t for t in wb if core(t)}
    joined_a = {core(t): t for t in wa if core(t)}
    pairs_a = {(core(x), core(y)) for x, y in zip(wa, wa[1:])}
    pairs_b = {(core(x), core(y)) for x, y in zip(wb, wb[1:])}
    found: list[tuple[str, str, str]] = []   # (direction, apart, together)
    for side_words, other_joined, other_pairs, direction in (
        (wa, joined_b, pairs_b, "missing"), (wb, joined_a, pairs_a, "added"),
    ):
        for x, y in zip(side_words, side_words[1:]):
            cx, cy = core(x), core(y)
            if not cx or not cy or x.endswith(("-", "‐", "‑")) or len(cx + cy) < 4:
                continue
            # Both halves must meet with no punctuation between them: "16 Sector".
            if not (x[-1].isalnum() and y[0].isalnum()):
                continue
            together = other_joined.get(cx + cy)
            if together and (cx, cy) not in other_pairs:
                found.append((direction, f"{x.rstrip('.,;:')} {y.rstrip('.,;:')}", together.rstrip(".,;:")))
    if not found:
        return []
    seen: set = set()
    out: list[dict] = []
    for direction, apart, together in found:
        if (direction, apart.casefold()) in seen:
            continue
        seen.add((direction, apart.casefold()))
        if direction == "missing":
            summary = f"Space missing in Staging — Production prints “{apart}”, Staging prints “{together}”."
            ops = [{"type": "del", "text": apart}, {"type": "ins", "text": together}]
        else:
            summary = f"Space added in Staging — Production prints “{together}”, Staging prints “{apart}”."
            ops = [{"type": "del", "text": together}, {"type": "ins", "text": apart}]
        out.append({"type": "text-space", "kind": KIND_TEXT, "summary": summary, "detail": "", "word_diff": ops})
    return out


_COVERAGE_MIN_TOKENS = 2    # shorter runs are the text checks' own business
# The two checks below excuse a run because its WORDS, as a bag, exist
# somewhere in the whole other document - no phrase, no order, nothing
# actually saying it is the SAME sentence - so for a short run this is nearly
# always true of pure coincidence (the ten most common words in any manual
# recur constantly) and would excuse genuinely misplaced content this very
# function exists to catch (content under the WRONG heading, invisible to
# every topic-scoped check above it). Trusted only once a run is long enough
# that its full word-for-word coverage elsewhere stops being a coincidence;
# a shorter run must still clear the exact-phrase check just above it.
_COVERAGE_BAG_MIN_TOKENS = 6
_PAGE_REF_WORDS = {"on", "page", "see", "refer", "to"}
_COVERAGE_REORDER_MAX_TOKENS = 8  # longer than this, reused words are not an excuse


@lru_cache(maxsize=64)
def _chapter_word_counts(doc: fitz.Document, pages: tuple) -> Counter:
    """Every word printed on this side's copy of a chapter, counted."""
    return Counter(_TOKEN_RE.findall(_normalise(
        " ".join(doc[p].get_text("text") for p in pages if 0 <= p < doc.page_count))))


def _topic_stream(elements: list[Element]) -> list[tuple[str, Element]]:
    """Everything a topic prints, word by word in reading order, with the
    element each word came from - paragraphs, list items, table cells, callout
    text. Callout labels and list markers are styling, not words."""
    out: list[tuple[str, Element]] = []
    for el in sorted(elements, key=lambda e: e.order):
        if el.kind == KIND_FIGURE:
            continue
        for token in _TOKEN_RE.findall(_normalise(_xref_free(el.text))):
            if _BARE_CALLOUT_RE.match(token):
                continue
            out.append((token, el))
    return out


def _xref_free(text: str | None) -> str:
    """Text with its cross-references taken out: Production's "(See page 47)"
    and Staging's link naming the section instead ("(See “Optimizing image
    quality by Auto Cinema mode”)") are the same reference - also when the
    text ends or starts part-way through one, or its lines are split."""
    text = _XREF_RE.sub(" ", _PAGE_REF_RE.sub(" ", text or ""))
    text = "\n".join(_XREF_HEAD_RE.sub(" ", _XREF_TAIL_RE.sub(" ", line)) for line in text.split("\n"))
    return _XREF_HEAD_RE.sub(" ", _XREF_TAIL_RE.sub(" ", text))


def _phrase_boxes(doc: fitz.Document, el: Element, printed: str) -> list[tuple[int, tuple]]:
    """Where `printed` sits inside the element: from its first words to its
    last, on the page they print on. The element's own boxes when not found."""
    words = printed.replace("…", " ").split()
    if not words:
        return list(el.boxes)
    head, tail = " ".join(words[:5]), " ".join(words[-5:])
    for page_index, bbox in el.boxes:
        try:
            clip = fitz.Rect(*bbox) + (-2, -2, 2, 2)
            first = doc[page_index].search_for(head, clip=clip)
            last = doc[page_index].search_for(tail, clip=clip) if tail != head else first
        except Exception:
            continue
        if first:
            rect = fitz.Rect(first[0])
            if last:
                rect |= fitz.Rect(last[-1])
            return [(page_index, (rect.x0, rect.y0, rect.x1, rect.y1))]
    return list(el.boxes)


def coverage_changes(chapter: "Chapter", topic: str, exp_topic: list[Element], act_topic: list[Element],
                     expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """The last word on content: every run of words Production prints in a topic
    that Staging does not (and the reverse), which no other check reported.

    Production's whole topic is aligned word by word, in order, against
    Staging's. A run the other side prints elsewhere in the topic only moved or
    wrapped, and is not missing; a run an earlier issue already names is not
    reported twice. What is left was on the page and nobody said so."""
    a, b = _topic_stream(exp_topic), _topic_stream(act_topic)
    if not a and not b:
        return []
    ta, tb = [t for t, _ in a], [t for t, _ in b]
    covered: set[str] = set()
    for d in chapter.differences:
        if d.get("section") != topic:
            continue
        said = " ".join([d.get("summary", "")] + list(d.get("gone") or []) + list(d.get("extra") or []))
        for key in ("exp", "act"):
            el = d.get(key)
            if el is not None and d.get("type") in ("missing", "added", "table-row-missing", "table-row-added",
                                                    "table-as-text", "table-cell"):
                said += " " + (el.text or "")
        covered |= set(_TOKEN_RE.findall(_normalise(said)))
    # A heading or line at a topic's first or last line lands in the next topic
    # on one side ("Japan RoHS" set as a page-top heading): the neighbouring
    # topics' text counts as printed, for this check only.
    order = list(dict.fromkeys(e.section for e in chapter.exp_elements + chapter.act_elements))
    at = order.index(topic) if topic in order else -1
    near = {order[k] for k in (at - 1, at + 1) if 0 <= k < len(order)} if at >= 0 else set()
    near_a = [t for t, _ in _topic_stream([e for e in chapter.exp_elements if e.section in near])]
    near_b = [t for t, _ in _topic_stream([e for e in chapter.act_elements if e.section in near])]
    stream_a = f" {' '.join(ta)} " + f" {' '.join(near_a)} "
    stream_b = f" {' '.join(tb)} " + f" {' '.join(near_b)} "
    # The same words with their spacing taken out: "third-party" split at a
    # line end, "10-AACCA6005C" wrapped after its hyphen - the same word.
    solid_a, solid_b = "".join(ta), "".join(tb)
    count_a, count_b = Counter(ta), Counter(tb)

    # Last check before calling words absent: what the other document's own
    # pages for this chapter PRINT, read straight off the page. A chapter
    # boundary a heading apart ("Japan RoHS" filed under the next chapter on one
    # side) or a line the element collection skipped is on the page all the same.
    def page_stream(doc: fitz.Document, pages: list[int]) -> str:
        wanted = sorted({p + d for p in pages for d in (-1, 0, 1) if 0 <= p + d < doc.page_count})
        return " " + " ".join(
            " ".join(_TOKEN_RE.findall(_normalise(doc[p].get_text("text")))) for p in wanted
        ) + " "

    printed_cache: dict[str, str] = {}

    def printed_on_pages(run: list[str], kind: str) -> bool:
        if kind not in printed_cache:
            printed_cache[kind] = (page_stream(actual, chapter.act_pages or []) if kind == "missing"
                                   else page_stream(expected, chapter.exp_pages or []))
        return f" {' '.join(run)} " in printed_cache[kind]

    out: list[dict] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for run, start, other, solid, counts, source, doc, kind in (
            (ta[i1:i2], i1, stream_b, solid_b, count_b, a, expected, "missing"),
            (tb[j1:j2], j1, stream_a, solid_a, count_a, b, actual, "added"),
        ):
            if len(run) < _COVERAGE_MIN_TOKENS or _CJK_RE.search("".join(run)):
                continue
            if f" {' '.join(run)} " in other or set(run) <= covered:
                continue  # moved or wrapped, or already reported
            if "".join(run) in solid:
                continue  # the same characters, only split or joined differently
            if all(counts[t] >= n for t, n in Counter(run).items()):
                continue  # every word is printed in the topic - set in another order or block
            if printed_on_pages(run, kind):
                continue  # printed on the other document's pages for this chapter
            el = source[start][1]
            printed = _printed_form(" ".join(run), [el.text or ""])
            shown = printed if len(printed) <= 150 else printed[:149] + "…"
            place = Element(kind=KIND_TEXT, text=shown, key=_normalise(printed),
                            boxes=_phrase_boxes(doc, el, printed), section=topic)
            covered |= set(run)
            out.append({
                "type": kind, "kind": KIND_TEXT, "section": topic,
                "summary": (f"Text missing in Staging — “{shown}” is printed in Production and not in Staging."
                            if kind == "missing" else
                            f"Text extra in Staging — “{shown}” is printed in Staging and not in Production."),
                "detail": "",
                "exp": place if kind == "missing" else None,
                "act": place if kind == "added" else None,
            })
    return out


def _span_lines(doc: fitz.Document, pages: list[int], span: tuple, body: float,
                titles: set[str], skip: list[tuple] | None = None) -> list[dict]:
    """Every printed line of a chapter's pages inside its span - figure
    labels, table cells, callouts, fragments, all of it; a scanned page's
    lines come from OCR (see `_page_lines`)."""
    start, end = span
    out: list[dict] = []
    for p in sorted(set(pages)):
        y0 = start[1] if start and p == start[0] else 0.0
        y1 = end[1] - 0.5 if end and p == end[0] else 1e9
        if y1 <= y0:
            continue
        out.extend(_page_lines(doc, p, y0, y1, body, titles))

    def skipped(ln: dict) -> bool:  # table of contents / short Q&A: left out on purpose
        at = (ln["page"], ln["bbox"][1])
        return any(s_start <= at and (s_end is None or at < s_end) for s_start, s_end, *_ in (skip or []))
    return [ln for ln in out if not skipped(ln)]


_HEADER_REPEAT_TOKENS = 8     # a table header line is at most this many words
_HEADER_REPEAT_REACH = 20.0   # pt below a table's top edge its header line sits
_HEADER_REPEAT_BAND = 0.12    # share of the page's height from its top where a repeated header prints


_ENDS_PAGE_RE = re.compile(r"\bpage\s*$", re.IGNORECASE)
_STARTS_NUMBER_RE = re.compile(r"^\s*(\d{1,4})\b")


def _rejoin_page_refs(lines: list[dict]) -> list[dict]:
    """A page reference wrapped over two lines ("(See page" / "25 )") put back
    on one, so the reference is recognised as one - in reading order the two
    can sit apart, with a diagram callout ("11") printed between them."""
    out = [dict(ln) for ln in lines]
    for i, ln in enumerate(out):
        if not _ENDS_PAGE_RE.search(ln["text"]):
            continue
        x0, _, _, y1 = ln["bbox"]
        height = max(1.0, ln["bbox"][3] - ln["bbox"][1])
        # Only the very next line of the same column continues it.
        below = [n for n in out[i + 1:i + 12] if n["page"] == ln["page"] and abs(n["bbox"][0] - x0) <= 30
                 and -0.5 * height <= n["bbox"][1] - y1 <= 1.2 * height]
        if below:
            nxt = min(below, key=lambda n: n["bbox"][1])
            m = _STARTS_NUMBER_RE.match(nxt["text"])
            if m:
                ln["text"] = f"{ln['text']} {m.group(1)}"
                nxt["text"] = nxt["text"][m.end():]
    return out


def _join_audit_lines(lines: list[dict]) -> str:
    """The lines as one text, a word split by a hyphen at a line end rejoined
    only with the line that continues it - the next line of the same column.
    A table label wrapped as "User / HDR-" with the next column's "Accesses"
    beside it read as "HDRAccesses"."""
    parts: list[str] = []
    for ln, nxt in zip(lines, lines[1:] + [None]):
        parts.append(ln["text"])
        if nxt is None:
            break
        x0, y0, x1, y1 = ln["bbox"]
        continues = (nxt["page"] == ln["page"] and nxt["bbox"][1] >= y1 - 0.5 * (y1 - y0)
                     and nxt["bbox"][0] < x1 and nxt["bbox"][2] > x0)
        parts.append("\n" if continues else " | ")
    return "".join(parts)


_ENDS_HYPHEN_RE = re.compile(r"\w[-\u00ad]\s*$")
_STARTS_WORD_RE = re.compile(r"^\s*([^\W\d_]+)")


def _rejoin_hyphens(lines: list[dict]) -> list[str]:
    """A word broken at a line end inside a table cell ("Receiv-" / "er.")
    put beside the line that continues it - the next line of its own column,
    which in reading order can sit several cells later. The two halves are
    only brought together, not joined: the count check already cancels a pair
    of adjacent words the other side prints as one ("receiv" + "er" against
    "receiver"), and that keeps a real hyphen ("USB-" / "C") split as the
    other side prints it too."""
    out = [dict(ln) for ln in lines]
    for i, ln in enumerate(out):
        if not _ENDS_HYPHEN_RE.search(ln["text"]):
            continue
        x0, _, x1, y1 = ln["bbox"]
        height = max(1.0, ln["bbox"][3] - ln["bbox"][1])
        below = [n for n in out[i + 1:i + 12]
                 if n["page"] == ln["page"] and abs(n["bbox"][0] - x0) <= 20
                 and -0.5 * height <= n["bbox"][1] - y1 <= 1.6 * height]
        if not below:
            continue
        nxt = min(below, key=lambda n: n["bbox"][1])
        m = _STARTS_WORD_RE.match(nxt["text"])
        if m:
            ln["text"] = f"{ln['text'].rstrip()[:-1].rstrip()} {m.group(1)}"
            nxt["text"] = nxt["text"][m.end():]
    return out


def _audit_tokens(text: str) -> list[str]:
    # A cross-reference's page ("(See page 25)") and Staging's link naming the
    # section in its place ("(“Microphone and remote control LED indicator”)")
    # are the same reference, not words either side lost or gained.
    text = _XREF_RE.sub(" ", _PAGE_REF_RE.sub(" ", text or ""))
    return [t for t in _TOKEN_RE.findall(_normalise(text)) if len(t) > 1 or t.isdigit()]


_CALLOUT_NUMBER_RE = re.compile(r"^\s*\(?\d{1,2}\)?[.:]?\s*$")


# A label drawn ON a diagram, alongside the bare callout numbers: a dimension
# ("3 cm"), a state ("ON"), an axis name. Short, because a caption or a line of
# body text that happens to overlap the artwork is neither.
_ARTWORK_LABEL_WORDS = 3
_ARTWORK_LABEL_CHARS = 20
# An "artwork box" that covers most of the page is a scanned page, not a
# diagram: every line on it sits inside it, and dropping them all would leave
# the page's whole text uncounted on that side alone.
_ARTWORK_LABEL_MAX_PAGE_SHARE = 0.6


def _is_artwork_callout(doc: "fitz.Document | None", line: dict) -> bool:
    """Text printed ON a diagram - the "4" pointing at the TOUCH SCREEN port,
    the "3 cm" beside a clearance arrow - rather than a word of the topic.
    Production draws its diagrams as vectors, so this text is text and
    countable; Staging prints the same diagram as one flat image, where it is
    pixels too small for OCR to settle. Counting it reported every callout
    number - and every dimension label - missing from Staging.

    A bare number counts when it sits about where the artwork is; anything
    else must be a SHORT label drawn wholly inside the artwork, so a caption
    under the figure or a sentence beside it is never swallowed."""
    text = (line.get("text") or "").strip()
    if doc is None or not text:
        return False
    callout = bool(_CALLOUT_NUMBER_RE.match(text))
    label = len(text) <= _ARTWORK_LABEL_CHARS and len(text.split()) <= _ARTWORK_LABEL_WORDS
    if not (callout or label):
        return False
    if _inside_table(doc, line["page"], line["bbox"]):
        return False  # a table's own cell ("50" in a screen-size column), not a callout
    x0, y0, x1, y1 = line["bbox"]
    try:
        boxes = _page_artwork_boxes(doc, line["page"])
    except Exception:
        return False
    if callout:
        centre = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        return any((r + (-8, -8, 8, 8)).contains(centre) for r in boxes)
    if line.get("ocr"):
        return False  # read off a scanned page: the whole page is "artwork"
    try:
        page_area = doc[line["page"]].rect.get_area() or 1.0
    except Exception:
        return False
    return any(r.contains(fitz.Rect(x0, y0, x1, y1))
               and r.get_area() <= _ARTWORK_LABEL_MAX_PAGE_SHARE * page_area
               for r in boxes)


_BLOCK_GRAM = 10      # words in the window matched against the other side
_BLOCK_MIN_WORDS = 25
_BLOCK_SURPLUS = 0.5   # this share of a run's words must be surplus on this side
_BLOCK_SAID = 0.7      # this share already named by another finding: not reported again  # a run this long is a block of content, not a stock phrase


def _topic_tokens(lines: list[dict], at) -> dict[str, list[tuple[str, dict]]]:
    """`{topic: [(word, the line it was printed on), ...]}`, in reading order."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for ln in lines:
        topic = at(ln["page"], ln["bbox"][1])
        if not topic or _BARE_CALLOUT_RE.match(ln["text"]):
            continue
        for token in _audit_tokens(ln["text"]):
            out.setdefault(topic, []).append((token, ln))
    return out


def _extra_runs(mine: list[str], theirs: list[str]) -> list[tuple[int, int]]:
    """Spans of `mine` this side prints MORE often than the other does.

    Whole blocks, not loose words: a run of `_BLOCK_GRAM` words is "extra"
    only where this side prints that exact run more times than the other side
    does, so a procedure printed twice here and once there marks its second
    copy and a phrase both sides use once marks nothing. Wrapping cannot
    matter - the words are counted in reading order, with line breaks gone."""
    grams = lambda seq: [tuple(seq[i:i + _BLOCK_GRAM]) for i in range(len(seq) - _BLOCK_GRAM + 1)]
    budget = Counter(grams(theirs))
    marked: set[int] = set()
    for i, gram in enumerate(grams(mine)):
        if budget[gram] > 0:
            budget[gram] -= 1  # the other side's copy of this run - not extra
        else:
            marked.update(range(i, i + _BLOCK_GRAM))
    runs: list[tuple[int, int]] = []
    for i in sorted(marked):
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return [(a, b) for a, b in runs if b - a + 1 >= _BLOCK_MIN_WORDS]


def _pictured_words(figs: list[Element] | None, doc: "fitz.Document | None") -> dict[str, Counter]:
    """`{topic: words read by OCR off that topic's pictures}` - a word the
    other side prints only inside its artwork (a figure label baked into
    Staging's image, set as text in Production) is printed there to read, and
    can settle a shortfall, never raise one."""
    out: dict[str, Counter] = {}
    if doc is None:
        return out
    from pdfval import ocr
    for f in figs or []:
        if not f.boxes or not f.section:
            continue
        page_index, bbox = f.boxes[0]
        try:
            read = ocr.words_in(_page_words(doc).ocr_words(page_index), bbox, margin=4)
        except Exception:
            continue
        out.setdefault(f.section, Counter()).update(_audit_tokens(" ".join(read)))
    return out


def _named_words(diff: dict) -> tuple[list[str], list[str]]:
    """`(what this finding says Production prints, what it says Staging does)` -
    the words themselves, not the sentence built around them."""
    prod, stage = [], []
    for op in diff.get("word_diff") or []:
        if op.get("type") == "del":
            prod += _audit_tokens(op.get("text", ""))
        elif op.get("type") == "ins":
            stage += _audit_tokens(op.get("text", ""))
    prod += [t for x in (diff.get("gone") or []) for t in _audit_tokens(str(x))]
    stage += [t for x in (diff.get("extra") or []) for t in _audit_tokens(str(x))]
    if diff.get("type") == "missing" and diff.get("exp") is not None:
        prod += _audit_tokens(getattr(diff["exp"], "text", "") or "")
    if diff.get("type") == "added" and diff.get("act") is not None:
        stage += _audit_tokens(getattr(diff["act"], "text", "") or "")
    return prod, stage


def _hyphen_piece(token: str, others: set[str]) -> bool:
    """Is this not a word at all, but a piece of one the other side prints?

    A narrow table column hyphenates: Production's "Receiv-/er", "sup-/ply",
    "manage-/ment" read out as fragments, and cells read in another order glue
    two of them together ("sup" + "make" -> "supmake"). Every such piece
    answers to a word the other side prints whole."""
    def piece(t: str) -> bool:
        return len(t) >= 2 and any(len(o) > len(t) and (o.startswith(t) or o.endswith(t)) for o in others)
    if token in others or piece(token):
        return True
    # Two pieces run together where the other side reads them apart.
    return any((a in others or piece(a)) and (b in others or piece(b))
               for a, b in ((token[:i], token[i:]) for i in range(2, len(token) - 1)))


def _accounted_for(named: list[str], other: Counter, joined: Counter,
                   others: set[str] | None = None) -> bool:
    """Is every word this finding names printed on the other side anyway?

    Counted, not merely looked up, so a block printed twice here and once
    there is still short. A word split over a line end on one side only
    ("Receiv-" / "er" in Production's narrow table column, "Receiver" in
    Staging's wide one) is the same word differently broken: the two halves
    answer to the whole, and the whole to the two halves."""
    left, pairs = +other, +joined
    i = 0
    while i < len(named):
        t = named[i]
        if left[t] > 0:
            left[t] -= 1
        elif pairs[t] > 0:  # the other side breaks this word over a line end
            pairs[t] -= 1
        elif i + 1 < len(named) and left[t + named[i + 1]] > 0:
            left[t + named[i + 1]] -= 1  # THIS side breaks it; the other prints it whole
            i += 1
        elif others is not None and _hyphen_piece(t, others):
            pass  # a hyphenated piece of a word the other side prints whole
        else:
            return False
        i += 1
    return bool(named)


def drop_reflow_noise(chapter: "Chapter", exp_lines: list[dict], act_lines: list[dict],
                      exp_at, act_at) -> None:
    """Drop the text findings that are the same content, set differently.

    Two documents set the same table in different column widths: Production
    hyphenates "connect-/ing", "Receiv-/er", "sup-/ply" and reads its cells in
    another order, Staging does neither. Compared as a stream of words that
    reads as dozens of missing and extra fragments, and a reader cannot find
    the few real differences among them. A finding survives only where a word
    it names is genuinely not printed on the other side under the same
    heading - order, wrapping and hyphenation aside."""
    def tokens(lines, at) -> tuple[dict[str, Counter], dict[str, Counter]]:
        seq: dict[str, list[str]] = {}
        for ln in lines:
            topic = at(ln["page"], ln["bbox"][1])
            if topic:
                seq.setdefault(topic, []).extend(_audit_tokens(ln["text"]))
        whole = {k: Counter(v) for k, v in seq.items()}
        joined = {k: Counter(x + y for x, y in zip(v, v[1:])) for k, v in seq.items()}
        return whole, joined

    exp_w, exp_j = tokens(exp_lines, exp_at)
    act_w, act_j = tokens(act_lines, act_at)
    empty = Counter()
    # A finding whose heading the other side has no copy of is still checked,
    # against that side's whole chapter: hyphenated pieces are the typesetting
    # of the page they are printed on, wherever the two split their headings.
    exp_all = sum(exp_w.values(), Counter())
    act_all = sum(act_w.values(), Counter())
    exp_all_j = sum(exp_j.values(), Counter())
    act_all_j = sum(act_j.values(), Counter())
    kept = []
    for diff in chapter.differences:
        if diff.get("type") not in _REFLOW_CHECKED or diff.get("block"):
            kept.append(diff)
            continue
        topic = diff.get("section")
        prod, stage = _named_words(diff)
        # Each side's words against the OTHER document's copy of this topic.
        here_act, here_exp = act_w.get(topic), exp_w.get(topic)
        said_prod = not prod or _accounted_for(
            prod, here_act if here_act is not None else act_all,
            act_j.get(topic, act_all_j) if here_act is not None else act_all_j, set(act_all))
        said_stage = not stage or _accounted_for(
            stage, here_exp if here_exp is not None else exp_all,
            exp_j.get(topic, exp_all_j) if here_exp is not None else exp_all_j, set(exp_all))
        if (prod or stage) and said_prod and said_stage:
            continue  # the same content on both sides, only set differently
        kept.append(diff)
    chapter.differences = kept


_REFLOW_CHECKED = {"text", "missing", "added", "table-cell"}


def _already_said(chapter: "Chapter", topic: str) -> set[str]:
    """Every word the findings already made about this topic name - a block
    another check has reported is not reported again as a block."""
    said: list[str] = []
    for d in chapter.differences:
        if d.get("section") not in (topic, None, ""):
            continue
        said += [d.get("summary", ""), str(d.get("detail") or "")]
        for key in ("exp", "act"):
            el = d.get(key)
            if el is not None:
                said.append(getattr(el, "text", "") or "")
    return set(_TOKEN_RE.findall(_normalise(" ".join(said))))


def repeated_block_changes(chapter: "Chapter", exp_lines: list[dict], act_lines: list[dict],
                           exp_at, act_at, exp_figs: list[Element] | None = None,
                           act_figs: list[Element] | None = None,
                           expected: "fitz.Document | None" = None,
                           actual: "fitz.Document | None" = None) -> list[dict]:
    """A whole block of content one side prints and the other does not - most
    often a procedure Staging repeats twice under one heading where Production
    prints it once.

    `word_audit_changes` cannot see this: it compares how often each WORD is
    printed, and a duplicated block is the same words again, so its counts
    very nearly cancel and what survives is a handful of loose words pointing
    at scattered lines instead of the repeated block itself."""
    out: list[dict] = []
    exp_t, act_t = _topic_tokens(exp_lines, exp_at), _topic_tokens(act_lines, act_at)
    # What each side prints only inside its pictures counts as printed there.
    drawn = {"missing": _pictured_words(act_figs, actual),
             "added": _pictured_words(exp_figs, expected)}
    # The same words counted over the WHOLE chapter: content the other side
    # prints under a NEIGHBOURING heading is there to read, wherever the two
    # documents happen to split their headings, and reporting it "missing"
    # was wrong. Only what no topic of the other side prints is reported.
    whole = {"missing": Counter(t for ts in exp_t.values() for t, _ in ts),
             "added": Counter(t for ts in act_t.values() for t, _ in ts)}
    whole["missing"].subtract(Counter(t for ts in act_t.values() for t, _ in ts))
    whole["added"].subtract(Counter(t for ts in exp_t.values() for t, _ in ts))
    for topic in dict.fromkeys(list(exp_t) + list(act_t)):
        a, b = exp_t.get(topic, []), act_t.get(topic, [])
        covered = _already_said(chapter, topic)
        for mine, theirs, kind in ((a, b, "missing"), (b, a, "added")):
            # Words this side genuinely prints more often, counted over the
            # whole topic. A run only reads as extra in the order it happens
            # to be printed in - a table read column-first on one side, a word
            # hyphenated over a line break on the other - is the same content,
            # and every one of its words is already there on the other side.
            surplus = Counter(t for t, _ in mine)
            surplus.subtract(Counter(t for t, _ in theirs))
            for start, end in _extra_runs([t for t, _ in mine], [t for t, _ in theirs]):
                span = mine[start:end + 1]
                enough = True
                for counts in (surplus - drawn[kind].get(topic, Counter()),
                               whole[kind] - sum(drawn[kind].values(), Counter())):
                    left = +counts
                    if sum(1 for t, _ in span
                           if left[t] > 0 and not left.subtract({t: 1})) < len(span) * _BLOCK_SURPLUS:
                        enough = False  # printed on the other side too - reordered, or under another heading
                        break
                if not enough:
                    continue
                # Trimmed to the words this side really prints more often:
                # a run can open or close on ordinary words both sides print
                # (the sentence around Production's picture captions), and
                # quoting those named the wrong text and boxed the wrong lines.
                core = [k for k, (t, _) in enumerate(span) if surplus[t] > 0]
                if core:
                    span = span[core[0]:core[-1] + 1]
                if sum(1 for t, _ in span if t in covered) >= len(span) * _BLOCK_SAID:
                    continue  # another finding already reports this content
                words = " ".join(t for t, _ in span)
                boxes, seen = [], set()
                for _, ln in span:
                    at = (ln["page"], tuple(ln["bbox"]))
                    if at not in seen:
                        seen.add(at)
                        boxes.append(at)
                # Printed twice on this side, once on the other: a repeat, not
                # content the other side is missing outright.
                head = tuple(t for t, _ in span[:_BLOCK_GRAM])
                all_mine = [t for t, _ in mine]
                repeat = sum(1 for i in range(len(all_mine) - _BLOCK_GRAM + 1)
                             if tuple(all_mine[i:i + _BLOCK_GRAM]) == head) > 1
                side = "Staging" if kind == "added" else "Production"
                shown = words[:120] + ("..." if len(words) > 120 else "")
                out.append({
                    "type": kind, "kind": KIND_TEXT, "section": topic, "block": True,
                    "summary": (
                        f"Block repeated in {side} — {side} prints this block twice under this heading, "
                        f"the other document prints it once: “{shown}”."
                        if repeat else
                        (f"Block of text missing in Staging — printed in Production, not in Staging: “{shown}”."
                         if kind == "missing" else
                         f"Block of text extra in Staging — printed in Staging, not in Production: “{shown}”.")
                    ),
                    "detail": words[:400],
                    "exp": (Element(kind=KIND_TEXT, text=words, key=words, section=topic,
                                    boxes=boxes[:12]) if kind == "missing" else None),
                    "act": (Element(kind=KIND_TEXT, text=words, key=words, section=topic,
                                    boxes=boxes[:12]) if kind == "added" else None),
                })
    return out


def word_audit_changes(chapter: "Chapter", exp_lines: list[dict], act_lines: list[dict],
                       exp_at, act_at, exp_figs: list[Element] | None = None,
                       act_figs: list[Element] | None = None, expected: fitz.Document | None = None,
                       actual: fitz.Document | None = None) -> list[dict]:
    """The final safety net: every word printed in each topic, counted on both
    sides straight off the page (OCR for a scanned page), with no element,
    pairing or noise filter in between. A word Production prints more often
    than Staging in the same topic - or the reverse - that no other finding
    already names is reported. Order and wrapping cannot matter: only counts
    are compared.

    Words the other side prints only as part of a picture (Staging's
    dimension drawing with "241.3" in its artwork, Production's as text) are
    read off that side's figures by OCR and count as printed there - they
    can settle a shortfall, never raise one."""
    exp_pictured = _pictured_words(exp_figs, expected)
    act_pictured = _pictured_words(act_figs, actual)

    def table_tops(elements) -> dict[int, list[tuple]]:
        tops: dict[int, list[tuple]] = {}
        for e in elements or []:
            if e.kind == KIND_TABLE:
                for page_index, bbox in e.boxes:
                    tops.setdefault(page_index, []).append(tuple(bbox))
        return tops

    header_words: dict[str, set] = {}

    def by_topic(lines, at, tops, doc=None):
        out: dict[str, list] = {}
        first_seen: dict[tuple, int] = {}
        for ln in lines:
            topic = at(ln["page"], ln["bbox"][1])
            if not topic:
                continue
            # A callout's label ("Note", "TIP:") is its styling, never content.
            if _BARE_CALLOUT_RE.match(ln["text"]):
                continue
            if _is_artwork_callout(doc, ln):
                continue  # a number printed on a diagram, not a word of the topic
            tokens = _audit_tokens(ln["text"])
            # A table's header reprinted at the top of its continuation page
            # ("Item | Descriptions" again on p.35) is the page break's, not
            # the text's: the other side may not break the table there.
            # Its words are left out of the count on BOTH sides - skipping the
            # line on one side only unbalanced the counts.
            seen = first_seen.setdefault((topic, tuple(tokens)), ln["page"])
            x0, y0, x1, y1 = ln["bbox"]
            near_top = False
            if doc is not None:
                try:
                    near_top = y0 <= doc[ln["page"]].rect.height * _HEADER_REPEAT_BAND
                except Exception:
                    near_top = False
            if seen < ln["page"] and 0 < len(tokens) <= _HEADER_REPEAT_TOKENS and (near_top or any(
                    b[0] - 2 <= x0 and x1 <= b[2] + 2 and b[1] - 3 <= y0 <= b[1] + _HEADER_REPEAT_REACH
                    for b in tops.get(ln["page"], ()))):
                header_words.setdefault(topic, set()).update(tokens)
            out.setdefault(topic, []).append((ln, tokens))
        return out

    exp_t = by_topic(_rejoin_hyphens(_rejoin_page_refs(exp_lines)), exp_at, table_tops(getattr(chapter, "exp_elements", None)), expected)
    act_t = by_topic(_rejoin_hyphens(_rejoin_page_refs(act_lines)), act_at, table_tops(getattr(chapter, "act_elements", None)), actual)
    # The same count over the WHOLE chapter, not just the topic. The topic
    # boundaries the two sides fall into are not identical - a heading one
    # side bookmarks and the other does not, a step whose boilerplate
    # ("...to go to a sub menu, and then use...", printed seven times in both
    # documents) lands on the far side of a boundary - and a word counted in
    # topic A here and topic B there then shows as missing from A and extra in
    # B, though the chapter prints it exactly as often on both sides. The
    # question this check exists to answer is whether the words are printed at
    # all; the chapter-wide count is what answers it.
    chapter_exp = Counter(_audit_tokens(_join_audit_lines([ln for lines in exp_t.values() for ln, _ in lines])))
    chapter_act = Counter(_audit_tokens(_join_audit_lines([ln for lines in act_t.values() for ln, _ in lines])))
    out: list[dict] = []
    for topic in dict.fromkeys(list(exp_t) + list(act_t)):
        a, b = exp_t.get(topic, []), act_t.get(topic, [])
        # Counted over the topic's whole text, so a page reference wrapped
        # over two lines ("on" / "page 26") is still recognised and dropped.
        ta = _audit_tokens(_join_audit_lines([ln for ln, _ in a]))
        tb = _audit_tokens(_join_audit_lines([ln for ln, _ in b]))
        ca, cb = Counter(ta), Counter(tb)
        covered: set[str] = set()
        for d in chapter.differences:
            if d.get("section") not in (topic, None, ""):
                continue
            said = " ".join([d.get("summary", ""), str(d.get("detail") or "")]
                            + [str(x) for x in (d.get("gone") or [])] + [str(x) for x in (d.get("extra") or [])])
            for key in ("exp", "act"):
                el = d.get(key)
                if el is not None:
                    said += " " + (getattr(el, "text", "") or "")
            covered |= set(_TOKEN_RE.findall(_normalise(said)))
        for mine, theirs, own, other, lines, kind, drawn, mine_all, theirs_all in (
            (ca, cb, ta, tb, a, "missing", act_pictured.get(topic, Counter()), chapter_exp, chapter_act),
            (cb, ca, tb, ta, b, "added", exp_pictured.get(topic, Counter()), chapter_act, chapter_exp),
        ):
            short = {t: min(n - theirs[t] - drawn[t], mine_all[t] - theirs_all[t]) for t, n in mine.items()
                     if n > theirs[t] + drawn[t] and mine_all[t] > theirs_all[t] and t not in covered
                     and t not in header_words.get(topic, ())}
            short = {t: n for t, n in short.items() if n > 0}
            if not short:
                continue
            # A word hyphenated or split at a line end on one side only
            # ("con-" / "tents"): the same letters, differently broken.
            other_joined = {x + y for x, y in zip(other, other[1:])}
            other_set = set(other)
            for x, y in zip(own, own[1:]):
                if x + y in other_set:
                    short.pop(x, None); short.pop(y, None)
            short = {t: n for t, n in short.items() if t not in other_joined}
            if not short:
                continue
            words = [t for t in dict.fromkeys(own) if t in short]
            boxes, seen = [], set()
            # Only the lines the other side does not print: "Some of the
            # accessories may vary by region." holds "region" too, but Staging
            # prints that sentence - the caption "(Varies by region)" is the
            # missing one.
            other_text = " " + " ".join(other) + " "
            other_doc = actual if kind == "missing" else expected
            for ln, ts in lines:
                if ts and f" {' '.join(ts)} " in other_text:
                    continue
                # The line IS printed in the other document, just not under
                # the topic this side files it under - the two outlines
                # disagree about where a boilerplate step belongs, or a
                # heading one side bookmarks splits a topic the other keeps
                # whole. The words are there; only the filing differs, and
                # this check is about whether the content is printed at all.
                if _printed_anywhere_in(other_doc, ln["text"]):
                    continue
                if any(t in short for t in ts) and (ln["page"], ln["bbox"]) not in seen:
                    seen.add((ln["page"], ln["bbox"]))
                    boxes.append((ln["page"], tuple(ln["bbox"])))
            # Every line that would have been boxed turned out to be printed in
            # the other document: nothing of this topic's content is missing.
            if not boxes:
                continue
            shown = ", ".join(f"“{w}”" + (f" ×{short[w]}" if short[w] > 1 else "") for w in words[:40])
            if len(words) > 40:
                shown += f" and {len(words) - 40} more"
            ocr_page = any(ln.get("ocr") for ln, ts in lines if any(t in short for t in ts))
            place = Element(kind=KIND_TEXT, text=" ".join(words), key=" ".join(words),
                            boxes=boxes[:12], section=topic, ocr=ocr_page)
            out.append({
                "type": kind, "kind": KIND_TEXT, "section": topic, "audit": True,
                "summary": (f"Words missing in Staging — printed in Production in this topic, not in Staging: {shown}."
                            if kind == "missing" else
                            f"Words extra in Staging — printed in Staging in this topic, not in Production: {shown}."),
                "detail": " ".join(ln["text"] for ln, ts in lines if any(t in short for t in ts)
                                   and not (ts and f" {' '.join(ts)} " in other_text)
                                   and not _printed_anywhere_in(other_doc, ln["text"]))[:400],
                "exp": place if kind == "missing" else None,
                "act": place if kind == "added" else None,
                **({"ocr_sides": ["Production" if kind == "missing" else "Staging"]} if ocr_page else {}),
            })
    return out


_ROW_NUMBER_RE = re.compile(r"^\d{1,3}\.$")  # "8." - a bare "8" is a diagram callout


def _numbered_cell_labels(lines: list[dict]) -> dict[tuple, dict]:
    """`{(row number, label): line}` - the first text of each numbered table
    row ("8." | "YouTube"), with `drop`: how far below the row number's own
    line it starts. Level with the number when an icon sits beside the label;
    a line or more down when the icon is stacked above it."""
    out: dict[tuple, dict] = {}
    nums = [ln for ln in lines if _ROW_NUMBER_RE.match(ln["text"].strip())]
    for num in nums:
        page, (nx0, ny0, nx1, ny1) = num["page"], num["bbox"]
        h = max(1.0, ny1 - ny0)
        # Up to the next row number in the same column, the cell to the right.
        below = [m["bbox"][1] for m in nums if m["page"] == page and abs(m["bbox"][0] - nx0) < 4
                 and m["bbox"][1] > ny0 + 1]
        limit = min(below, default=ny0 + 6 * h)
        cell = sorted((ln for ln in lines if ln["page"] == page and ln is not num
                       and ln["bbox"][0] >= nx1 - 1 and ln["bbox"][0] < nx1 + 160
                       and ny0 - 2 <= ln["bbox"][1] < limit - 1
                       and _audit_tokens(ln["text"])),
                      key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
        if not cell:
            continue
        first = cell[0]
        label = " ".join(_audit_tokens(first["text"]))
        key = (num["text"].strip().rstrip("."), label)
        out.setdefault(key, {**first, "drop": (first["bbox"][1] - ny0) / h})
    return out


def table_icon_layout_changes(exp_lines: list[dict], act_lines: list[dict],
                              exp_at, act_at) -> list[dict]:
    """A numbered table row's icon printed beside its label on one side
    ("[icon] YouTube" on the row's first line) and stacked above it on the
    other (the icon on one line, "YouTube" on the next) - same words, but the
    icon and its label no longer line up. One finding per topic, every such
    row boxed on both sides."""
    def by_topic(lines, at):
        out: dict[str, list] = {}
        for ln in lines:
            topic = at(ln["page"], ln["bbox"][1])
            if topic:
                out.setdefault(topic, []).append(ln)
        return out

    exp_t, act_t = by_topic(exp_lines, exp_at), by_topic(act_lines, act_at)
    out: list[dict] = []
    for topic in exp_t:
        if topic not in act_t:
            continue
        a, b = _numbered_cell_labels(exp_t[topic]), _numbered_cell_labels(act_t[topic])
        stacked, beside = [], []
        for key in a.keys() & b.keys():
            la, lb = a[key], b[key]
            if la["drop"] < 0.5 and lb["drop"] >= 1.2:
                stacked.append((key, la, lb))
            elif lb["drop"] < 0.5 and la["drop"] >= 1.2:
                beside.append((key, la, lb))
        for rows, in_staging in ((stacked, "stacked above its label"), (beside, "beside its label")):
            if not rows:
                continue
            rows.sort(key=lambda r: int(r[0][0]))
            names = ", ".join(f"{n}. {la['text'].strip()}" for (n, _), la, _ in rows[:12])
            other = "beside it on the same line" if in_staging.startswith("stacked") else "stacked above it"
            mk = lambda side: Element(kind=KIND_TABLE, text=names, key=names, section=topic,
                                      boxes=[(ln["page"], tuple(ln["bbox"])) for ln in side][:12])
            out.append({
                "type": "table-cell-layout", "kind": KIND_TABLE, "section": topic,
                "summary": (f"Icon and label misaligned in table — in Staging the icon is {in_staging}, "
                            f"Production prints it {other}: {names}."),
                "detail": "",
                "exp": mk([la for _, la, _ in rows]), "act": mk([lb for _, _, lb in rows]),
            })
    return out


_PLAIN_TEXT_KINDS_EXCLUDED = {KIND_FIGURE, KIND_TABLE}


def _book_stream(chapter_elements: list[list[Element]]) -> list[tuple[str, Element]]:
    """Like `_topic_stream`, but reading order across several chapters at
    once: `Element.order` is only unique WITHIN the chapter that collected it
    (`collect_elements` restarts it at 0 for every chapter), so sorting a
    flattened list of several chapters' elements by `.order` interleaves
    chapter 2's early elements ahead of chapter 1's later ones and scrambles
    the book. Each chapter is sorted by its own `.order` first, and chapters
    are kept in the order the caller already put them in (reading order)."""
    out: list[tuple[str, Element]] = []
    for elements in chapter_elements:
        for el in sorted(elements, key=lambda e: e.order):
            if el.kind == KIND_FIGURE:
                continue
            for token in _TOKEN_RE.findall(_normalise(_xref_free(el.text))):
                if _BARE_CALLOUT_RE.match(token):
                    continue
                out.append((token, el))
    return out


def plain_text_scan(chapters: list["Chapter"], expected: fitz.Document,
                    actual: fitz.Document) -> list[tuple["Chapter", dict]]:
    """A last, whole-book pass over plain text alone (paragraphs, headings,
    callouts, list items - never table cells or figures, which are checked on
    their own terms elsewhere).

    Every chapter above already compares its own topics word for word, but
    only inside that topic: content that landed under the wrong heading
    because the two tables of contents did not line up there, or a whole L1
    heading only one side has, is invisible to a check that never looks
    outside its own topic. Here every chapter's plain text is joined into one
    continuous book, on each side, and matched once more straight across -
    catching anything genuinely absent from one side that survived every
    topic-scoped check.

    Nothing already reported by any chapter is repeated: every word any
    existing difference already names is excluded up front, so this only
    surfaces content nobody has flagged yet.
    """
    owner: dict[int, "Chapter"] = {}
    exp_by_chapter: list[list[Element]] = []
    act_by_chapter: list[list[Element]] = []
    for ch in chapters:
        exp_kept = [e for e in ch.exp_elements if e.kind not in _PLAIN_TEXT_KINDS_EXCLUDED]
        act_kept = [e for e in ch.act_elements if e.kind not in _PLAIN_TEXT_KINDS_EXCLUDED]
        for e in exp_kept:
            owner[id(e)] = ch
        for e in act_kept:
            owner[id(e)] = ch
        exp_by_chapter.append(exp_kept)
        act_by_chapter.append(act_kept)
    a, b = _book_stream(exp_by_chapter), _book_stream(act_by_chapter)
    if not a and not b:
        return []
    ta, tb = [t for t, _ in a], [t for t, _ in b]
    covered: set[str] = set()
    for ch in chapters:
        for d in ch.differences:
            said = " ".join([d.get("summary", "")] + list(d.get("gone") or []) + list(d.get("extra") or []))
            for key in ("exp", "act"):
                el = d.get(key)
                if el is not None and el.text:
                    said += " " + el.text
            covered |= set(_TOKEN_RE.findall(_normalise(said)))
    solid_a, solid_b = "".join(ta), "".join(tb)
    count_a, count_b = Counter(ta), Counter(tb)
    stream_a, stream_b = f" {' '.join(ta)} ", f" {' '.join(tb)} "

    # The plain-text-only streams above are exactly what should NOT decide
    # "is this printed on the other side" on their own: the two documents
    # often classify the very same words differently (a caption a table cell
    # in Production, a bare paragraph in Staging), so a run excluded from one
    # side's stream by that side's own table/figure detection would wrongly
    # look absent. Read straight off the page instead - the raw text layer
    # does not care how element collection classified anything.
    def raw_page_text(doc: fitz.Document) -> str:
        return " " + " ".join(
            " ".join(_TOKEN_RE.findall(_normalise(doc[p].get_text("text"))))
            for p in range(doc.page_count)
        ) + " "

    raw_exp, raw_act = raw_page_text(expected), raw_page_text(actual)

    out: list[tuple["Chapter", dict]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for run, start, other, solid, counts, source, kind, raw_other in (
            (ta[i1:i2], i1, stream_b, solid_b, count_b, a, "missing", raw_act),
            (tb[j1:j2], j1, stream_a, solid_a, count_a, b, "added", raw_exp),
        ):
            if len(run) < _COVERAGE_MIN_TOKENS or _CJK_RE.search("".join(run)):
                continue
            if set(run) <= _PAGE_REF_WORDS:
                continue  # what is left of a page reference Staging drops (the user's rule)
            if f" {' '.join(run)} " in other:
                continue  # the exact phrase, printed elsewhere in the book
            if "".join(run) in solid:
                continue  # the same characters, only split or joined differently
            if len(run) >= _COVERAGE_BAG_MIN_TOKENS and (
                set(run) <= covered or all(counts[t] >= n for t, n in Counter(run).items())
            ):
                continue  # already reported, or every word printed in the book - just in another order or place
            # The same content, broken differently: a narrow table column
            # hyphenates ("Receiv-/er", "sup-/ply") and its cells read out in
            # another order, so the run is pieces of words the other side
            # prints whole. Only a run carrying such a piece is tested this
            # way - a run of real words missing outright still reports.
            vocab = set(counts)
            debris = [t for t in set(run) if counts[t] == 0 and _hyphen_piece(t, vocab)]
            if debris and all(counts[t] >= n or _hyphen_piece(t, vocab)
                              for t, n in Counter(run).items()):
                continue
            if f" {' '.join(run)} " in raw_other:
                continue  # on the page, just classified differently (a table cell, a heading)
            el = source[start][1]
            ch = owner.get(id(el))
            if ch is None:
                continue
            # A short run whose every word the other side prints inside THIS
            # chapter is the same content read in another order - a table's
            # cells taken column-first on one side and row-first on the other.
            # Only short runs: a whole sentence or paragraph that happens to
            # reuse the chapter's vocabulary is still a real loss.
            if len(run) <= _COVERAGE_REORDER_MAX_TOKENS:
                pages = (ch.act_pages if kind == "missing" else ch.exp_pages) or []
                here = _chapter_word_counts(actual if kind == "missing" else expected, tuple(pages))
                if all(here[t] >= n for t, n in Counter(run).items()):
                    continue
            # Already reported as part of a sentence one side lacks: the same
            # loss, not a second finding pinned at the chapter's first page.
            side = "exp" if kind == "missing" else "act"
            phrase = f" {' '.join(run)} "
            def names(d: dict) -> str:
                if d.get("type") == "text":
                    runs = d.get("gone" if kind == "missing" else "extra") or []
                    return " " + " ".join(runs) + " "
                if d.get("type") == kind and d.get(side) is not None:
                    return " " + " ".join(_TOKEN_RE.findall(_normalise(getattr(d[side], "text", "") or ""))) + " "
                return ""
            if any(phrase in names(d) for d in ch.differences):
                continue
            other_doc = actual if kind == "missing" else expected
            other_pages = (ch.act_pages if kind == "missing" else ch.exp_pages) or []
            if _ocr_confirms(other_doc, other_pages, run):
                continue  # baked into the other side's own artwork as pixels
            printed = _printed_form(" ".join(run), [el.text or ""])
            shown = printed if len(printed) <= 150 else printed[:149] + "…"
            doc = expected if kind == "missing" else actual
            place = Element(kind=KIND_TEXT, text=shown, key=_normalise(printed),
                            boxes=_phrase_boxes(doc, el, printed), section=el.section)
            covered |= set(run)
            diff = {
                "type": kind, "kind": KIND_TEXT, "section": el.section,
                "summary": (f"Text missing in Staging — “{shown}” is printed in Production and not anywhere "
                            f"in Staging."
                            if kind == "missing" else
                            f"Text extra in Staging — “{shown}” is printed in Staging and not anywhere in "
                            f"Production."),
                "detail": "",
                "exp": place if kind == "missing" else None,
                "act": place if kind == "added" else None,
            }
            # Where to put a marker on the side with nothing to box: this
            # content's own section may not even have a matched topic (front
            # matter before the first shared heading, say), so the report
            # cannot always place it from the topic alone - the chapter's own
            # page range on that side always exists.
            if kind == "missing" and ch.act_pages:
                diff["act_anchor"] = (ch.act_pages[0], 0.0)
            if kind == "added" and ch.exp_pages:
                diff["exp_anchor"] = (ch.exp_pages[0], 0.0)
            out.append((ch, diff))
    return out


_WRAP_MIN_TOKENS = 3        # a run this long printed in order on the other side is only wrapped
WRAP_DROPPED: Counter = Counter()


def _in_stream(text: str, stream: str) -> bool:
    toks = _TOKEN_RE.findall(_normalise(text or ""))
    return len(toks) >= _WRAP_MIN_TOKENS and f" {' '.join(toks)} " in stream


def _wrap_only(d: dict, exp_words: str, act_words: str) -> bool:
    """True when every word a difference calls missing or extra is printed, in
    order, in the other side's copy of the same topic - the text only wrapped
    at another point (another width, cell, callout or page). Logged per check
    in `WRAP_DROPPED`, so a joiner that missed it can be found."""
    kind = d.get("type")
    if kind == "text":
        gone, extra = d.get("gone") or [], d.get("extra") or []
        hit = bool(gone or extra) and all(_in_stream(r, act_words) for r in gone) \
            and all(_in_stream(r, exp_words) for r in extra)
    elif kind in ("missing", "table-row-missing") and d.get("kind") != KIND_FIGURE and d.get("exp") is not None:
        hit = _in_stream(d["exp"].text, act_words)
    elif kind in ("added", "table-row-added") and d.get("kind") != KIND_FIGURE and d.get("act") is not None:
        hit = _in_stream(d["act"].text, exp_words)
    else:
        return False
    if hit:
        WRAP_DROPPED[kind] += 1
    return hit


_QUOTED_RE = re.compile(r"“([^”]+)”")


def _printed_form(phrase: str, texts: list[str]) -> str:
    """`phrase` as one of the documents prints it - its case, its spacing -
    when a normalised comparison produced it. Unchanged when not found."""
    words = [w for w in phrase.replace("…", " ").split() if w and w not in ("/", "|", "·")]
    if not words:
        return phrase
    # Between two words the page may print a bullet, a tab or a cell break.
    pattern = r"[\s•/|·\-–]+".join(re.escape(w) for w in words)
    for text in texts:
        m = re.search(pattern, text or "", re.IGNORECASE)
        if m:
            return _WS_RE.sub(" ", m.group(0)).strip() + ("…" if phrase.endswith("…") else "")
    return phrase


def _box_text(doc: fitz.Document | None, el: Element | None) -> list[str]:
    """The element's own text plus what the page prints inside its boxes - a
    table row rebuilt from normalised cells keeps no printed text of its own."""
    if el is None:
        return []
    out = []
    if doc is not None:
        # The page first: an element's own text may already be normalised.
        for page_index, bbox in el.boxes[:6]:
            try:
                out.append(doc[page_index].get_text("text", clip=fitz.Rect(*bbox)))
            except Exception:
                pass
    return out + ([el.text] if el.text else [])


def _quote_as_printed(diff: dict, expected: fitz.Document | None = None,
                      actual: fitz.Document | None = None) -> None:
    """Comments quote the text as PRINTED: "HEVC(H26 5)", not the lower-cased
    key "hevc(h26 5)" the comparison ran on."""
    exp_texts = _box_text(expected, diff.get("exp"))
    act_texts = _box_text(actual, diff.get("act"))
    both = exp_texts + act_texts
    if not both:
        return
    diff["summary"] = _QUOTED_RE.sub(lambda m: f"“{_printed_form(m.group(1), both)}”", diff.get("summary") or "")
    if diff.get("gone"):
        diff["gone"] = [_printed_form(r, exp_texts + act_texts) for r in diff["gone"]]
    if diff.get("extra"):
        diff["extra"] = [_printed_form(r, act_texts + exp_texts) for r in diff["extra"]]


_WORD_RE = re.compile(r"\s+")
# diff-match-patch (Myers' algorithm + semantic-boundary cleanup - the same
# family of diff used by most text-diff tools, e.g. diffchecker.com) works on
# characters; the standard technique for a WORD-level diff with it is to map
# each unique word to one private-use-area character first; run the char-diff
# on those; then map back. One shared instance since it is stateless per call.
_WORD_DMP = diff_match_patch()

# Ordinary sentence punctuation only - never a symbol that carries its own
# meaning. "©", "®", "™", "°", "±", "%", a currency sign: losing one of
# these is losing real content (a missing "©" is a missing copyright notice;
# "100°C" missing its "°" is a different, wrong number), not a trivial
# formatting difference the way a dropped comma or a period swapped for a
# dash can be. `_TOKEN_RE` (word characters only) used to gate this and
# swallowed every one of those symbols along with genuine punctuation, since
# neither is alnum - this explicit set is deliberately narrow instead.
# "/" joins two words ("and/or", "components/equipment") far more often than
# it means anything on its own, and sits right where a line wraps often
# enough that a compound term wrapping at the slash on one side and not the
# other left a stray space next to it - "components/ equipment" against
# "components/equipment" - which is nothing but that wrap, not a changed
# symbol, and must not be judged as one just because "/" is attached to it.
_TRIVIAL_PUNCT_CHARS = set(".,;:!?'\"‘’“”()[]{}–—-…*/")


_CALLOUT_WORD_RE = re.compile(r"(?i)note|notes|tip|tips|warning|caution|important|attention|danger")
_BULLET_GLYPH_RE = re.compile(r"[•●○◦▪▫■□►▸‣⁃∙]")
_MARK_SPLIT_RE = re.compile(r"([^\W_]+)", re.UNICODE)


def _mark_change(where: str, x: str, y: str) -> str | None:
    """One gap's difference in plain words: which marks Staging lacks or adds,
    or - the marks being the same - a space missing or extra."""
    ca, cb = Counter(x.replace(" ", "")), Counter(y.replace(" ", ""))
    lost, added = "".join((ca - cb).elements()), "".join((cb - ca).elements())
    said = []
    if lost:
        said.append("“" + " ".join(lost) + "” missing in Staging")
    if added:
        said.append("“" + " ".join(added) + "” extra in Staging")
    if not said:
        if x.count(" ") > y.count(" "):
            said.append("space missing in Staging")
        elif y.count(" ") > x.count(" "):
            said.append("extra space in Staging")
        elif x != y:
            said.append(f"“{x.strip()}” in Production, “{y.strip()}” in Staging")
    return f"{where}: {', '.join(said)}" if said else None


_LEAD_DASH_RE = re.compile(r"[-\u2013\u2014](?=\s*$)")


def mark_changes(a_text: str, b_text: str, wraps_a=(), wraps_b=()) -> list[str]:
    """Punctuation, quote marks and spaces that differ between the SAME words
    on both sides: a period dropped after "drive", quotes lost around a
    cross-reference, a space missing in "drive.Next". Words themselves are the
    word check's business - only what sits between two words both sides
    print, in the same order, is compared here. Curly and straight quotes
    are the same mark; line wrapping is only whitespace and never counts:
    `wraps_a` / `wraps_b` name the (word, next word) pairs each side prints
    across a line or page break, where the space is the wrap's own - Production
    breaking "Support." / "BenQ.com" over two lines is not a space in the text."""
    def parts(text: str) -> tuple[list[str], list[str]]:
        text = _PAGE_REF_RE.sub("", _CONTROL_RE.sub("", text or "")).translate(_QUOTES)
        text = _INLINE_CALLOUT_RE.sub(" ", text)  # a callout glyph one side draws and the other sets
        # The list marker itself, and the gap between it and the step's first
        # word: "2.Remove" against "2. Remove" is the marker set tight, not a
        # space added to the sentence. Marker style is the marker checks'
        # business (see `_LIST_MARKER_TOKEN_RE`); reported here it turned every
        # numbered step in the manual into a spacing finding of its own.
        text = _LIST_MARKER_TOKEN_RE.sub(" ", text)
        # A line break after a hyphen or slash ("power- saving", "and/ or",
        # "support. benq") is wrapping, not a space in the text.
        text = re.sub(r"(\w[-/])\s+(?=\w)|(\w\.)\s+(?=[a-z])", lambda m: m.group(1) or m.group(2), text)
        bits = _MARK_SPLIT_RE.split(text)
        # Bullet glyphs are list markers, the marker checks' business - never punctuation.
        words, gaps = bits[1::2], [re.sub(r"\s+", " ", _BULLET_GLYPH_RE.sub(" ", g)) for g in bits[0::2]]
        return words, gaps  # gaps[i] sits before words[i]; gaps[-1] after the last word

    wa, ga = parts(a_text)
    wb, gb = parts(b_text)
    out: list[str] = []
    fold_a, fold_b = [w.casefold() for w in wa], [w.casefold() for w in wb]
    wrap_a, wrap_b = set(wraps_a or ()), set(wraps_b or ())
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=fold_a, b=fold_b, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        # The gaps strictly between the matched words, plus the one after the
        # run when both runs end their text (a sentence's final period).
        for k in range(1, i2 - i1):
            x, y = ga[i1 + k], gb[j1 + k]
            if x != y and x.replace(" ", "") == y.replace(" ", ""):
                # Only a space differs: not one when that side breaks the line there.
                if (" " in x and (fold_a[i1 + k - 1], fold_a[i1 + k]) in wrap_a) \
                        or (" " in y and (fold_b[j1 + k - 1], fold_b[j1 + k]) in wrap_b):
                    continue
            if x != y and _LEAD_DASH_RE.sub("", x).replace(" ", "") == _LEAD_DASH_RE.sub("", y).replace(" ", ""):
                # A dash opening a new line is that line's list marker ("- a
                # wireless device..." against Staging's bullet), not punctuation.
                if ((fold_a[i1 + k - 1], fold_a[i1 + k]) in wrap_a and _LEAD_DASH_RE.search(x)) \
                        or ((fold_b[j1 + k - 1], fold_b[j1 + k]) in wrap_b and _LEAD_DASH_RE.search(y)):
                    continue
            if x != y and not _CALLOUT_WORD_RE.fullmatch(wa[i1 + k - 1]):  # "Tip:" / "Note:" is callout styling
                out.append(_mark_change(f"between “{wa[i1 + k - 1]}” and “{wa[i1 + k]}”", x, y))
        if i2 == len(wa) and j2 == len(wb):
            x, y = ga[-1].strip(), gb[-1].strip()
            if x != y:
                out.append(_mark_change(f"after “{wa[-1]}”", x, y))
        if i1 == 0 and j1 == 0:
            x, y = ga[0].strip(), gb[0].strip()
            # a leading list marker / bullet is the marker checks' business
            if x != y and any(ch in "\"'([" for ch in x + y):
                out.append(_mark_change(f"before “{wa[0]}”", x, y))
    out = [m for m in out if m]
    return out


def _is_trivial_punct(text: str) -> bool:
    """True when `text` is nothing but ordinary punctuation and whitespace -
    safe to treat as unchanged regardless of which side has it. False for
    anything else, including a symbol that is not a letter or digit but is
    not mere punctuation either."""
    stripped = text.replace(" ", "")
    return not stripped or all(ch in _TRIVIAL_PUNCT_CHARS for ch in stripped)


def _only_bullet_glyphs(text: str) -> bool:
    """The whole change is list-bullet characters - Production's "\u2022" against
    Staging's "\u25cf". That is a marker-style change across a list, which
    `bullet_glyph_changes` reports once for the whole list; reported again here
    it reads as a CONTENT difference on every bullet, and the box lands on the
    item's words rather than on the marker that actually changed."""
    if not (text or "").strip():
        return False
    return not _BULLET_GLYPH_RE.sub("", text).strip()


def _has_symbol(text: str) -> bool:
    """A real symbol - ©, ®, °, ±, %, a currency mark, anything that is
    neither a letter/digit nor ordinary sentence punctuation - appears in
    `text`. Used to tell "the wording is the same, only a comma moved" apart
    from "the wording is the same, but a copyright mark disappeared" - the
    second one is content lost, not a formatting nuance, however small it
    looks on the page."""
    return any(not ch.isspace() and not ch.isalnum() and ch not in _TRIVIAL_PUNCT_CHARS for ch in text)


def _unwrap_hyphens(words: list[str], other: list[str]) -> list[str]:
    """A word broken over a line end ("man-" / "agement") put back together,
    the way the other side prints it: "management", or "power-saving" when
    the hyphen is the word's own. Wrapping is never a wording change."""
    printed = {w.casefold().strip(".,;:()\"“”") for w in other}
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.endswith("-") and len(w) > 1 and i + 1 < len(words) and words[i + 1][:1].islower():
            joined, kept = w[:-1] + words[i + 1], w + words[i + 1]
            core = lambda x: x.casefold().strip(".,;:()\"“”")  # noqa: E731
            if core(joined) in printed:
                out.append(joined)
                i += 2
                continue
            if core(kept) in printed:
                out.append(kept)
                i += 2
                continue
        out.append(w)
        i += 1
    return out


def _word_diff(before: str, after: str) -> list[dict]:
    """The wording change, word by word, for the report to mark up."""
    # List markers ("1.", "a.", "•") are the marker checks' business: one side
    # prints them in the text run and the other beside it, which is no change
    # in wording and must never be boxed as a missing word.
    a = [w for w in _WORD_RE.split(before.strip()) if w and not _LIST_MARKER_TOKEN_RE.fullmatch(w)]
    b = [w for w in _WORD_RE.split(after.strip()) if w and not _LIST_MARKER_TOKEN_RE.fullmatch(w)]
    a, b = _unwrap_hyphens(a, b), _unwrap_hyphens(b, a)
    # Curly and straight quotes are the same character to a reader: “display’s”
    # against "display's" is no wording change.
    fa, fb = [w.translate(_QUOTES) for w in a], [w.translate(_QUOTES) for w in b]

    word_to_char: dict[str, str] = {}
    orig_a: dict[str, str] = {}
    orig_b: dict[str, str] = {}

    def encode(folded: list[str], original: list[str], orig_map: dict[str, str]) -> str:
        chars = []
        for fw, ow in zip(folded, original):
            c = word_to_char.get(fw)
            if c is None:
                c = chr(0xE000 + len(word_to_char))
                word_to_char[fw] = c
            chars.append(c)
            orig_map.setdefault(c, ow)
        return "".join(chars)

    diffs = _WORD_DMP.diff_main(encode(fa, a, orig_a), encode(fb, b, orig_b))
    _WORD_DMP.diff_cleanupSemantic(diffs)

    ops: list[dict] = []
    i = 0
    while i < len(diffs):
        op, chunk = diffs[i]
        if op == 0:
            ops.append({"type": "equal", "text": " ".join(orig_a[c] for c in chunk)})
            i += 1
            continue
        # A maximal run of consecutive non-equal chunks is judged as one unit
        # (the same way SequenceMatcher's single "replace" span was), not
        # chunk by chunk - diff-match-patch can still hand back an adjacent
        # del then ins for what is really one substitution.
        run_del: list[str] = []
        run_ins: list[str] = []
        while i < len(diffs) and diffs[i][0] != 0:
            rop, rchunk = diffs[i]
            (run_del if rop == -1 else run_ins).extend(orig_a[c] if rop == -1 else orig_b[c] for c in rchunk)
            i += 1
        # A run of ordinary punctuation or spacing alone ("." / "—") is never
        # boxed: boxes and insertion marks are for missing words - but never
        # for a missing SYMBOL either, which is content, not formatting
        # (see `_is_trivial_punct`). "&" is a word ("og", "and"), never
        # punctuation: "& (EU)" -> "og (EU)" is a change.
        if "&" not in run_del and "&" not in run_ins and _is_trivial_punct(" ".join(run_del + run_ins)):
            if run_del or run_ins:
                ops.append({"type": "equal", "text": " ".join(run_del or run_ins)})
            continue
        if run_del:
            ops.append({"type": "del", "text": " ".join(run_del)})
        if run_ins:
            ops.append({"type": "ins", "text": " ".join(run_ins)})
    return ops


# --- chapters --------------------------------------------------------------


def _skipped_heading(entry: TocEntry) -> bool:
    """A heading whose section is out of scope: a table of contents, a Q&A or
    FAQ section, or a regulatory declaration - never a heading `_mark_front_matter`
    excluded only for not lining up as a shared chapter boundary. That flag
    reuses the same `excluded` attribute for an unrelated purpose, and reading
    it here fed front matter straight into `skip_spans`: real prose (a
    "Copyright and Disclaimer" wrapper heading nesting the very sections the
    other document bookmarks at the top level, say) was wiped from
    `collect_elements` even on the page range `chapter_pairs` deliberately
    swept into chapter 0 to keep comparing it."""
    return is_excluded_heading(entry.title or "")


_QA_FAQ_SKIP_MAX_PAGES = 1  # a topic-jump index or trailing blurb runs at most this far


def skip_spans(entries: list[TocEntry], page_count: int | None = None) -> list[tuple]:
    """`(start, end, title)` for every table-of-contents / Q&A section at ANY
    level, running to the next heading at the same level or above that is not
    itself skipped - so a "Q&A" sub-section inside a chapter drops out of that
    chapter's comparison with everything under it, and the rest of the chapter
    is still compared.

    A Q&A/FAQ-titled heading only counts when its own section is short - at
    most a page beyond where it starts, the "trailing question and answer" or
    "topic-jump index that may run onto one continuation page" this exclusion
    was built for. A real, substantial FAQ/Troubleshooting section spanning
    several pages of genuine step-by-step answers is content a reader needs
    compared, not a directory to skip outright - title alone cannot tell the
    two apart, but length can. (A table of contents/index or a regulatory
    declaration is excluded at any length: those are never real prose.)
    """
    ordered = sorted(entries, key=lambda e: (e.page, e.y))
    out: list[tuple] = []
    for i, entry in enumerate(ordered):
        if not _skipped_heading(entry):
            continue
        end = None
        for later in ordered[i + 1:]:
            if later.level <= entry.level and not _skipped_heading(later):
                end = (later.page, later.y)
                break
        if is_qa_faq_heading(entry.title):
            end_page = end[0] if end else (page_count - 1 if page_count else entry.page)
            if end_page - entry.page > _QA_FAQ_SKIP_MAX_PAGES:
                continue
        out.append(((entry.page, entry.y - 2.0), end, entry.title))
    return out


def _in_spans(page: int, y: float, spans: list[tuple]) -> str | None:
    for start, end, title in spans:
        if (page, y) >= start and (end is None or (page, y) < end):
            return title
    return None


_TOC_PAGE_MIN_ENTRIES = 3


def _listing_page(doc: fitz.Document, page_index: int) -> str | None:
    """"Table of contents" or "Q&A index" when this whole page is one of those
    listings, whether or not anything bookmarks it - a contents page is often
    just printed, and a Q&A index can run onto an untitled continuation page."""
    try:
        texts = [b[4] for b in doc[page_index].get_text("blocks") if b[4].strip()]
    except Exception:
        return None
    if _looks_like_qa_index_page(texts):
        return "Q&A index"
    # Counted line by line: a contents page can print every entry in one block.
    lines = [ln.strip() for t in texts for ln in t.splitlines() if ln.strip()]
    if sum(1 for t in texts if looks_like_toc_listing(t)) >= _TOC_PAGE_MIN_ENTRIES \
            or sum(1 for ln in lines if looks_like_toc_listing(ln)) >= _TOC_PAGE_MIN_ENTRIES:
        return "Table of contents"
    return None


# --- what a PRINTED contents page itself lists ------------------------------
#
# `_listing_page` correctly keeps a printed "Table of Contents" page out of
# every chapter's own span: the two documents paginate differently, so the
# NUMBER beside every single entry always differs, and comparing the page as
# ordinary prose would flood the report with false "page number changed"
# findings. But the page number is not the only thing printed there - the
# ENTRY NAMES are a real list of sections a reader sees before the manual
# even starts, and Staging silently gaining or dropping one (a "Disclaimer"
# entry that only its own copy lists, say) is real content one document
# promises and the other does not - invisible everywhere else, since the
# whole page is skipped rather than compared. Checked once for the whole
# document, page numbers stripped, never inside a normal chapter span.

_TOC_ENTRY_TRAILER_RE = re.compile(r"[.\s]{2,}\d{1,4}\s*$")
_TOC_ENTRY_SPLIT_RE = re.compile(r"\.{3,}\s*\d{1,4}(?=\s|$)")  # one entry's dot leader and page number
_TOC_LISTED = 6  # entries shown by name before the rest fold into "N more"
_TOC_RENAMED_RATIO = 0.9  # two entries this alike are one entry, changed


def _toc_page_titles(doc: fitz.Document, page_index: int) -> list[str]:
    """Every entry name printed on this contents page, its own trailing dot
    leader and page number stripped, in reading order."""
    try:
        blocks = [b[4] for b in doc[page_index].get_text("blocks") if b[4].strip()]
    except Exception:
        return []
    out: list[str] = []
    for block in blocks:
        # An entry too long for one line wraps, and only its last line carries
        # the dot leader and page number: "Download SmartRemoote for BenQ
        # Projector app to your" / "mobile device.....37" is one entry, not
        # "mobile device".
        pending: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if not looks_like_toc_listing(line):
                pending.append(line)
                continue
            # One extracted line can hold several entries, each ending in its
            # own dot leader and page number ("...via LAN.....33 Logging into
            # ... wireless network.....35").
            pieces = [p.strip(" .") for p in _TOC_ENTRY_SPLIT_RE.split(line) if p.strip(" .")]
            for n, piece in enumerate(pieces):
                title = (" ".join(pending + [piece]) if n == 0 else piece).strip(" .")
                if title and title.lower() not in ("table of contents", "table of content"):
                    out.append(title)
            pending = []
    return out


def printed_toc_changes(expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """Every printed contents page's own entries, Production against Staging,
    across the whole document."""
    exp_titles: list[str] = []
    for page_index in range(expected.page_count):
        if _listing_page(expected, page_index) == "Table of contents":
            exp_titles.extend(_toc_page_titles(expected, page_index))
    act_titles: list[str] = []
    for page_index in range(actual.page_count):
        if _listing_page(actual, page_index) == "Table of contents":
            act_titles.extend(_toc_page_titles(actual, page_index))
    if not exp_titles and not act_titles:
        return []
    if not exp_titles or not act_titles:
        # One side has no printed contents page at all - not a handful of
        # entries changed, the WHOLE page is new (or gone), reported once.
        side = "Staging" if exp_titles else "Production"
        count = len(exp_titles or act_titles)
        return [{
            "type": "section-added" if exp_titles else "section-missing", "kind": KIND_TEXT,
            "summary": (f"Table of contents printed in {'Production' if exp_titles else 'Staging'} only — "
                        f"{'Staging has' if exp_titles else 'Production has'} no such page at all, "
                        f"listing {count} entries."),
            "detail": ", ".join((exp_titles or act_titles)[:_TOC_LISTED]),
        }]
    exp_keys = [_normalise(t) for t in exp_titles]
    act_keys = [_normalise(t) for t in act_titles]
    gone = [t for t, k in zip(exp_titles, exp_keys) if k not in act_keys]
    extra = [t for t, k in zip(act_titles, act_keys) if k not in exp_keys]
    if not gone and not extra:
        return []
    out: list[dict] = []
    # An entry reworded or respelled ("SmartRemoote" -> "SmartRemote") is one
    # changed entry, not one dropped and another added.
    renamed: list[tuple[str, str]] = []
    for t in list(gone):
        best = max(extra, key=lambda e: _ratio(_normalise(t), _normalise(e)), default=None)
        if best is not None and _ratio(_normalise(t), _normalise(best)) >= _TOC_RENAMED_RATIO:
            renamed.append((t, best))
            gone.remove(t)
            extra.remove(best)
    if renamed:
        shown = "; ".join(f"“{t}” → “{e}”" for t, e in renamed[:_TOC_LISTED])
        more = f" and {len(renamed) - _TOC_LISTED} more" if len(renamed) > _TOC_LISTED else ""
        out.append({
            "type": "text", "kind": KIND_TEXT,
            "summary": f"Table of contents entry changed in Staging — {shown}{more}.",
            "detail": "",
        })
    if gone:
        shown = ", ".join(f"“{t}”" for t in gone[:_TOC_LISTED])
        more = f" and {len(gone) - _TOC_LISTED} more" if len(gone) > _TOC_LISTED else ""
        out.append({
            "type": "text", "kind": KIND_TEXT,
            "summary": (f"Table of contents drops an entry in Staging — Production's printed contents page "
                        f"lists {shown}{more}, Staging's does not."),
            "detail": "",
        })
    if extra:
        shown = ", ".join(f"“{t}”" for t in extra[:_TOC_LISTED])
        more = f" and {len(extra) - _TOC_LISTED} more" if len(extra) > _TOC_LISTED else ""
        out.append({
            "type": "text", "kind": KIND_TEXT,
            "summary": (f"Table of contents adds an entry in Staging — its printed contents page lists "
                        f"{shown}{more} that Production's does not."),
            "detail": "",
        })
    return out


def _top_level(entries: list[TocEntry]) -> int:
    """The document's L1 - the shallowest level its heading list actually
    uses, so a document whose outline starts at level 2 still yields chapters.
    """
    levels = [e.level for e in entries if not e.excluded]
    return min(levels) if levels else 1


@dataclass
class ChapterSpan:
    """One L1 heading, and the stretch of each document it covers."""

    exp_entry: TocEntry
    act_entry: TocEntry
    exp_span: tuple          # ((start_page, start_y), (end_page, end_y) | None)
    act_span: tuple
    exp_only_tops: list = field(default_factory=list)  # L1 headings only this side has
    act_only_tops: list = field(default_factory=list)


def chapter_pairs(expected: fitz.Document, actual: fitz.Document) -> list[ChapterSpan]:
    """Every L1 heading both documents have, with the stretch it covers on each
    side.

    Bounded by the next MUTUALLY MATCHED L1 heading, not simply by the next L1
    heading in each document. Production bookmarks "Settings" as a chapter of
    its own where Staging keeps that material inside "Basic operations": bound
    each side by its own next L1 and the two spans stop describing the same
    stretch of manual - seven Production pages against twenty Staging ones -
    and every element in between reads as missing or added. Bounded by the next
    SHARED heading, both sides cover the same material however each one chooses
    to bookmark it, and the heading one side is missing is reported as its own
    difference instead of derailing the comparison.
    """
    exp_entries, act_entries = resolve_entries(expected, actual)
    if not exp_entries or not act_entries:
        return []
    # A printed table of contents or Q&A / FAQ index is not a chapter to
    # validate: it is a list of page numbers pointing at the real chapters,
    # checked on its own by the TOC comparison, and it differs between any two
    # documents that paginate differently. Not a chapter, not a boundary.
    exp_tops = [e for e in exp_entries if e.level == _top_level(exp_entries) and not _skipped_heading(e)]
    act_tops = [e for e in act_entries if e.level == _top_level(act_entries) and not _skipped_heading(e)]
    if not exp_tops or not act_tops:
        return []

    matched = [
        m for m in match_toc_entries(exp_tops, act_tops)
        if m.expected_index is not None and m.actual_index is not None
    ]
    if not matched:
        return []
    # In true reading order on the Production side; the two documents list
    # their chapters in the same order, and a chapter's span has to run
    # forwards.
    matched.sort(key=lambda m: (exp_tops[m.expected_index].page, exp_tops[m.expected_index].y))

    out: list[ChapterSpan] = []
    for place, m in enumerate(matched):
        exp_entry = exp_tops[m.expected_index]
        act_entry = act_tops[m.actual_index]
        nxt = matched[place + 1] if place + 1 < len(matched) else None
        exp_end = (
            (exp_tops[nxt.expected_index].page, exp_tops[nxt.expected_index].y) if nxt else None
        )
        act_end = (
            (act_tops[nxt.actual_index].page, act_tops[nxt.actual_index].y) if nxt else None
        )
        exp_start = (exp_entry.page, exp_entry.y)
        act_start = (act_entry.page, act_entry.y)
        if place == 0:
            # Front matter this document's own outline nests a level deeper
            # than the other's - Production bookmarks "Copyright"/"Disclaimer"
            # as their own top-level entries, Staging as children of an early
            # heading - never becomes an L1-to-L1 boundary at all, and sitting
            # BEFORE the first one that does, its content is invisible to every
            # chapter's span: never reported missing, never compared, just
            # silently skipped. Starting the very first chapter at the top of
            # its own SECOND page instead of at its own heading sweeps that
            # front matter in (page two onward - Copyright, Disclaimer, Safety
            # Warnings, whatever either side bookmarks differently), so a
            # genuine difference there is still caught rather than passing the
            # whole run in silence.
            #
            # Page one itself is never swept in this way, deliberately: it is
            # the cover - the title, product photo, logo, legal boilerplate a
            # reader sees before any real section - and it is pure design,
            # not prose. It legitimately differs release to release and
            # language to language (a rebrand, a new photo, a different
            # legal line for a different market), and comparing it word for
            # word only ever produces confusing, mis-attributed findings
            # (a cover title's own wording reported as belonging to whatever
            # real chapter happens to start next).
            exp_start = (min(1, exp_entry.page), 0.0)
            act_start = (min(1, act_entry.page), 0.0)

        def within(tops, start, end, skip) -> list[str]:
            return [
                t.title for t in tops
                if t is not skip and (t.page, t.y) > start and (end is None or (t.page, t.y) < end)
            ]

        out.append(ChapterSpan(
            exp_entry=exp_entry,
            act_entry=act_entry,
            exp_span=(exp_start, exp_end),
            act_span=(act_start, act_end),
            exp_only_tops=within(exp_tops, exp_start, exp_end, exp_entry),
            act_only_tops=within(act_tops, act_start, act_end, act_entry),
        ))
    return out


_SPEC_FRAGMENT_MAX_LETTER_RUN = 3  # a letter run this short is a unit suffix (C, m, kg), not a word
_ARTWORK_CAPTION_REACH = 60.0      # pt: a spec fragment printed this close to a picture is its caption


def _looks_like_spec_fragment(text: str) -> bool:
    """A short run of numbers and units - a temperature/humidity/altitude
    range printed under an icon, not a sentence. Every run of letters in it is
    short enough to be a unit suffix (`C`, `m`, `kg`, `Hz`), never a real
    word, and it carries more than one number - a single stray digit in an
    ordinary sentence must not qualify."""
    if _CJK_RE.search(text) or len(re.findall(r"\d+", text)) < 2:
        return False
    return all(len(w) <= _SPEC_FRAGMENT_MAX_LETTER_RUN for w in re.findall(r"[A-Za-z]+", text))


def _near_a_figure(el: "Element", figures: list["Element"], reach: float = _ARTWORK_CAPTION_REACH) -> bool:
    """A picture on the same page, close enough to `el` to be what it
    captions - within reach on any side, not only directly below."""
    ex0, ey0, ex1, ey1 = el.bbox
    for f in figures:
        if f.page != el.page:
            continue
        fx0, fy0, fx1, fy1 = f.bbox
        if ex1 < fx0 - reach or ex0 > fx1 + reach:
            continue
        if ey0 > fy1 + reach or ey1 < fy0 - reach:
            continue
        return True
    return False


_TABLE_ROWS_PRINTED = 0.8  # share of an unpaired table's rows the other side must print nearby
_SQUASHED_PAGES: dict[tuple, str] = {}


def _nearby_squashed(doc: fitz.Document, pages: list[int]) -> str:
    """The text of these pages and two either side, as one run of letters and
    digits - page references and bracketed cross-references taken out."""
    want = tuple(sorted({p + d for p in pages for d in range(-2, 3) if 0 <= p + d < doc.page_count}))
    key = (doc_key(doc), want)
    if key not in _SQUASHED_PAGES:
        _SQUASHED_PAGES[key] = "".join(_audit_tokens(" ".join(doc[p].get_text("text") for p in want)))
    return _SQUASHED_PAGES[key]


def _drop_printed_nearby(chapter: "Chapter", expected: fitz.Document, actual: fitz.Document) -> None:
    """A "missing" / "added" finding whose words are printed on the OTHER
    document's own pages RIGHT AROUND WHERE THIS ELEMENT ITSELF SITS is not
    reported: a table or list split across a page break moved that row a
    page (or the row's own picture caption doubled its text and broke the
    normal pairing), it did not delete it. Checked on a small window around
    the element's own page - one page either way - not the chapter's whole
    span, so a genuinely missing line is not waved through just because the
    same short phrase happens to print somewhere else in the chapter.

    An exact run repeated back to back in the element's own text ("Quick
    Start Guide Quick Start Guide") is read once, not twice, before the
    search: two overlapping text layers - the row's own label and a caption
    laid over its picture - print the same words once each, and the other
    side need only print them once to still hold this row.
    """
    exp_start = chapter.exp_pages[0] if chapter.exp_pages else None
    act_start = chapter.act_pages[0] if chapter.act_pages else None
    page_offset = (act_start - exp_start) if exp_start is not None and act_start is not None else 0

    exp_cache: dict[tuple[int, ...], str] = {}
    act_cache: dict[tuple[int, ...], str] = {}

    def page_words(doc: fitz.Document, pages: list[int], cache: dict[tuple[int, ...], str]) -> str:
        key = tuple(pages)
        if key not in cache:
            # Two pages either way: the page offset between the documents is
            # one number for the chapter, and drifts by a page or two along it.
            want = sorted({p + d for p in pages for d in (-2, -1, 0, 1, 2) if 0 <= p + d < doc.page_count})
            cache[key] = " " + " ".join(
                " ".join(_TOKEN_RE.findall(_normalise(doc[p].get_text("text")))) for p in want
            ) + " "
        return cache[key]

    lost_types = {"missing", "table-row-missing"}
    gained_types = {"added", "table-row-added"}
    exp_figures = [e for e in chapter.exp_elements if e.kind == KIND_FIGURE]
    act_figures = [e for e in chapter.act_elements if e.kind == KIND_FIGURE]
    keep: list[dict] = []
    for d in chapter.differences:
        t = d.get("type")
        drop = False
        if d.get("kind") != KIND_FIGURE and (t in lost_types or t in gained_types):
            lost = t in lost_types
            el = d.get("exp") if lost else d.get("act")
            # A short label captioning a nearby icon is never excused this
            # way: the word merely printing somewhere else in the chapter's
            # running prose ("Battery" inside an unrelated disposal sentence)
            # is not evidence the icon still carries its own caption.
            own_figures = exp_figures if lost else act_figures
            if el is not None and el.text and len(el.text) <= _LAYOUT_SHORT_CHARS and _near_a_figure(
                el, own_figures, reach=_HEADING_CAPTION_REACH
            ):
                # The word merely printing somewhere else in the chapter's
                # running prose ("Battery" inside an unrelated disposal
                # sentence) is not evidence the icon still carries its own
                # caption - but Tesseract reading these same words specifically
                # INSIDE an embedded picture on the other side (not just
                # anywhere on the page) is: that is exactly what a diagram
                # flattened to one raster image looks like.
                own_pages = sorted({p for p, _ in el.boxes}) or [el.page]
                valid = (chapter.act_pages if lost else chapter.exp_pages) or own_pages
                target = [min(max(p + (page_offset if lost else -page_offset), valid[0]), valid[-1])
                          for p in own_pages]
                other_doc = actual if lost else expected
                toks = _TOKEN_RE.findall(_normalise(el.text))
                if _ocr_confirms_in_image(other_doc, target, toks):
                    drop = True
                elif _ocr_confirms(other_doc, target, toks):
                    # A diagram redrawn with its callouts as vector paths
                    # (line-art outlines, not an embedded raster XObject) has
                    # no image for `_ocr_confirms_in_image` to restrict to and
                    # always reads as "no image here" - whole-page OCR is the
                    # only way to see pixels that were never a real XObject.
                    # Still gated on `_near_a_figure` above, so this is only
                    # trusted for a short caption already known to sit right
                    # beside a figure, not any missing text on the page.
                    drop = True
                el = None
            if el is not None and el.kind == KIND_TABLE and el.cells and not drop:
                # A table the two documents divide differently (Staging's third
                # piece holding rows 7-9 of one table and row 20 of another)
                # pairs with neither piece - yet every row of it is printed
                # there. Row by row, spacing, page references and section-link
                # wording aside.
                own_pages = sorted({p for p, _ in el.boxes}) or [el.page]
                valid = (chapter.act_pages if lost else chapter.exp_pages) or own_pages
                target = [min(max(p + (page_offset if lost else -page_offset), valid[0]), valid[-1])
                          for p in own_pages]
                other_doc = actual if lost else expected
                printed = _nearby_squashed(other_doc, target)
                # Cell by cell: one printed row can hold two tables' rows side
                # by side ("7. Digital zoom out" beside "20. Projector Assistant").
                rows = ["".join(_audit_tokens(c)) for row in el.cells for c in row if c]
                rows = [r for r in rows if len(r) >= 3]
                if rows and sum(1 for r in rows if r in printed) >= _TABLE_ROWS_PRINTED * len(rows):
                    drop = True
                el = None
            if el is not None and el.text:
                toks = _TOKEN_RE.findall(_normalise(el.text))
                half = len(toks) // 2
                if half and toks[:half] == toks[half : half * 2]:
                    toks = toks[:half]
                phrase = " ".join(toks)
                # A single short numeric token (a page number, a stray list
                # marker) is too coincidental to trust here - it prints
                # somewhere on almost every page - so it is never treated as
                # evidence this element is present elsewhere.
                if phrase and not (len(toks) == 1 and len(phrase) <= 3 and phrase.isdigit()):
                    own_pages = sorted({p for p, _ in el.boxes}) or [el.page]
                    valid = (chapter.act_pages if lost else chapter.exp_pages) or own_pages
                    target = [min(max(p + (page_offset if lost else -page_offset), valid[0]), valid[-1])
                              for p in own_pages]
                    other_doc = actual if lost else expected
                    cache = act_cache if lost else exp_cache
                    printed = page_words(other_doc, target, cache)
                    squashed = phrase.replace(" ", "")
                    if f" {phrase} " in printed or (
                            len(squashed) >= _SQUASHED_MIN_CHARS and squashed in printed.replace(" ", "")):
                        # ... or the same characters spaced or hyphenated
                        # differently ("Touchback" / "Touch-back").
                        drop = True
                    else:
                        # A single-word phrase is normally too coincidental for
                        # OCR to confirm on its own - but an all-caps label
                        # ("DP", "USB", "HDMI") is a distinctive abbreviation,
                        # not a common word that would turn up by chance.
                        distinctive = len(toks) == 1 and len(toks[0]) >= 2 and el.text.strip().isupper()
                        if _ocr_confirms(other_doc, target, toks, allow_single=distinctive):
                            drop = True
        if not drop:
            keep.append(d)
    if len(keep) == len(chapter.differences):
        return
    old = chapter.differences
    position = {id(d): k for k, d in enumerate(keep)}
    for row in chapter.rows:
        row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
        row["differences"] = [d for d in row.get("differences", []) if id(d) in position]
    chapter.differences = keep


def _drop_confirmed_content_changes(chapter: "Chapter", expected: fitz.Document, actual: fitz.Document) -> None:
    """A "Text content changed" finding is not reported when every one of its
    GONE words is printed on Staging's own page (just not captured as part of
    THIS comparable element) and every one of its EXTRA words is printed on
    Production's own page the same way.

    A numbered step's own title ("1. Remove the monitor stand.") sits beside a
    badge graphic drawn apart from it - the same pattern that puts a bare list
    marker in its own stray element (see `_looks_like_bare_marker`) can just as
    easily leave the WORDS beside that badge out of every comparable element
    entirely, so Production's title+body paragraph pairs against Staging's
    body alone and "loses" the title. The words are still right there on the
    page; only element collection failed to capture them as their own unit.

    Checked directly on the two elements this diff already pairs - unlike a
    "missing"/"added" finding, a "text" diff already has both a real Production
    AND a real Staging element, so there is no cross-document page to estimate:
    GONE is searched around the Staging element's own page, EXTRA around the
    Production element's own page, one page either way.
    """
    def confirmed(phrases: list[str], doc: fitz.Document, el: "Element") -> bool:
        if not phrases:
            return True
        pages = sorted({p for p, _ in el.boxes}) or [el.page]
        want = sorted({p + d for p in pages for d in (-1, 0, 1) if 0 <= p + d < doc.page_count})
        blob = " " + " ".join(
            " ".join(_TOKEN_RE.findall(_normalise(doc[p].get_text("text")))) for p in want
        ) + " "
        return all(f" {' '.join(_TOKEN_RE.findall(_normalise(ph)))} " in blob for ph in phrases)

    keep: list[dict] = []
    changed = False
    for d in chapter.differences:
        a, b = d.get("exp"), d.get("act")
        if d.get("type") != "text" or a is None or b is None:
            keep.append(d)
            continue
        gone, extra = d.get("gone") or [], d.get("extra") or []
        if (gone or extra) and confirmed(gone, actual, b) and confirmed(extra, expected, a):
            changed = True
            continue
        keep.append(d)
    if not changed:
        return
    old = chapter.differences
    position = {id(d): k for k, d in enumerate(keep)}
    for row in chapter.rows:
        row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
        row["differences"] = [d for d in row.get("differences", []) if id(d) in position]
    chapter.differences = keep


def _ocr_confirms(doc: fitz.Document, pages: list[int], toks: list[str], allow_single: bool = False) -> bool:
    """Whether most of these words are what Tesseract reads off the RENDERED
    page - text baked into artwork as pixels, invisible to `get_text` but
    visible to a person looking at the page (a diagram's own caption, a label
    drawn inside a screenshot, rather than live text). Most tokens, not an
    exact phrase: OCR often reads a tightly-grouped caption in a different
    left-to-right/top-to-bottom order than the text layer would have used, and
    degrades gracefully to "no" wherever Tesseract is not installed. A single
    token is normally too coincidental to trust - it is one OCR misread away
    from a false "yes" - so it is only accepted when the caller has already
    judged this one word distinctive enough (`allow_single`, e.g. an all-caps
    port label like "DP" or "HDMI", not a common word)."""
    if len(toks) < 2 and not (allow_single and len(toks) == 1):
        return False
    from pdfval import ocr
    if not ocr.available():
        return False
    want = [w for w in (ocr.normalize_word(t) for t in toks) if w]
    if not want:
        return False
    # In order and together, on one page: most of the words merely turning up
    # somewhere across five pages confirmed any ordinary sentence ("Go to
    # Wi-Fi menu of the mobile device, and you can find the SSID...") from the
    # common words of the paragraphs around it, and a sentence Staging really
    # dropped was reported only as two stray words.
    pages_seen = sorted({p + d for p in pages for d in range(-2, 3) if 0 <= p + d < doc.page_count})
    if len(want) > 2:
        need = max(2, round(0.7 * len(want)))
        for p in pages_seen:
            read = [w for w in (ocr.normalize_word(x[4]) for x in _page_words(doc).ocr_words(p)) if w]
            blocks = [b for b in difflib.SequenceMatcher(a=want, b=read, autojunk=False).get_matching_blocks() if b.size]
            matched = sum(b.size for b in blocks)
            if matched >= need and blocks[-1].b + blocks[-1].size - blocks[0].b <= 2 * len(want) + 2:
                return True
        return False
    seen: set[str] = set()
    # A wider window than the plain-text search above: the cross-document
    # page offset is a single number for the whole chapter, but two versions
    # of the same manual rarely reflow at a perfectly constant rate - a run of
    # pages earlier in the chapter with more or less content than its
    # counterpart drifts the true offset for everything after it. Wider is
    # affordable here: unlike the plain-text search (checked for every
    # "missing"/"added" finding), OCR only runs once the cheap check already
    # failed, and confirming still takes most of several distinct words, not
    # one coincidental match.
    for p in sorted({p + d for p in pages for d in range(-2, 3) if 0 <= p + d < doc.page_count}):
        for w in _page_words(doc).ocr_words(p):
            seen.add(ocr.normalize_word(w[4]))
    hits = sum(1 for w in want if w in seen)
    if len(want) == 1:
        return hits == 1
    return hits >= max(2, round(0.7 * len(want)))


def _ocr_confirms_in_image(doc: fitz.Document, pages: list[int], toks: list[str]) -> bool:
    """Whether these words are what Tesseract reads specifically INSIDE an
    embedded raster image on one of these pages - not merely somewhere on the
    page. A diagram flattened to a single picture bakes its captions into
    those exact pixels, so restricting the OCR match to an image's own
    bounding box (instead of the whole page, like `_ocr_confirms` does) is a
    strong enough signal to trust even for a short caption beside an icon,
    where a same-word coincidence in unrelated running prose is the real risk
    that keeps the page-wide checks above from being used for this case."""
    from pdfval import ocr
    if not ocr.available():
        return False
    want = [w for w in (ocr.normalize_word(t) for t in toks) if w]
    if len(want) < 2:
        return False
    need = max(2, round(0.7 * len(want)))
    for p in sorted({p + d for p in pages for d in (-1, 0, 1) if 0 <= p + d < doc.page_count}):
        try:
            images = doc[p].get_image_info()
        except Exception:
            continue
        if not images:
            continue
        read = _page_words(doc).ocr_words(p)
        if not read:
            continue
        for info in images:
            seen = {ocr.normalize_word(w) for w in ocr.words_in(read, info["bbox"], margin=4)}
            if sum(1 for w in want if w in seen) >= need:
                return True
    return False


_SECTION_HEADING_BAND = 22.0  # pt: a heading's own printed line sits at most this far from its TocEntry y


def _heading_own_bbox(doc: fitz.Document, entry: "TocEntry") -> tuple[float, float, float, float]:
    """The printed line's own bounding box for a TOC entry - a `TocEntry`
    carries only its y-position, not a box to highlight, so the page is
    searched for the line at that position whose (squashed) text matches the
    title. Falls back to a plain band across the column at that y when the
    exact line can't be matched (a title split oddly across text runs), so
    the section is still boxed somewhere close rather than not at all.
    """
    target = re.sub(r"\s+", "", normalize_title(entry.title))
    best = None
    try:
        data = doc[entry.page].get_text("dict")
    except Exception:
        data = {"blocks": []}
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            bbox = line.get("bbox")
            if not bbox:
                continue
            y0 = bbox[1]
            if abs(y0 - entry.y) > _SECTION_HEADING_BAND:
                continue
            text = "".join(s.get("text", "") for s in line.get("spans", []))
            squashed = re.sub(r"\s+", "", normalize_title(text))
            if target and (squashed == target or squashed.startswith(target) or target.startswith(squashed)):
                return tuple(float(v) for v in bbox)
            if best is None or abs(y0 - entry.y) < abs(best[1] - entry.y):
                best = tuple(float(v) for v in bbox)
    if best is not None:
        return best
    try:
        width = doc[entry.page].rect.width
    except Exception:
        width = 550.0
    return (40.0, entry.y, min(560.0, width - 40.0), entry.y + 18.0)


_DOC_TEXT_CACHE: dict = {}
_TOC_LEADER_RE = re.compile(r"\.{4,}.*$", re.MULTILINE)


def _document_squashed_text(doc: fitz.Document) -> str:
    """The whole document's printed text, lowercased with every space removed,
    for one question: is this printed here at all? Contents-page listings (the
    lines with dot leaders) are cut out first - a heading LISTED in the other
    document's table of contents is not the same as its section being printed
    there."""
    key = (doc_key(doc), doc.page_count)
    if key not in _DOC_TEXT_CACHE:
        parts = []
        for i in range(doc.page_count):
            try:
                parts.append(_TOC_LEADER_RE.sub("", doc[i].get_text()))
            except Exception:
                continue
        _DOC_TEXT_CACHE[key] = re.sub(r"\s+", "", "\n".join(parts).lower())
    return _DOC_TEXT_CACHE[key]


def _printed_anywhere_in(doc: fitz.Document | None, text: str) -> bool:
    """`text` is printed somewhere in `doc`, ignoring all spacing - so
    "2.Connect the S Switch" and "2. Connect the S Switch" are the same words
    printed, which is the only question a content check asks."""
    target = re.sub(r"\s+", "", normalize_title(text or ""))
    if doc is None or len(target) < 8:
        return False
    return target in _document_squashed_text(doc)


def _section_missing_diffs(
    entries: list["TocEntry"], span_start: tuple, span_end: tuple | None, doc: fitz.Document, lost: bool,
    other_doc: fitz.Document | None = None,
) -> list[dict]:
    """One finding per heading this side has with no counterpart at all, on
    either side, at any level - not just the L1 chapters `ChapterSpan`
    already tracks - so a whole gone section reads as itself, not as a pile
    of separately-missing paragraphs with nothing saying they shared a
    vanished parent. Highlighted at the heading's own printed line - the
    NAME of the section is what makes this finding legible, not its body,
    which is why it is boxed there rather than spanning everything under it.
    """
    out: list[dict] = []
    for entry in entries:
        pos = (entry.page, entry.y)
        if not (pos > span_start and (span_end is None or pos < span_end)):
            continue
        # The two sides' outlines need not agree on what deserves a bookmark:
        # Staging bookmarks every numbered step of an assembly procedure where
        # Production bookmarks only the chapter around them. An unmatched
        # heading is a MISSING SECTION only when its words are genuinely not
        # printed in the other document - otherwise the content is there, laid
        # out the same, and only the navigation differs.
        if _printed_anywhere_in(other_doc, entry.title):
            continue
        bbox = _heading_own_bbox(doc, entry)
        el = Element(kind=KIND_HEADING, text=entry.title, key=normalize_title(entry.title), boxes=[(entry.page, bbox)])
        other = "Staging" if lost else "Production"
        out.append({
            "type": "section-missing" if lost else "section-added",
            "kind": KIND_HEADING,
            "exp": el if lost else None,
            "act": None if lost else el,
            "summary": (
                f"Section missing in Staging — “{entry.title}” and everything under it "
                f"is not printed anywhere in {other}."
                if lost else
                f"Section only in Staging — “{entry.title}” and everything under it "
                f"is not printed anywhere in {other}."
            ),
            "detail": "",
        })
    return out


def _soften_artwork_text(chapter: "Chapter") -> None:
    """A short numeric/spec fragment ("0-40°C 10-90% 0-3000m") printed right
    beside a picture, and apparently missing on the other side, is not
    asserted as a hard content failure when the other side's matching topic
    still has artwork of its own: the same manual very often bakes exactly
    this kind of caption into an icon's own pixels on one side and prints it
    as live text on the other, and OCR cannot reliably read tiny,
    symbol-heavy captions like these to confirm it either way (a printed '°'
    is read back as a '%' about as often as not). Shown for review, not
    failed on a guess.
    """
    lost_types = {"missing", "table-row-missing"}
    gained_types = {"added", "table-row-added"}
    for d in chapter.differences:
        t = d.get("type")
        if d.get("kind") == KIND_FIGURE or (t not in lost_types and t not in gained_types):
            continue
        lost = t in lost_types
        el = d.get("exp") if lost else d.get("act")
        if el is None or not el.text or not el.section or not _looks_like_spec_fragment(el.text):
            continue
        own_figures = [e for e in (chapter.exp_elements if lost else chapter.act_elements) if e.kind == KIND_FIGURE]
        other_figures = [e for e in (chapter.act_elements if lost else chapter.exp_elements) if e.kind == KIND_FIGURE]
        if not _near_a_figure(el, own_figures):
            continue
        if any(f.section == el.section for f in other_figures):
            d["review_only"] = True


# A figure that only "may differ" is a guess, not a finding - never reported.
# Everything else about a matched figure (replaced, resized, aligned
# differently, a callout label lost, spots that render differently) is.
_APPEARANCE_ONLY_TYPES = {"figure-review"}


def _drop_appearance_only_figures(chapter: "Chapter") -> None:
    keep = [d for d in chapter.differences if d.get("type") not in _APPEARANCE_ONLY_TYPES]
    if len(keep) == len(chapter.differences):
        return
    old = chapter.differences
    position = {id(d): k for k, d in enumerate(keep)}
    for row in chapter.rows:
        row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
        row["differences"] = [d for d in row.get("differences", []) if id(d) in position]
    chapter.differences = keep


def _keep_table_checks_only(chapter: "Chapter") -> None:
    """Image/figure and text-content validation disabled - table checks only
    (see `_ONLY_TABLE_CHECKS`). Figure work is skipped upstream so it is never
    even computed; this drops everything else (text, bold, links, list
    markers, headings, icons) that still ran as part of the shared per-block
    pass, the same way `_drop_appearance_only_figures` above drops a subset."""
    keep = [d for d in chapter.differences if d.get("kind") == KIND_TABLE]
    if len(keep) == len(chapter.differences):
        return
    old = chapter.differences
    position = {id(d): k for k, d in enumerate(keep)}
    for row in chapter.rows:
        row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
        row["differences"] = [d for d in row.get("differences", []) if id(d) in position]
    chapter.differences = keep


def _drop_all_validation(chapter: "Chapter") -> None:
    """Every check disabled (see `_VALIDATION_DISABLED`) - every diff-producing
    call upstream still runs (the report pipeline below still needs
    `chapter.rows`/elements to render the two documents side by side), but
    nothing it found is ever reported."""
    if not chapter.differences and all(not row.get("differences") for row in chapter.rows):
        return
    for row in chapter.rows:
        row["refs"] = []
        row["differences"] = []
        row["status"] = "match"
    chapter.differences = []


def _placed(el) -> tuple | None:
    """Where an element prints, for telling two findings on it apart."""
    if el is None or not getattr(el, "boxes", None):
        return None
    return (getattr(el, "key", "") or getattr(el, "text", ""), tuple(sorted({p for p, _ in el.boxes})))


def _drop_repeated_findings(chapter: "Chapter", seen: set[tuple]) -> None:
    """Keep the first of findings with the same type, summary and place,
    `seen` carrying the keys across chapters; rows' refs follow."""
    old = chapter.differences
    kept: list[dict] = []
    for d in old:
        key = (d.get("type"), (d.get("summary") or "").casefold(), _placed(d.get("exp")), _placed(d.get("act")))
        if key not in seen:
            seen.add(key)
            kept.append(d)
    if len(kept) == len(old):
        return
    chapter.differences = kept
    position = {id(d): i for i, d in enumerate(kept)}
    dropped = {id(d) for d in old if id(d) not in position}
    for row in getattr(chapter, "rows", []) or []:
        row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
        row["differences"] = [d for d in row.get("differences", []) if id(d) not in dropped]


def compare_chapters(
    expected: fitz.Document,
    actual: fitz.Document,
    expected_path: str | None = None,
    actual_path: str | None = None,
    progress_cb=None,
) -> list[Chapter]:
    """Walk every shared L1 chapter and compare the whole of it, both sides."""
    pairs = chapter_pairs(expected, actual)
    if not pairs:
        return []
    exp_entries, act_entries = resolve_entries(expected, actual)
    exp_titles = {normalize_title(e.title) for e in exp_entries if e.title.strip()}
    act_titles = {normalize_title(e.title) for e in act_entries if e.title.strip()}
    all_titles = exp_titles | act_titles
    exp_body, _ = body_style(expected)
    act_body, _ = body_style(actual)
    exp_skip = skip_spans(exp_entries, expected.page_count)
    act_skip = skip_spans(act_entries, actual.page_count)
    size_scale = (act_body / exp_body) if exp_body and act_body else 1.0
    if exp_body and act_body:
        biggest = max(exp_body, act_body)
        _FIGURE_MIN_BY_DOC[doc_key(expected)] = _FIGURE_MIN_SIDE * exp_body / biggest
        _FIGURE_MIN_BY_DOC[doc_key(actual)] = _FIGURE_MIN_SIDE * act_body / biggest

    # Topics: every heading both documents have, at every level, matched in
    # order. Content is compared only inside its own topic, straight across.
    topics: dict[str, dict] = {}
    exp_anchors: list[tuple] = []
    act_anchors: list[tuple] = []
    all_heading_matches = match_toc_entries(exp_entries, act_entries)
    for k, m in enumerate(all_heading_matches):
        if m.expected_index is None or m.actual_index is None:
            continue
        e, a = exp_entries[m.expected_index], act_entries[m.actual_index]
        topic = f"t{k}"
        topics[topic] = {"title": e.title, "prod": (e.page, e.y), "stage": (a.page, a.y)}
        exp_anchors.append((e.page, e.y, topic))
        act_anchors.append((a.page, a.y, topic))
    exp_anchors.sort()
    act_anchors.sort()
    # A heading with no counterpart at all, at any level - not just the L1
    # chapters ChapterSpan already tracks - is a whole SECTION gone, not a
    # paragraph gone: reported as its own finding below, one per chapter,
    # rather than left to read as a pile of unrelated missing paragraphs with
    # no word saying they all vanished together because their heading did.
    # Matched with `include_excluded=True` and filtered by `is_excluded_heading`
    # (TOC/Q&A/RoHS-style titles) here, NOT the entry's own `.excluded` flag -
    # that flag is overloaded by `_mark_front_matter` to also mean "sits before
    # the first heading shared with the other document", which is true of a
    # real wrapper heading like "Copyright & Disclaimer" nesting sub-headings
    # the other document bookmarks at the top level (it sits right before its
    # own first child heading, the "first shared heading" on that side) - using
    # `match_toc_entries`'s default would drop it from consideration entirely,
    # not just fail to match it, and it would never be reported missing/added.
    all_heading_matches_incl_excluded = match_toc_entries(exp_entries, act_entries, include_excluded=True)
    unmatched_exp_headings = [
        exp_entries[m.expected_index] for m in all_heading_matches_incl_excluded
        if m.expected_index is not None and m.actual_index is None
        and not is_excluded_heading(exp_entries[m.expected_index].title)
    ]
    unmatched_act_headings = [
        act_entries[m.actual_index] for m in all_heading_matches_incl_excluded
        if m.actual_index is not None and m.expected_index is None
        and not is_excluded_heading(act_entries[m.actual_index].title)
    ]

    from pdfval.validators import camelot_tables
    use_camelot = bool(expected_path and actual_path) and not _VALIDATION_DISABLED and camelot_tables.available()
    exp_camelot = camelot_tables.extract(expected, expected_path) if use_camelot else []
    act_camelot = camelot_tables.extract(actual, actual_path) if use_camelot else []

    chapters: list[Chapter] = []
    for n, span in enumerate(pairs):
        if progress_cb:
            try:
                progress_cb(n, len(pairs), span.exp_entry.title)
            except Exception:
                pass
        exp_skipped: list[str] = []
        act_skipped: list[str] = []
        exp_elements, exp_pages = collect_elements(
            expected, expected_path, span.exp_span[0], span.exp_span[1], all_titles, exp_body,
            skip=exp_skip, skipped=exp_skipped,
        )
        act_elements, act_pages = collect_elements(
            actual, actual_path, span.act_span[0], span.act_span[1], all_titles, act_body,
            skip=act_skip, skipped=act_skipped,
        )
        chapter = Chapter(
            title=span.exp_entry.title,
            exp_pages=exp_pages,
            act_pages=act_pages,
            exp_elements=exp_elements,
            act_elements=act_elements,
        )
        chapter.skipped = {"prod": exp_skipped, "stage": act_skipped}
        chapter.exp_only_tops = list(span.exp_only_tops)
        chapter.act_only_tops = list(span.act_only_tops)
        assign_sections(exp_elements, exp_anchors)
        assign_sections(act_elements, act_anchors)
        chapter.topics = topics
        exp_figs = [e for e in exp_elements if e.kind == KIND_FIGURE]
        act_figs = [e for e in act_elements if e.kind == KIND_FIGURE]
        exp_texts = [e for e in exp_elements if e.kind != KIND_FIGURE]
        act_texts = [e for e in act_elements if e.kind != KIND_FIGURE]
        # Paired topic by topic, straight across: Production's lines under a
        # heading against Staging's lines under the same heading - never against
        # another topic's, and nothing is explained away by text printed there.
        chapter.pairs = []
        words: dict[str, tuple] = {}
        act_by_topic: dict[str, list[Element]] = {}
        exp_by_topic: dict[str, list[Element]] = {}
        exp_figs_by_topic: dict[str, list[Element]] = {}
        act_figs_by_topic: dict[str, list[Element]] = {}
        exp_tables_by_topic: dict[str, list[Element]] = {}
        act_tables_by_topic: dict[str, list[Element]] = {}
        spare_words: dict[str, tuple] = {}
        for topic in dict.fromkeys(e.section for e in exp_texts + act_texts):
            exp_topic = [e for e in exp_texts if e.section == topic]
            act_topic = [e for e in act_texts if e.section == topic]
            exp_topic_figs = [e for e in exp_figs if e.section == topic]
            act_topic_figs = [e for e in act_figs if e.section == topic]
            topic_pairs = pair_elements(exp_topic, act_topic)
            chapter.pairs += reconcile_layout(topic_pairs, exp_topic, act_topic, exp_topic_figs, act_topic_figs)
            words[topic] = _topic_words(exp_topic, act_topic, expected, actual, exp_topic_figs, act_topic_figs)
            # Content already accounted for by a normal, both-sided pair is
            # spoken for - settling a missing/added finding by finding those
            # same words is what let Staging's OWN genuine duplicate of an
            # entire numbered procedure excuse itself by pointing at the very
            # Production text its legitimate, correctly-matched twin was
            # already paired with (see `reconcile_layout`'s own fix for the
            # same class of bug). Built from only the SPARE (still one-sided)
            # elements of this topic, so a real duplicate has nothing left,
            # once-matched, to settle against.
            matched_exp_ids = {id(e) for eg, ag, _ in topic_pairs if eg and ag for e in eg}
            matched_act_ids = {id(e) for eg, ag, _ in topic_pairs if eg and ag for e in ag}
            exp_spare_topic = [e for e in exp_topic if id(e) not in matched_exp_ids]
            act_spare_topic = [e for e in act_topic if id(e) not in matched_act_ids]
            spare_words[topic] = _topic_words(
                exp_spare_topic, act_spare_topic, expected, actual, exp_topic_figs, act_topic_figs)
            act_by_topic[topic] = act_topic
            exp_by_topic[topic] = exp_topic
            exp_figs_by_topic[topic] = exp_topic_figs
            act_figs_by_topic[topic] = act_topic_figs
            exp_tables_by_topic[topic] = [e for e in exp_topic if e.kind == KIND_TABLE]
            act_tables_by_topic[topic] = [e for e in act_topic if e.kind == KIND_TABLE]
        repeated: dict[tuple, int] = {}
        # Where one-sided content belongs on the other side: just after the
        # last element there the pairing put before it (in the same topic),
        # not at the topic's heading.
        last_exp: Element | None = None
        last_act: Element | None = None
        gap_after: dict[int, tuple] = {}
        for pair_index, (exp_group, act_group, _n, _c) in enumerate(chapter.pairs):
            topic_here = (exp_group or act_group)[0].section
            if exp_group and not act_group and last_act is not None and last_act.section == topic_here:
                page_index, bbox = last_act.boxes[-1]
                gap_after[pair_index] = ("act", (page_index, bbox[3] + 2))
            if act_group and not exp_group and last_exp is not None and last_exp.section == topic_here:
                page_index, bbox = last_exp.boxes[-1]
                gap_after[pair_index] = ("exp", (page_index, bbox[3] + 2))
            if exp_group and exp_group[-1].boxes:
                last_exp = exp_group[-1]
            if act_group and act_group[-1].boxes:
                last_act = act_group[-1]
        for pair_index, (exp_group, act_group, note, container) in enumerate(chapter.pairs):
            group_topic = (exp_group or act_group)[0].section
            exp_words, act_words, exp_units, act_units, exp_keys, act_keys = words[group_topic]
            spare_exp_words, spare_act_words, spare_exp_units, spare_act_units, _, _ = spare_words[group_topic]
            diffs = (
                [] if note == "layout"
                else settle_content(
                    compare_group(exp_group, act_group, size_scale, moved=note == "moved"),
                    exp_words, act_words, merge_elements(exp_group), merge_elements(act_group),
                    exp_units, act_units, exp_keys, act_keys,
                    exp_figs_by_topic.get(group_topic, []), act_figs_by_topic.get(group_topic, []),
                    exp_tables_by_topic.get(group_topic, []), act_tables_by_topic.get(group_topic, []),
                    spare_exp_words, spare_act_words, spare_exp_units, spare_act_units,
                )
            )
            diffs = settle_tables(diffs, merge_elements(exp_group), merge_elements(act_group), exp_units, act_units)
            exp_table, act_table = merge_elements(exp_group), merge_elements(act_group)
            if not act_group and exp_table is not None and exp_table.kind == KIND_TABLE:
                converted = table_as_text(exp_table, act_by_topic.get(exp_group[0].section, []))
                if converted:
                    diffs = [converted]
            if (not act_group and len(exp_group) == 1 and exp_group[0].kind == KIND_HEADING
                    and _printed_as_line(actual, act_pages, exp_group[0].key)):
                # Staging prints the heading - often restyled as a bold label the
                # furniture filter or heading detector skipped ("ST6504" set 13.5pt
                # bold at the top of the page). Its styling is Staging's design.
                diffs = []
            if (note != "layout" and exp_table is not None and act_table is not None
                    and exp_table.kind == KIND_TABLE and act_table.kind == KIND_TABLE
                    and exp_table.cells and act_table.cells):
                # Row by row replaces the whole-grid shape and cell checks.
                diffs = [d for d in diffs if d.get("type") not in _TABLE_GRID_TYPES]
                diffs += _drop_labels_drawn_as_icons(table_row_changes(exp_table, act_table, exp_units, act_units),
                                                     exp_table, act_table, expected, actual)
                diffs += table_fill_changes(exp_table, act_table, expected, actual)
                diffs = settle_tables(diffs, exp_table, act_table, exp_units, act_units)
            diffs = confirm_bold(diffs, merge_elements(exp_group), merge_elements(act_group), expected, actual)
            topic = (exp_group or act_group)[0].section
            diffs = [d for d in diffs if not _wrap_only(d, exp_words, act_words)]
            diffs = [
                d for d in diffs
                if not (d.get("type") == "bold-added"
                        and _same_weight_in_topic(d.get("phrases") or [], exp_by_topic.get(topic, []), expected))
                and not (d.get("type") == "bold-missing"
                         and _same_weight_in_topic(d.get("phrases") or [], act_by_topic.get(topic, []), actual))
            ]
            if exp_group and act_group:
                diffs += marker_changes(exp_group, act_group, expected, actual)
                diffs += shading_changes(exp_group, act_group, expected, actual,
                                         exp_by_topic.get(topic, []), act_by_topic.get(topic, []))
                # Staging dropping a cross-reference's own "on page N" is not
                # reported: the wording still says where to look, just not
                # the page number, which is valid.
                diffs += list_label_layout_changes(exp_group, act_group, expected, actual)
                # One stray space, one finding: the punctuation check already
                # names it ("after “location”: extra space in Staging").
                if not any(d.get("type") in ("punctuation", "symbol")
                           and "space" in " ".join(d.get("mark_notes") or [d.get("summary", "")])
                           for d in diffs):
                    diffs += stray_space_changes(exp_group, act_group)
                diffs += line_spacing_changes(exp_group, act_group, expected, actual,
                                              exp_by_topic.get(topic, []), act_by_topic.get(topic, []))
                if not any(e.kind == KIND_TABLE for e in exp_group + act_group):
                    # Table cells report their own space changes, cell by cell,
                    # and their own icon-in-a-cell comparisons (an "OSD icon"
                    # column, say) rather than this word-anchored check: a
                    # table's grid is not a line of running text with an icon
                    # beside one word in it - and a table read as one grouped
                    # slot on one side but as plain paragraphs (the same table
                    # missed by table detection on the other side) is not even
                    # the same shape to anchor a word against. One shared
                    # table on both sides of the pair is enough to skip this;
                    # a mismatched table/text pairing is exactly the case that
                    # must not be word-anchored against each other.
                    diffs += space_changes(exp_group, act_group)
                    diffs += icon_changes(exp_group, act_group, expected, actual,
                                          exp_by_topic.get(topic, []), act_by_topic.get(topic, []))
            refs: list[int] = []
            for diff in diffs:
                # A difference that already knows its exact place (a table row,
                # a list marker) keeps it; the rest point at their group.
                diff.setdefault("exp", merge_elements(exp_group))
                diff.setdefault("act", merge_elements(act_group))
                diff.setdefault("section", (exp_group or act_group)[0].section)
                if pair_index in gap_after:
                    side, point = gap_after[pair_index]
                    diff.setdefault(f"{side}_anchor", point)
                _quote_as_printed(diff, expected, actual)
                # Staging sets every sub-heading at 13.5pt where Production uses
                # 16pt: one decision, printed on twenty headings. Reported once,
                # with every heading still boxed under the same number, it reads
                # as the single change it is instead of burying the chapter's
                # real differences under twenty identical lines.
                key = (diff["type"], diff.get("kind"), diff["summary"]) if diff["type"] in _REPEATABLE else None
                if key is not None and key in repeated:
                    first = chapter.differences[repeated[key]]
                    first.setdefault("repeats", []).append((diff["exp"], diff["act"]))
                    refs.append(repeated[key])
                    continue
                if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                    if key is not None:
                        repeated[key] = len(chapter.differences)
                    refs.append(len(chapter.differences))
                    chapter.differences.append(diff)
            for diff in chapter.differences:
                count = len(diff.get("repeats") or [])
                if count and not diff.get("_counted") == count:
                    base = diff.get("_base_summary") or diff["summary"]
                    diff["_base_summary"] = base
                    diff["summary"] = (
                        f"{base} Same change on {count + 1} "
                        f"{KIND_LABEL.get(diff.get('kind'), 'element').lower()}s in this chapter."
                    )
                    diff["_counted"] = count
            chapter.rows.append({
                "exp": exp_group,
                "act": act_group,
                "differences": diffs,
                "container": container,
                "refs": refs,
                "status": (
                    "layout" if note == "layout" else
                    "missing" if not act_group else
                    "added" if not exp_group else
                    "changed" if diffs else "match"
                ),
            })
        # Content nobody reported: every topic's whole text, Production against
        # Staging, after every other check has had its say.
        for topic in dict.fromkeys(e.section for e in exp_texts + act_texts):
            for diff in coverage_changes(chapter, topic, exp_by_topic.get(topic, []),
                                         act_by_topic.get(topic, []), expected, actual):
                if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                    chapter.differences.append(diff)
        # Figures, matched on their own terms (see `figure_changes`) - skipped
        # entirely (not just filtered after the fact) so the expensive
        # whole-document picture crawl never runs while `_ONLY_TABLE_CHECKS`.
        for fig_row in ([] if _ONLY_TABLE_CHECKS else figure_changes(
                                      exp_figs, act_figs, exp_texts, act_texts, expected, actual,
                                      exp_entries, act_entries, exp_pages, act_pages,
                                      exp_bands=_chapter_bands(expected, exp_pages, span.exp_span),
                                      act_bands=_chapter_bands(actual, act_pages, span.act_span),
                                      scale=size_scale,
                                      exp_section_at=_section_at(exp_anchors),
                                      act_section_at=_section_at(act_anchors))):
            refs = []
            fig_row["differences"] = reportable(fig_row["differences"])
            for diff in fig_row["differences"]:
                diff.setdefault("exp", fig_row["exp"][0] if fig_row["exp"] else None)
                diff.setdefault("act", fig_row["act"][0] if fig_row["act"] else None)
                if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                    refs.append(len(chapter.differences))
                    chapter.differences.append(diff)
            chapter.rows.append({
                "exp": fig_row["exp"], "act": fig_row["act"], "differences": fig_row["differences"],
                "container": None, "refs": refs,
                "status": ("missing" if not fig_row["act"] else "added" if not fig_row["exp"]
                           else "changed" if fig_row["differences"] else "match"),
            })
        scale = len(exp_elements) / max(1, len(act_elements))
        chapter.rows.sort(key=lambda r: r["exp"][0].order if r["exp"] else r["act"][0].order * scale)
        for diff in numbering_changes(exp_elements, act_elements) + marker_size_changes(
            exp_elements, act_elements, expected, actual
        ) + list_indent_changes(
            exp_elements, act_elements, expected, actual, scale=size_scale
        ) + bullet_glyph_changes(exp_elements, act_elements):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        partners: dict[int, list[Element]] = {}
        for exp_group, act_group, _note, _container in chapter.pairs:
            if exp_group and act_group:
                for e in exp_group:
                    partners[id(e)] = list(act_group)
                for e in act_group:
                    partners[id(e)] = list(exp_group)
        for diff in table_header_repeats(actual, actual_path, act_pages) + list_structure_changes(
                exp_elements, act_elements, expected, actual, partners=partners):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        exp_links = collect_links(expected, exp_pages, span.exp_span, exp_entries)
        act_links = collect_links(actual, act_pages, span.act_span, act_entries)
        assign_sections(exp_links, exp_anchors)
        assign_sections(act_links, act_anchors)
        for diff in link_changes(exp_links, act_links, exp_elements, act_elements, expected, actual):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        for diff in page_ref_consistency_changes(act_links):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        for diff in broken_image_changes(actual, act_pages, _section_at(act_anchors)):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        for diff in page_ref_in_staging_changes(act_elements, actual):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        for diff in glyph_render_changes(act_elements, actual):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        _drop_printed_nearby(chapter, expected, actual)
        _drop_confirmed_content_changes(chapter, expected, actual)
        _soften_artwork_text(chapter)
        _drop_appearance_only_figures(chapter)
        for diff in (
            _section_missing_diffs(unmatched_exp_headings, span.exp_span[0], span.exp_span[1], expected,
                                   lost=True, other_doc=actual)
            + _section_missing_diffs(unmatched_act_headings, span.act_span[0], span.act_span[1], actual,
                                     lost=False, other_doc=expected)
        ):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        # Every ruled table read a second time, with Camelot, and compared row
        # by row - only what no finding above already names is added.
        exp_span_lines = _span_lines(expected, exp_pages, span.exp_span, exp_body, all_titles, exp_skip)
        act_span_lines = _span_lines(actual, act_pages, span.act_span, act_body, all_titles, act_skip)
        if use_camelot:
            exp_at_, act_at_ = _section_at(exp_anchors), _section_at(act_anchors)
            printed_bags: dict = {}
            for side, lines, at_ in (("prod", exp_span_lines, exp_at_), ("stage", act_span_lines, act_at_)):
                for ln in lines:
                    printed_bags.setdefault((side, at_(ln["page"], ln["bbox"][1])), Counter()).update(
                        camelot_tables._norm(ln["text"]).split())

            def printed_of(side: str, topic: str) -> Counter:
                return printed_bags.get((side, topic), Counter())

            def covered_of(topic: str, _chapter=chapter) -> set[str]:
                words: set[str] = set()
                for d in _chapter.differences:
                    if d.get("section") != topic:
                        continue
                    said = " ".join([d.get("summary", ""), str(d.get("detail") or "")]
                                    + [str(x) for x in (d.get("gone") or []) + (d.get("extra") or [])])
                    for key in ("exp", "act"):
                        el = d.get(key)
                        if el is not None and d.get("type") in ("missing", "added") or (
                                el is not None and str(d.get("type", "")).startswith("table-")):
                            said += " " + (getattr(el, "text", "") or "")
                    words |= set(_TOKEN_RE.findall(_normalise(said)))
                return words
            for diff in camelot_tables.table_changes(
                exp_camelot, act_camelot, span.exp_span, span.act_span,
                exp_at_, act_at_, Element, KIND_TABLE, covered_of, printed_of,
            ) + camelot_tables.merge_changes(
                act_camelot, span.act_span, act_at_, expected, exp_pages, Element, KIND_TABLE,
            ):
                if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                    chapter.differences.append(diff)
        # Every word of every topic, counted straight off the page on both
        # sides - the net under every filter above (see `word_audit_changes`).
        if not _ONLY_TABLE_CHECKS and not _VALIDATION_DISABLED:
            # Whole blocks first: a repeated or dropped block reported here
            # is `covered` when the per-word audit below runs, so the same
            # content is not also listed a second time as loose words.
            for diff in repeated_block_changes(
                chapter, exp_span_lines, act_span_lines,
                _section_at(exp_anchors), _section_at(act_anchors),
                exp_figs, act_figs, expected, actual,
            ):
                if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                    chapter.differences.append(diff)
            for diff in word_audit_changes(
                chapter,
                exp_span_lines, act_span_lines,
                _section_at(exp_anchors), _section_at(act_anchors),
                exp_figs, act_figs, expected, actual,
            ) + table_icon_layout_changes(
                exp_span_lines, act_span_lines,
                _section_at(exp_anchors), _section_at(act_anchors),
            ):
                if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                    chapter.differences.append(diff)
        # Last: the same content set differently (a table hyphenated into
        # narrow columns on one side) is not a difference - see
        # `drop_reflow_noise`.
        if not _ONLY_TABLE_CHECKS and not _VALIDATION_DISABLED:
            drop_reflow_noise(chapter, exp_span_lines, act_span_lines,
                              _section_at(exp_anchors), _section_at(act_anchors))
        if _ONLY_TABLE_CHECKS:
            _keep_table_checks_only(chapter)
        if _VALIDATION_DISABLED:
            _drop_all_validation(chapter)
        # Font size and text colour are never compared (the user's rule). Nor
        # is a table's row structure on its own: Production nesting several
        # items in one cell where Staging rules each into a row of its own is a
        # different container for the same text. A merged cell Staging splits
        # leaving an EMPTY cell is still reported (`camelot_tables.merge_changes`).
        chapter.differences = [d for d in chapter.differences if d.get("type") not in _NEVER_REPORTED
                               and not (d.get("type") == "table-merge" and not d.get("camelot"))]
        # The same finding on the same place, raised by two checks (or by one
        # check over two groups holding the same element), is one finding.
        _drop_repeated_findings(chapter, set())
        # Severity decides everything downstream: the order the report lists a
        # difference in, whether its box is red or orange, whether it fails.
        for diff in chapter.differences:
            diff["severity"] = severity_of(diff)
            diff["minor"] = diff["severity"] > FAILING_SEVERITY
            if "section" not in diff:
                placed = diff.get("exp") or diff.get("act")
                diff["section"] = placed.section if placed is not None else ""
        chapters.append(chapter)

    # A last, whole-book pass over plain text alone: catches content lost
    # between chapters (moved to an unmatched heading, or landed under the
    # wrong topic where the two tables of contents did not line up) that a
    # check scoped to one topic at a time could never see. See `plain_text_scan`.
    for chapter, diff in ([] if _VALIDATION_DISABLED or _ONLY_TABLE_CHECKS
                          else plain_text_scan(chapters, expected, actual)):
        if len(chapter.differences) >= MAX_ISSUES_PER_CHAPTER:
            continue
        diff["severity"] = severity_of(diff)
        diff["minor"] = diff["severity"] > FAILING_SEVERITY
        chapter.differences.append(diff)

    # A printed "Table of Contents" page is deliberately kept out of every
    # chapter's own span above (see `_listing_page`) - its own entry list is
    # still real content a reader sees, so it is checked once here, for the
    # whole document, and its findings anchored on the first chapter for lack
    # of any topic of their own.
    if chapters and not _VALIDATION_DISABLED and not _ONLY_TABLE_CHECKS:
        for diff in printed_toc_changes(expected, actual):
            if len(chapters[0].differences) < MAX_ISSUES_PER_CHAPTER:
                diff["severity"] = severity_of(diff)
                diff["minor"] = diff["severity"] > FAILING_SEVERITY
                diff.setdefault("section", "")
                chapters[0].differences.append(diff)
    # Every hyperlink finding says where each side's link goes.
    def _where(label: str) -> str:
        if not label:
            return "nowhere (no target)"
        if label.startswith("section: "):
            return f"the “{label[len('section: '):]}” section"
        if label.startswith("page "):
            return label
        return label
    for chapter in chapters:
        for d in chapter.differences:
            if not str(d.get("type", "")).startswith("link-") or d.get("_where_added"):
                continue
            parts = []
            for side, key in (("Production", "exp"), ("Staging", "act")):
                el = d.get(key)
                if el is not None and getattr(el, "kind", "") == KIND_LINK:
                    parts.append(f"{side} link goes to {_where(el.label)}")
            if parts:
                d["summary"] = d["summary"].rstrip(".") + ". " + "; ".join(parts) + "."
                d["_where_added"] = True
    # Every punctuation / symbol finding names the exact marks: which quote,
    # period or space Staging lacks or adds, and between which words.
    for chapter in chapters:
        for d in chapter.differences:
            if d.get("type") not in ("punctuation", "symbol") or d.get("mark_notes"):
                continue
            a_el, b_el = d.get("exp"), d.get("act")
            if a_el is None or b_el is None or not a_el.text or not b_el.text:
                continue
            notes = mark_changes(a_el.text, b_el.text, a_el.wraps, b_el.wraps)
            if notes:
                d["mark_notes"] = notes
                head = "Symbol changed" if d["type"] == "symbol" else "Punctuation or spacing changed"
                d["summary"] = f"{head} — " + "; ".join(notes[:6]) + (
                    f"; and {len(notes) - 6} more" if len(notes) > 6 else "") + "."
    # Text read off a scanned page is an OCR reading, not the text layer:
    # say so on every finding it produced, after every summary rewrite above.
    for chapter in chapters:
        for d in chapter.differences:
            sides = d.get("ocr_sides")
            if sides and d.get("summary") and "read by OCR" not in d["summary"]:
                d["summary"] += (f" (read by OCR from a scanned page in {' and '.join(sides)}"
                                 f" — recognition errors possible)")
    # Last: a finding raised twice anywhere (two passes, or two chapters
    # sharing a page) is one finding.
    seen_all: set[tuple] = set()
    for ch in chapters:
        _drop_repeated_findings(ch, seen_all)
    return chapters



# Which differences fail a run, and which are shown but do not.
# --- severity ---------------------------------------------------------------
#
# One fixed order for every difference, so the report reads the same way in
# every chapter and the run fails for the same reasons every time:
#
#   1 Content     - what the document says: wording, anything missing or added,
#                   list numbering, callout type, a table cell's text, a figure
#                   replaced
#   2 Hyperlinks  - a link gone, added, pointing somewhere else, or broken
#   3 Table layout- rows/columns, merged or split cells, the header row's
#                   background colour
#   4 Images      - the same picture shown at another size, or not confirmed
#   5 Formatting  - the same words set differently: weight, size, colour, case,
#                   punctuation
#   6 Layout      - read-out artefacts: moved, word order, laid out differently
#
# 1-3 fail the run. 4-6 are listed for review.
SEVERITY_OF = {
    # 1 - what the chapter says, and how its lists are marked
    "text": 1, "missing": 1, "added": 1, "section-missing": 1, "section-added": 1, "note-label": 1, "table-cell": 1,
    "table-cell-sequence": 1, "text-encoding": 1, "glyph-render": 1, "symbol": 1, "page-ref-in-staging": 1, "paragraph-gap": 1,
    "numbering": 1, "list-marker-missing": 1, "list-marker-added": 1,
    # 2 - hyperlinks
    "link-missing": 2, "link-added": 2, "link-target": 2, "link-broken": 2,
    "link-page-ref-inconsistent": 2, "link-page-ref-dropped": 2,
    # 4 - images
    "figure-missing": 4, "figure-added": 4, "figure-broken": 1, "figure-elsewhere": 4, "figure-different": 4,
    "figure-content": 4, "figure-label-missing": 4, "figure-size": 4, "figure-visual": 4,
    # 5 - emphasis
    "bold-missing": 5, "bold-added": 5, "shading": 5, "marker-size": 5,
    "underline-missing": 5, "underline-added": 5, "line-spacing": 5,
    "italic-missing": 5, "italic-added": 5,
    "icon-missing": 4, "icon-added": 4, "icon-colour": 4, "icon-changed": 4,
    # 6 - how it is laid out, not what it says - shown, never fails the run
    "list-indent": 6, "figure-alignment": 6, "text-space": 1, "moved": 6, "figure-wrong-section": 6,
    "punctuation": 1,
    "list-label-layout": 6, "list-marker-glyph": 6,
    # (table-header-fill is not reported: Staging's header colour is its design.
    # A background gone altogether is: table-fill-missing.)
    "table-fill-missing": 3,
    "table-shape": 6, "table-merge": 6, "table-header-repeat": 6, "table-cell-layout": 6,
    "table-row-missing": 6, "table-row-added": 6, "table-row-count": 6, "table-columns": 6, "table-as-text": 6,
}
_NEVER_REPORTED = {"size", "colour", "color", "marker-size", "font-size", "text-colour",
                   # Production underlining a menu path or a cross-reference
                   # that Staging prints plain is expected (user rule).
                   "underline-missing",
                   # Staging dropping "on page N" is expected (user rule); one it
                   # keeps is reported by `page_ref_in_staging_changes` instead.
                   "link-page-ref-dropped", "link-page-ref-inconsistent"}
SEVERITY_LABEL = {
    1: "Content & lists", 2: "Hyperlinks", 4: "Images", 5: "Bold", 6: "Layout",
}
# Only what is listed above is reported at all. Font size, text colour,
# capitalisation, punctuation, word order, an element that merely moved, a
# figure that "may differ", a table header's background colour and a table
# read with a different grid are deliberately NOT - they are how the two
# documents were typeset, not defects. Every difference that IS reported down
# to FAILING_SEVERITY is a bug to fix and fails the run; how a table is ruled,
# a list's indent, a run of extra/missing space and a figure's alignment
# (level 6) are still shown, boxed in orange, but do not - and level 1
# (what the document actually says or is missing outright) is the one real
# defect a reader must not miss, so it is drawn in a darker, more urgent red
# than every other failing level.
FAILING_SEVERITY = 5


def severity_of(diff: dict) -> int:
    # A finding that marks itself critical is level 1 whatever its type's usual
    # tier - it is content-scope, not the cosmetic change the type implies.
    if diff.get("critical"):
        return 1
    base = SEVERITY_OF.get(diff.get("type"), 1)
    # A finding `_soften_artwork_text` could not confirm either way (a spec
    # caption plausibly baked into the other side's artwork) is shown, not
    # failed on a guess.
    if diff.get("review_only") and base <= FAILING_SEVERITY:
        return FAILING_SEVERITY + 1
    return base


def reportable(diffs: list[dict]) -> list[dict]:
    return [d for d in diffs if d.get("type") in SEVERITY_OF]


# --- categories ---------------------------------------------------------------
#
# The six things a reviewer checks Staging against Production for, in the order
# they are listed. Every report and report.json file a difference under exactly
# one of these, so a count means the same thing wherever it is read. `help` is
# said once per category, not repeated on every issue.
CATEGORIES = [
    {"key": "content", "label": "Content",
     "help": "Text missing from Staging, extra in Staging, or worded differently. Lines that wrap "
             "differently and paragraphs that continue onto the next page are joined before comparing, "
             "so only real wording differences are listed."},
    {"key": "images", "label": "Images",
     "help": "Pictures missing, extra, replaced, moved to another section, oversized, aligned "
             "differently (left instead of centre), or with a label missing."},
    {"key": "links", "label": "Hyperlinks",
     "help": "Links missing or extra in Staging, going somewhere else, or not working."},
    {"key": "lists", "label": "Lists",
     "help": "List items (ul / li) that lost or gained their bullet or number, are numbered "
             "differently, or are indented or aligned differently within their list."},
    {"key": "bold", "label": "Bold",
     "help": "Text that prints bold in Production and visibly lighter in Staging. Judged by how dark "
             "the text prints, not by the font's weight name - text that looks as dark is fine."},
    {"key": "tables", "label": "Tables",
     "help": "Table merge issues only: cells merged or split differently, or a different number of "
             "columns. What a cell or row says - changed words, a missing row, a lost space - is Content."},
    {"key": "formatting", "label": "Formatting",
     "help": "The same text set visibly differently - printed on a shaded box in one document only."},
    # Only ever filled when the optional AI review is switched on. It carries
    # the page-by-page sweep's own gaps, boxed where they were measured, with
    # the model's reading of what they mean - findings no named rule produced.
    # The completeness sweep: every mirrored page measured and compared as a
    # whole, so a difference no rule names is still found. Filled whether or
    # not the optional AI review runs; when it does, its notes join it here.
    {"key": "sweep", "label": "Page sweep",
     "help": "Differences found by measuring both pages and comparing them, rather than by a named "
             "rule - words, links or tables on one page and not its counterpart. Shown for review."},
]
CATEGORY_LABEL = {c["key"]: c["label"] for c in CATEGORIES}

_TYPE_CATEGORY = {
    "text": "content", "note-label": "content", "section-missing": "content", "section-added": "content",
    "link-missing": "links", "link-added": "links", "link-target": "links", "link-broken": "links",
    "link-page-ref-inconsistent": "links", "link-page-ref-dropped": "links",
    "numbering": "lists", "list-marker-missing": "lists", "list-marker-added": "lists", "list-indent": "lists",
    "marker-size": "lists", "list-label-layout": "lists", "list-marker-glyph": "lists",
    "bold-missing": "bold", "bold-added": "bold", "shading": "formatting",
    "underline-missing": "formatting", "underline-added": "formatting", "line-spacing": "formatting",
    "italic-missing": "formatting", "italic-added": "formatting",
    "icon-missing": "images", "icon-added": "images", "icon-colour": "images", "icon-changed": "images",
    # Tables holds only how cells are merged: what a cell or row SAYS - a
    # changed word, a missing row, a lost space - is content like any other.
    "table-merge": "tables", "table-columns": "tables", "table-fill-missing": "tables",
    "table-cell-layout": "tables",
    "table-shape": "content", "table-header-repeat": "content", "table-cell": "content",
    "table-cell-sequence": "content",
    "table-row-missing": "content", "table-row-added": "content", "table-row-count": "content", "table-as-text": "content",
    "text-space": "content", "text-encoding": "content", "punctuation": "content", "glyph-render": "content",
    "symbol": "content",
    # A figure moved to the wrong section is a sequence/order problem, like a
    # paragraph out of reading order - it reads as content, not as the
    # picture's own look, so it is red like the rest of "moved", not blue.
    "figure-wrong-section": "content",
}


def category_of(diff: dict) -> str:
    """Which of the six categories a difference belongs to."""
    kind = diff.get("type") or ""
    if kind in _TYPE_CATEGORY:
        return _TYPE_CATEGORY[kind]
    if kind.startswith("figure-"):
        return "images"
    if kind in ("missing", "added"):
        # Content only one side has is filed by what it IS: a whole table
        # missing is a table issue, a whole picture an image issue.
        element = diff.get("kind")
        return "images" if element == KIND_FIGURE else "content"
    return "content"


# Formatting changes that are one styling decision when they repeat.
_REPEATABLE = {"size", "colour", "emphasis", "capitalisation", "word-order",
               "table-header-fill", "shading", "list-marker-added", "list-marker-missing",
               "icon-colour", "icon-changed", "icon-missing", "icon-added", "list-label-layout",
               "list-marker-glyph"}

_MINOR_TYPES = {
    "emphasis", "size", "colour", "figure-size", "figure-review", "moved", "table-extract",
    "capitalisation", "word-order", "marker-size",
}


def validate_chapters(
    expected: fitz.Document,
    actual: fitz.Document,
    expected_path: str | None = None,
    actual_path: str | None = None,
    chapters: list[Chapter] | None = None,
) -> CheckResult:
    """The chapter-by-chapter comparison, as a check result for the report.

    The evidence itself - every element of every chapter, side by side, with
    both documents' pages rendered and boxed - is in chapters.html; this is the
    summary that belongs beside the other checks.
    """
    result = CheckResult(name="Chapter Validation")
    if chapters is None:
        chapters = compare_chapters(expected, actual, expected_path, actual_path)
    result.summary_title = "Chapters compared (L1 heading to the next L1 heading)"
    for chapter in chapters:
        for diff in chapter.differences:
            a, b = diff.get("exp"), diff.get("act")
            details = {
                "heading": chapter.title,
                "element": KIND_LABEL.get(diff.get("kind"), "Content"),
                "difference": diff["summary"],
            }
            if a is not None:
                details["expected_page"] = a.page + 1
                if a.bbox and any(a.bbox):
                    details["exp_bbox"] = list(a.bbox)
            if b is not None:
                details["actual_page"] = b.page + 1
                if b.bbox and any(b.bbox):
                    details["act_bbox"] = list(b.bbox)
            if diff.get("detail"):
                details["detail"] = diff["detail"]
            if a is not None and a.text:
                details["expected"] = [a.text[:600]]
            if b is not None and b.text:
                details["actual"] = [b.text[:600]]
            level = severity_of(diff)
            category = CATEGORY_LABEL[category_of(diff)]
            details["category"] = category
            if level > FAILING_SEVERITY:
                details["confidence"] = "review"
            result.issues.append(Issue(
                severity="error" if level <= FAILING_SEVERITY else "warning",
                page=(a.page if a is not None else (b.page if b is not None else None)),
                message=f"{category}: {diff['summary'].split(' — ')[0]}",
                details=details,
            ))
        result.summary_rows.append({
            "chapter": chapter.title,
            "prod_pages": _page_range(chapter.exp_pages),
            "stage_pages": _page_range(chapter.act_pages),
            "elements": f"{len(chapter.exp_elements)} / {len(chapter.act_elements)}",
            "differences": len(chapter.differences),
            "status": "Differences" if chapter.differences else "Matches",
        })
    return result


def _page_range(pages: list[int]) -> str:
    if not pages:
        return "—"
    lo, hi = pages[0] + 1, pages[-1] + 1
    return f"p.{lo}" if lo == hi else f"p.{lo}–{hi}"
