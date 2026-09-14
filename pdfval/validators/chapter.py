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

import difflib
import itertools
import re
from collections import Counter
from dataclasses import dataclass, field

import fitz

from pdfval import i18n, imagefp, ocr
from pdfval.extractor import get_figures, get_tables, get_vector_figures
from pdfval.models import CheckResult, Issue
from pdfval.validators.alignment import classify_marker
from pdfval.validators.headings import body_style, resolve_entries
from pdfval.validators.links import USABLE_SCHEMES, _named_destination_resolves
from pdfval.validators.table import _document_margins, _headers_match, _stitch_continued_tables
from pdfval.validators.toc import (
    TocEntry,
    heading_at,
    _looks_like_qa_index_page,
    is_excluded_heading,
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
    raw_cells: tuple = ()          # the same cells as printed (case, punctuation), row-aligned with `cells`
    width: float = 0.0             # figure size on the page, points
    height: float = 0.0
    fp: object | None = None       # figure appearance (imagefp.Fingerprint), trimmed to its ink
    raw_bbox: tuple | None = None  # a figure's box as detected, before trimming
    raw_fp: object | None = None   # ... and its appearance over that box
    header_fill: int | None = None # table header row background, 0xRRGGBB; None = none
    spans: tuple = ()              # merged cells per table row (cells a merge swallowed)
    bold_words: tuple = ()         # the words of this element that are set bold, in order
    underline_words: tuple = ()    # the words of this element drawn with an underline rule, in order
    grid: tuple = ()               # table cell columns per row: ((x0, x1) | None, ...)
    row_boxes: tuple = ()          # table rows: ((page, bbox), ...), one per row of `cells`
    section: str = ""              # the nearest heading above it that BOTH documents have (normalised)
    icon_column: bool = False      # a table printing its icons in a column of their own
    drawn: bool = False            # a figure drawn as vector shapes rather than an embedded image
    headers: tuple = ()            # a table's header cells, as printed

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
_LIST_MARKER_TOKEN_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\d{1,2}|[a-z]|[ivx]{2,4})[.)](?=\s|$)|[\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219]",
    re.IGNORECASE,
)


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


_DOC_KEYS = __import__("itertools").count(1)


def _doc_key(doc) -> int:
    """A stable identity for a document, for per-document caches. id(doc) is not
    one: Python reuses an id once the document is freed, and the next comparison
    in the same process (the web server, the test suite) then read the previous
    document's cached lines, furniture and figures."""
    key = getattr(doc, "_pdfval_key", None)
    if key is None:
        key = next(_DOC_KEYS)
        try:
            setattr(doc, "_pdfval_key", key)
        except Exception:
            return id(doc)
    return key


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
    key = (_doc_key(doc), doc.page_count)
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
                texts.setdefault(_furniture_key(text), set()).add(page_index)
                if _at_edge((0.0, y0, 0.0, y1), rect):
                    bands.setdefault(_band_of(y0, rect), set()).add(page_index)

    pages = max(1, doc.page_count)
    out = (
        {t for t, seen in texts.items() if len(seen) >= _FURNITURE_MIN_PAGES},
        {b for b, seen in bands.items() if len(seen) / pages >= _FURNITURE_BAND_SHARE},
    )
    _FURNITURE_CACHE[key] = out
    return out


def _band_of(y0: float, rect) -> int:
    return int(y0 / max(1.0, rect.height) * _FURNITURE_BANDS)


def reset_furniture_cache() -> None:
    _FURNITURE_CACHE.clear()
    _NAMES_CACHE.clear()
    _FIG_INDEX.clear()
    _MARGIN_CACHE.clear()
    _LABEL_WORDS.clear()
    _CHAR_CACHE.clear()
    _LINE_GEOM_CACHE.clear()
    _FIGURE_MIN_BY_DOC.clear()
    _FILL_CACHE.clear()
    _ICON_CACHE.clear()
    _FIGURE_WORDS_CACHE.clear()


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
    texts, bands = _furniture(doc)
    key = _furniture_key(text)
    # Repeated text is furniture only in the outermost strip - or anywhere in
    # the edge band when it is shaped like a page number. A procedure that
    # opens "1. Open the System Settings menu." near the top of page after page
    # repeats too, and dropping it as a running header left every such
    # procedure one step short in Staging.
    if key in texts and (_at_edge(bbox, rect) or _PAGE_NUMBER_KEY_RE.match(key)):
        return True
    # A band holding text on most pages is where running headers sit - but in a
    # document set tight to the top of its pages it is also where every page's
    # first line of body text sits ("KONFORMITÄTSERKLÄRUNG", "• Direktiva LVD
    # 2014/35/EU"). A running header stands apart from the body; a first line
    # runs straight on into the next one.
    return _at_edge(bbox, rect) and _band_of(bbox[1], rect) in bands and _stands_apart(doc, page_index, bbox)


def _stands_apart(doc: fitz.Document, page_index: int, bbox: tuple) -> bool:
    """Is this line set apart from the page's body - a gap wider than line
    spacing between it and the nearest line toward the middle of the page?"""
    rect = doc[page_index].rect
    height = max(1.0, bbox[3] - bbox[1])
    upper = (bbox[1] + bbox[3]) / 2 < (rect.y0 + rect.y1) / 2
    # A neighbour is any line starting further toward the middle - including one
    # whose box OVERLAPS this one: Production sets 5pt type on 5pt line spacing,
    # the next line starts 1.6pt above this one's bottom, and skipping it made a
    # paragraph's first line on a page ("että laite kierrätetään…") look set
    # apart, so it was dropped as a running header.
    gaps = [
        (lb[1] - bbox[3]) if upper else (bbox[1] - lb[3])
        for lb, _ in _text_lines(doc, page_index)
        if (lb[1] > bbox[1] + 0.5 if upper else lb[3] < bbox[3] - 0.5)
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
    key = (_doc_key(doc), page_index)
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
            })
    out.sort(key=lambda ln: (round(ln["bbox"][1], 1), ln["bbox"][0]))
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
                joiner = "" if line["text"][:1] in ".,;:)!?" else " "
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
                })
                continue
        out.append(dict(line))
    return out


def _paragraphs(lines: list[dict], heading_titles: set[str] | None = None) -> list[Element]:
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
        chars = sum(len(ln["text"]) for ln in run) or 1
        out.append(((round(run[0]["bbox"][1], 1), run[0]["bbox"][0]), Element(
            kind=KIND_TEXT,
            text=_WS_RE.sub(" ", text).strip(),
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
        return (
            gap <= height * _PARAGRAPH_GAP
            and abs(line["size"] - prev["size"]) <= _PARAGRAPH_SIZE_STEP
            and not weight_flip
            and not titled
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


def _is_fragment(el: Element) -> bool:
    """A diagram's callout number ("2", "3.") printed beside the artwork rather
    than inside it. Not a sentence, and extracted as its own text on one side
    and glued to its label on the other - so it would be reported as missing or
    added in nearly every diagram chapter while meaning nothing."""
    text = el.text.strip()
    return len(text) <= 3 and not any(ch.isalpha() for ch in text)


def _table_element(table: dict, page_index: int) -> Element:
    rows = [[_normalise(c or "") for c in row] for row in (table.get("rows") or [])]
    text = " | ".join(" ".join(c for c in row if c) for row in rows if any(row))
    return Element(
        kind=KIND_TABLE,
        text=" | ".join(
            " ".join((c or "").strip() for c in row if (c or "").strip())
            for row in (table.get("rows") or []) if any((c or "").strip() for c in row)
        ),
        key=_normalise(text),
        boxes=[(page_index, tuple(float(v) for v in table["bbox"]))],
        rows=len(rows),
        cols=max((len(r) for r in rows), default=0),
        cells=tuple(tuple(r) for r in rows),
        raw_cells=tuple(tuple((c or "").strip() for c in row) for row in (table.get("rows") or [])),
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


_TABLE_FIGURE_OVERLAP = 0.4  # a "figure" this much inside a table is the table's rules
_TEXT_FIGURE_ROWS = 4        # this many rows of text inside a "figure" make it a text block


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
    # count, and the table was reported as a figure only Staging has.
    if tables and sum(_intersection(bbox, t) for t in tables) / area >= _TABLE_FIGURE_OVERLAP:
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


def _trim_figure(bbox: tuple, lines: list[dict]) -> tuple:
    """Pull a figure's box off the prose lines kept inside its top or bottom
    quarter, so the figure is measured - and boxed - as the artwork alone."""
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    for ln in lines:
        lx0, ly0, lx1, ly1 = ln["bbox"]
        if lx1 <= x0 or lx0 >= x1 or ly1 <= y0 or ly0 >= y1:
            continue
        if ly0 - bbox[1] <= 0.25 * height:
            y0 = max(y0, ly1)
        elif bbox[3] - ly1 <= 0.25 * height:
            y1 = min(y1, ly0)
    return (x0, y0, x1, y1) if y1 - y0 >= _FIGURE_MIN_SIDE else bbox


_LEADING_ICON_GAP = 12.0  # pt: this close to where a line of text starts, it is that line's own icon


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
            prev.text = f"{prev.text} {el.text}".strip()
            prev.key = _normalise(prev.text)
            prev.boxes = prev.boxes + el.boxes
            continue
        out.append(el)
    return out


def _classify(elements: list[Element], heading_titles: set[str], body_size: float) -> None:
    """Name each text element for what it is: a heading, a note, or prose."""
    for el in elements:
        if el.kind != KIND_TEXT:
            continue
        label = i18n.match_callout_label(el.text)
        if label:
            el.kind = KIND_NOTE
            el.label = label
            continue
        title = normalize_title(el.text)
        if title and title in heading_titles and len(el.text) <= 90:
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
            ]
        except Exception:
            figures = []

        table_regions = [tuple(float(v) for v in t["bbox"]) for t in tables]
        figure_regions = [tuple(float(v) for v in f.bbox) for f in figures]
        artwork = _raster_rects(doc, page_index)
        # Rows first: whether text belongs to a figure is a question about a
        # whole printed line, never about a fragment of one.
        all_lines = _page_lines(doc, page_index, y0, y1, body_size, heading_titles)
        page_rows = _merge_rows(list(all_lines))
        # A line inside a table's outline is the table's only when one of its
        # cells holds the words. A note printed inside the frame but in no cell
        # (Production's "備考2" under each Taiwan RoHS table, the Japanese 注記)
        # was dropped as table text and so never compared on either side.
        cell_words = [
            "".join(_TOKEN_RE.findall(_normalise(" ".join(c or "" for row in (t.get("rows") or []) for c in row))))
            for t in tables
        ]

        def _held_by_table(ln: dict) -> bool:
            words = "".join(_TOKEN_RE.findall(_normalise(ln.get("text") or "")))
            for region, held in zip(table_regions, cell_words):
                if _inside(ln["bbox"], [region]):
                    return not words or words in held
            return False

        rows = _merge_rows([ln for ln in all_lines if not _held_by_table(ln)])
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
            el for el in _paragraphs(lines, heading_titles)
            if not _is_fragment(el) and not _BARE_CALLOUT_RE.match(el.text)
        ]
        for t in tables:
            table_el = _table_element(t, page_index)
            table_el.header_fill = _header_fill(doc, page_index, _header_bbox(t))
            page_elements.append(table_el)
        page_elements += [
            _figure_element(doc, page_index, f, _trim_figure(tuple(float(v) for v in f.bbox), lines))
            for f in figures
            if not _mostly_text(tuple(float(v) for v in f.bbox), lines, artwork,
                                table_regions, page_rows)
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


def _merged_table_rows(elements: list[Element]) -> tuple[list[tuple], list, list, list, list]:
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
    raw_out: list[tuple] = []
    header = elements[0].cells[0] if elements[0].cells else None
    for i, e in enumerate(elements):
        rows, grid, spans, boxes = list(e.cells), list(e.grid), list(e.spans), list(e.row_boxes)
        raws = list(e.raw_cells) if len(e.raw_cells) == len(e.cells) else list(e.cells)
        # Every header row the continuation repeats, not just the first: a
        # two-row header ("Coarse Classification | Chemical Substances" over
        # "Pb | Hg | Cd …") left its second row behind, and with a blank first
        # cell it read as the tail of the row before the page break - gluing
        # "Pb", "Hg" … onto Production's "Power cord" row.
        head_rows = list(elements[0].cells[:3]) if header else []
        k = 0
        while (i > 0 and rows and k < len(head_rows)
               and _headers_match(_row_text(head_rows[k]), _row_text(rows[0]))):
            rows, grid, spans, boxes, raws = rows[1:], grid[1:], spans[1:], boxes[1:], raws[1:]
            k += 1
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
            raw_cont = raws.pop(0) if raws else ()
            if raw_out and raw_cont:
                raw_out[-1] = tuple(
                    f"{prev} {c}".strip() if (prev or "").strip() and (c or "").strip()
                    else (prev or c)
                    for prev, c in zip(raw_out[-1], raw_cont)
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
        raw_out.extend(raws)
    return cells_out, grid_out, spans_out, boxes_out, raw_out


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
        cell_rows, grid_rows, span_rows, box_rows, raw_rows = _merged_table_rows(elements)
        table_rows: list[tuple] | None = cell_rows
        grid, spans, row_boxes = tuple(grid_rows), tuple(span_rows), tuple(box_rows)
    else:
        table_rows = None
        raw_rows = []
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
        label=next((e.label for e in elements if e.label), ""),
        rows=len(table_rows) if table_rows is not None else sum(e.rows for e in elements),
        cols=max((e.cols for e in elements), default=0),
        cells=tuple(table_rows) if table_rows is not None else tuple(row for e in elements for row in e.cells),
        raw_cells=tuple(raw_rows) if table_rows is not None else tuple(row for e in elements for row in e.raw_cells),
        width=max((e.width for e in elements), default=0.0),
        height=max((e.height for e in elements), default=0.0),
        fp=next((e.fp for e in elements if e.fp is not None), None),
        header_fill=next((e.header_fill for e in elements if e.header_fill is not None), None),
        icon_column=any(e.icon_column for e in elements),
        headers=next((e.headers for e in elements if e.headers), ()),
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
            # Whichever side's next element has no future here at all moves on.
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
            if kj in used:
                continue
            score = _pairable(a, b)
            floor = _FIGURE_PAIR_MIN if a.kind == KIND_FIGURE else _PAIR_MIN_RATIO
            if score >= floor and score > best_score:
                best, best_score = kj, score
        if best is not None:
            used.add(best)
            pairs[ki] = ([a], [pairs[best][1][0]], "moved")
            pairs[best] = None
    pairs = [p for p in pairs if p is not None]
    return [(p[0], p[1], p[2] if len(p) > 2 else "") for p in pairs]


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
    r"(?:(?<=\s)|^)((?:\d{1,2}|[a-z]|ii|iii|iv|vi|vii|viii|ix|xi|xii)[.)]"
    r"|[\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219])(?=\s)",
    re.IGNORECASE,
)
_ITEM_MIN_KEY = 6          # an item's own words must identify it
_ITEM_KEY_CHARS = 80       # ... compared over its opening, not its whole length
_MARKER_VERB = {
    "number": "numbers", "letter": "letters",
    "roman": "numbers with roman numerals", "bullet": "bullets", "dash": "marks with dashes",
}
_DASH_MARKERS = {"-", "–", "—"}


# --- inline icons ------------------------------------------------------------
_ICON_MAX_SIDE = 40.0       # an image no larger than this, inside a text block, is an inline icon
_ICON_MIN_SIDE = 5.0
_ICON_HUE_DELTA = 25.0      # degrees of dominant hue before a colour change is reported
_ICON_SAME = 0.6            # appearance match below which a paired icon is a different picture
_ICON_COLOURED_SHARE = 0.08 # share of an icon's pixels that must be coloured for it to have a hue
_ICON_CACHE: dict[tuple, list] = {}


def _page_icons(doc: fitz.Document, page_index: int) -> list[tuple]:
    key = (_doc_key(doc), page_index)
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
        _ICON_CACHE[key] = [
            (page_index, b) for b in boxes
            if _ICON_MIN_SIDE <= b[2] - b[0] <= _ICON_MAX_SIDE
            and _ICON_MIN_SIDE <= b[3] - b[1] <= _ICON_MAX_SIDE
        ]
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


def icon_changes(exp_group: list[Element], act_group: list[Element],
                 expected: fitz.Document, actual: fitz.Document,
                 exp_topic: list[Element] | None = None, act_topic: list[Element] | None = None) -> list[dict]:
    """Inline icons inside a paired block - the ⚙ in "System Settings [icon]
    menu", the icons in a table cell - compared by the word they sit beside:
    an icon only one side prints, or the same icon in a clearly different
    colour. The order of two icons beside the same word is not a change."""
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
                    side, word = _icon_anchor(doc, page_index, icon[1])
                    # A callout's own icon (Note / Warning / TIP) is callout
                    # styling - Staging's design - and a bullet is no anchor.
                    if not word or not _TOKEN_RE.search(word) or _BARE_CALLOUT_RE.match(word):
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
        entries = [e for e in entries if not _same_picture_nearby(doc_mine, e[0], doc_other, other_blocks)]
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
    label = marker.rstrip(".)")
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
    if not texts_a or not texts_b or _list_items_in(texts_a) or _list_items_in(texts_b):
        return []
    added: list[tuple] = []
    removed: list[tuple] = []
    restyled: list[tuple] = []
    for b in texts_b:
        phrase = " ".join(b.key.split()[:4])
        a = next((e for e in texts_a if phrase and phrase in e.key), None)
        if a is None:
            continue
        stage_marker = _marker_before(actual, b.pages, b.key)
        prod_marker = _marker_before(expected, a.pages, b.key)
        if stage_marker is None or prod_marker is None or stage_marker == prod_marker:
            continue
        item = (prod_marker, stage_marker, b.key, a, b)
        if stage_marker and not prod_marker:
            added.append(item)
        elif prod_marker and not stage_marker:
            removed.append(item)
        elif _marker_kind(prod_marker) != _marker_kind(stage_marker):
            restyled.append(item)
    out: list[dict] = []
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
    key = (_doc_key(doc), page_index)
    if key not in _FILL_CACHE:
        page = doc[page_index]
        fills = []
        try:
            for d in page.get_drawings():
                fill = d.get("fill")
                r = fitz.Rect(d["rect"])
                if (fill and min(fill) < _SHADE_WHITE and r.width >= 30 and r.height >= 12
                        and r.get_area() < 0.6 * page.rect.get_area()):
                    fills.append(r)
        except Exception:
            pass
        _FILL_CACHE[key] = fills
    return _FILL_CACHE[key]


def _on_shading(doc: fitz.Document, el: Element) -> bool:
    if not el.boxes:
        return False
    page_index, bbox = el.boxes[0]
    r = fitz.Rect(bbox)
    area = max(1.0, r.get_area())
    return any((f & r).get_area() >= _SHADE_COVER * area for f in _page_fills(doc, page_index))


def shading_changes(exp_group: list[Element], act_group: list[Element],
                    expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """The same text printed on a shaded box in one document only - Staging sets
    every "Important" note on a green panel Production does not have."""
    texts_a = [e for e in exp_group if e.kind != KIND_FIGURE and e.kind != KIND_TABLE]
    texts_b = [e for e in act_group if e.kind != KIND_FIGURE and e.kind != KIND_TABLE]
    if not texts_a or not texts_b:
        return []
    shaded_a = any(_on_shading(expected, e) for e in texts_a)
    shaded_b = any(_on_shading(actual, e) for e in texts_b)
    # Staging's shaded panels are its design: shading it ADDS is expected and
    # never reported. Only shading it lacks is.
    if not shaded_a or shaded_b:
        return []
    return [{
        "type": "shading", "kind": texts_b[0].kind,
        "summary": ("Background shading added in Staging — the text is printed on a shaded box "
                    "that Production does not have." if shaded_b else
                    "Background shading missing in Staging — Production prints the text on a shaded box, "
                    "Staging on the plain page."),
        "detail": "",
    }]


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
        parts = _ITEM_MARKER_RE.split(el.text)
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
                "summary": (f"Table header missing on the continued page — the table that starts on Staging "
                            f"p.{anchor_page + 1} continues onto p.{cont['page'] + 1} without repeating its header row."),
                "detail": (f"Header row: “{(t.get('header') or '')[:100]}” · first row on p.{cont['page'] + 1}: "
                           f"“{(cont.get('first_row') or '')[:100]}”"),
            })
    return out


# --- list items (ul / li) -----------------------------------------------------

_LIST_ITEM_MIN_KEY = 12
_MARKER_REACH = 30.0          # points left of an item's first word its marker may sit
_MARKER_TAIL_RE = re.compile(
    r"(\d{1,2}[.)]|[a-z][.)]|[\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219\u25cf\u25a0\u2013\u2014-])\s*$",
    re.IGNORECASE,
)
_CHAR_CACHE: dict[tuple, list] = {}


def _page_chars(doc: fitz.Document, page_index: int) -> list[tuple]:
    key = (_doc_key(doc), page_index)
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


def _marker_before(doc: fitz.Document, pages: list[int], key: str) -> str | None:
    """What is printed just left of this item's first words on the page: a
    marker ("1.", "a)", "•", or a small drawn dot), "" when the words are found
    with nothing before them, None when they cannot be found at all.

    This is read off the page, not off the extracted paragraph, because the two
    documents extract markers differently and the extraction is exactly what
    cannot be trusted here: a table cell "(W × H × D)" parsed as list item
    "D)", a wrapped line read as a new item.
    """
    phrase = " ".join(key.split()[:4])
    for page_index in pages:
        try:
            hits = doc[page_index].search_for(phrase)
        except Exception:
            hits = []
        if not hits:
            continue
        r = hits[0]
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
            return None if joined[: m.start()].strip() else m.group(1)
        try:
            for d in doc[page_index].get_drawings():
                dr = d.get("rect")
                if dr is not None and dr.width <= 9 and dr.height <= 9 and \
                        r.x0 - _MARKER_REACH <= dr.x1 <= r.x0 + 1 and dr.y0 < r.y1 and dr.y1 > r.y0:
                    return "•"
        except Exception:
            pass
        return ""
    return None


def list_structure_changes(exp: list[Element], act: list[Element],
                           expected: fitz.Document | None = None,
                           actual: fitz.Document | None = None) -> list[dict]:
    """List items that stopped being list items, or started.

    An item Production prints as "• Keep the remote dry" whose words Staging
    prints with no marker at all has lost its list formatting (the <li> became
    a plain paragraph) - and the reverse. A marker that only changed STYLE is
    `numbering_changes`; an item whose words are gone is a content difference.
    """
    exp_items = {(it[3].section, it[2]): it for it in _list_items_in(exp)}
    act_items = {(it[3].section, it[2]): it for it in _list_items_in(act)}
    out: list[dict] = []
    restyled: dict[tuple, list] = {}
    for items, other_items, other, kind, own_doc, other_doc in (
        (exp_items, act_items, act, "list-marker-missing", expected, actual),
        (act_items, exp_items, exp, "list-marker-added", actual, expected),
    ):
        groups: dict[tuple, list] = {}
        for (section, key), item in items.items():
            if (section, key) in other_items or len(key) < _LIST_ITEM_MIN_KEY:
                continue
            # Same kind of block only: a list item never answers to a heading
            # ("proxy settings" inside "Configuring proxy settings").
            holder = next((e for e in other if e.kind not in (KIND_FIGURE, KIND_HEADING)
                           and e.section == section and key in e.key), None)
            if holder is None:
                continue
            if own_doc is None or other_doc is None:
                continue
            # Confirmed on the pages: a marker printed before the item on this
            # side, and the same words found on the other side with none.
            own_marker = _marker_before(own_doc, item[3].pages, key)
            if not own_marker:
                continue
            other_marker = _marker_before(other_doc, holder.pages, key)
            if other_marker is None:
                continue
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
_NEXT_ITEM_RE = re.compile(r"^\s*(?:[•●▪◦·\-–—]|\(?[0-9]{1,2}[.)]|\(?[a-zA-Z][.)])\s")
_LINE_GEOM_CACHE: dict[tuple, list] = {}


def _text_lines(doc: fitz.Document, page_index: int) -> list[tuple[tuple, str]]:
    """Every text line on the page as (bbox, text)."""
    key = (_doc_key(doc), page_index)
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

    exp_items, act_items = unique(_list_items_in(exp)), unique(_list_items_in(act))
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
    key = _doc_key(doc)
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
    named = [e for e in entries if e.page == page and normalize_title(e.title)
             and normalize_title(e.title) in words]
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
    out: list[dict] = []
    for a, b in _pair_links(exp_links, act_links):
        if b is None:
            # Only a LINK difference when the words are still there to click:
            # text that is gone altogether is already a content difference.
            if a.key in act_blob and (a.section, a.label) not in act_targets:
                out.append({"type": "link-missing", "kind": KIND_LINK, "exp": a, "act": None,
                            "summary": f"Hyperlink missing in Staging — “{a.text[:80]}” is a link in Production and plain text in Staging.",
                            "detail": f"Production link goes to: {a.label}"})
        elif a is None:
            if b.detail:
                out.append({"type": "link-broken", "kind": KIND_LINK, "exp": None, "act": b,
                            "summary": f"Hyperlink not working in Staging — “{b.text[:80] or 'link'}”: {b.detail}.",
                            "detail": ""})
            elif b.key in exp_blob and (b.section, b.label) not in exp_targets:
                out.append({"type": "link-added", "kind": KIND_LINK, "exp": None, "act": b,
                            "summary": f"Hyperlink added in Staging — “{b.text[:80]}” is plain text in Production.",
                            "detail": f"Staging link goes to: {b.label}"})
        elif b.detail and not a.detail:
            out.append({"type": "link-broken", "kind": KIND_LINK, "exp": a, "act": b,
                        "summary": f"Hyperlink not working in Staging — “{b.text[:80]}”: {b.detail}.",
                        "detail": f"Production link goes to: {a.label}"})
        elif a.label != b.label and not (a.label.startswith(("page ", "unresolved")) or b.label.startswith("page ")):
            out.append({"type": "link-target", "kind": KIND_LINK, "exp": a, "act": _on_words(actual, b, a.text),
                        "summary": f"Hyperlink goes somewhere else — “{a.text[:80]}”.",
                        "detail": f"Production: {a.label} · Staging: {b.label}"})
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
_FIGURE_ASPECT_LIMIT = 2.0   # width/height ratios further apart than this are different figures
_SIDE_BY_SIDE_Y = 12.0      # points: figures whose tops are this close sit in one row
_SIDE_BY_SIDE_GAP = 40.0    # points: ... and this close across are one group of artwork
_FIGURE_ELSEWHERE = 0.72     # a figure found elsewhere must look this alike
_LABEL_BONUS = 0.35
_FIG_INDEX: dict[int, list] = {}
# The smallest a figure can be in each document, scaled to its type size (set by
# `compare_chapters`): the same logo prints at 13pt in a document set in 5pt type
# and at 26pt in one set in 12pt, and one absolute minimum dropped it from the
# first while keeping it in the second.
_FIGURE_MIN_BY_DOC: dict[int, float] = {}


def _figure_min_side(doc: fitz.Document) -> float:
    return _FIGURE_MIN_BY_DOC.get(_doc_key(doc), _FIGURE_MIN_SIDE)


def _caption(fig: Element, texts: list[Element]) -> str:
    """The label printed nearest a figure, on its own page, if any."""
    x0, y0, x1, y1 = fig.raw_bbox or fig.bbox
    best, best_distance = None, None
    for t in texts:
        if t.page != fig.page or not t.text or len(t.text) > _CAPTION_MAX_CHARS:
            continue
        if t.kind == KIND_HEADING:
            continue  # a heading names the section, not the picture under it
        tx0, ty0, tx1, ty1 = t.bbox
        dx = max(tx0 - x1, x0 - tx1, 0.0)
        dy = max(ty0 - y1, y0 - ty1, 0.0)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= _CAPTION_REACH and (best_distance is None or distance < best_distance):
            best, best_distance = t, distance
    return best.text if best is not None else ""


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
    key = _doc_key(doc)
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


_OVERSIZE_STEP = 0.20       # a figure this much larger in Staging is oversized ...
_OVERSIZE_MIN_PT = 12.0     # ... and by at least this many points
_ALIGN_CENTRE = 0.12        # centre within this share of the text column = centred
_ALIGN_SHIFT = 0.12         # and it must move at least this far to count
_MARGIN_CACHE: dict[int, tuple] = {}


def _text_column(doc: fitz.Document, page_index: int) -> tuple[float, float]:
    key = _doc_key(doc)
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


def _page_words(doc: fitz.Document):
    from pdfval import ocr
    key = _doc_key(doc)
    if key not in _LABEL_WORDS:
        _LABEL_WORDS[key] = ocr.PageWords(doc)
    return _LABEL_WORDS[key]


def _label_keys(words: list[str]) -> set[str]:
    from pdfval import ocr
    keys = {ocr.normalize_word(w) for w in words}
    return {k for k in keys if len(k) >= _LABEL_MIN_CHARS and any(ch.isalpha() for ch in k)}


_CALLOUT_NUMBER_RE = re.compile(r"^\(?(\d{1,2})[.)]?$")
_CALLOUT_REACH = 30.0  # points around a figure where its callout numbers are printed


def _numbers_near(words: list[tuple], bbox: tuple) -> Counter:
    x0, y0, x1, y1 = bbox
    found: Counter = Counter()
    for wx0, wy0, wx1, wy1, text in words:
        m = _CALLOUT_NUMBER_RE.match((text or "").strip())
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if m and x0 - _CALLOUT_REACH <= cx <= x1 + _CALLOUT_REACH and y0 - _CALLOUT_REACH <= cy <= y1 + _CALLOUT_REACH:
            found[m.group(1)] += 1
    return found


def _callout_numbers_missing(a: Element, b: Element, name: str,
                             expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A diagram's callout numbers ("1", "2", "3.") are image labels: the user
    wants every one Production prints on or beside a figure found on Staging's
    copy - in its text, or read off the picture. `_is_fragment` keeps them out
    of the text comparison, so this is the only place they are checked."""
    exp_box, act_box = a.raw_bbox or a.bbox, b.raw_bbox or b.bbox
    wanted = _numbers_near(_page_words(expected).native_words(a.page), exp_box)
    if not wanted:
        return []
    have = _numbers_near(_page_words(actual).native_words(b.page), act_box)
    missing = wanted - have
    if missing:
        missing -= _numbers_near(_page_words(actual).ocr_words(b.page), act_box)
    if not missing:
        return []
    numbers = ", ".join(f"“{n}”" for n in sorted(missing.elements(), key=lambda v: int(v))[:10])
    return [{"type": "figure-label-missing", "kind": KIND_FIGURE,
             "summary": f"Image label missing in Staging — {name}: callout number {numbers} printed on the Production figure, not on the Staging one.",
             "detail": f"{sum(missing.values())} callout number(s) missing"}]


def figure_label_missing(a: Element, b: Element, name: str,
                         expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """Words printed ON the Production figure that the Staging figure does not
    carry. Staging very often bakes labels into the picture as pixels, so the
    Staging figure is read with OCR before a word is called missing - and when
    OCR cannot read it at all, nothing is asserted."""
    from pdfval import ocr
    exp_box = a.raw_bbox or a.bbox
    act_box = b.raw_bbox or b.bbox
    out = _callout_numbers_missing(a, b, name, expected, actual)
    wanted = _label_keys(ocr.words_in(_page_words(expected).native_words(a.page), exp_box))
    if not wanted:
        return out
    have = _label_keys(ocr.words_in(_page_words(actual).native_words(b.page), act_box, margin=4))
    missing = wanted - have
    if not missing:
        return out
    read = ocr.words_in(_page_words(actual).ocr_words(b.page), act_box, margin=4)
    if not read:
        return out  # could not read the Staging figure - cannot tell dropped from rasterised
    missing -= _label_keys(read)
    if not missing:
        return out
    words = ", ".join(f"“{w}”" for w in sorted(missing)[:8])
    return out + [{"type": "figure-label-missing", "kind": KIND_FIGURE,
             "summary": f"Image label missing in Staging — {name}: {words} printed on the Production figure, not on the Staging one.",
             "detail": f"{len(missing)} label word(s) missing"}]


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


# --- black spots ------------------------------------------------------------
#
# The same picture in both documents, but Staging's copy carries dark marks
# Production's does not - specks from a bad raster conversion, a transparency
# flattened to a black blot. Both figures are rendered at one common size and
# every near-black mark in Staging is looked for in Production nearby; a compact
# blot with nothing dark near it in Production, on a patch that is light there,
# is a spot.
_SPOT_WIDTH = 360            # px both figures are rendered to
_SPOT_DARK = 70              # grey at or below this is black
_SPOT_LIGHT = 185            # Production is at least this light where a spot sits
_SPOT_REACH = 0.025          # share of the width a mark may sit off its Production counterpart
_SPOT_MIN_PX = 9             # a spot covers at least this many pixels ...
_SPOT_MIN_AREA = 0.0003      # ... and this share of the figure ...
_SPOT_MAX_AREA = 0.04        # ... and at most this - a bigger change is a different picture
_SPOT_MIN_FILL = 0.40        # a blot fills this much of its own box ...
_SPOT_MIN_THICK = 3          # ... is at least this many px across both ways ...
_SPOT_MIN_SQUARENESS = 0.3   # ... and not a line: its short side at least this share of its long one
_SPOT_MAX_NEW_INK = 0.08     # more new black than this and the pictures simply differ
_SPOT_ASPECT_GAP = 0.08      # boxes this differently shaped cannot be laid over each other


def _figure_grey(doc: fitz.Document, page_index: int, rect: "fitz.Rect", width: int, height: int):
    """The figure's region as a greyscale array of exactly width x height."""
    import numpy as np
    from PIL import Image

    rect = rect & doc[page_index].rect
    if rect.is_empty or rect.width < 8 or rect.height < 8:
        return None
    zoom = max(width / rect.width, height / rect.height, 0.5)
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect,
                                     colorspace=fitz.csGRAY, alpha=False)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples).resize((width, height), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def _grow(mask, steps: int):
    """A boolean mask widened by `steps` pixels in every direction."""
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _blobs(mask) -> list[tuple[int, int, int, int, int]]:
    """Connected regions of a boolean mask, as (x0, y0, x1, y1, area)."""
    import numpy as np

    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack = [(int(y), int(x))]
        x0 = x1 = int(x)
        y0 = y1 = int(y)
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            x0, x1, y0, y1 = min(x0, cx), max(x1, cx), min(y0, cy), max(y1, cy)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((x0, y0, x1 + 1, y1 + 1, area))
    return out


def black_spots(a: Element, b: Element, name: str,
                expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """Dark spots on Staging's copy of a figure that Production's copy does not
    have. `marks` carries each spot's box on the Staging page."""
    if a is None or b is None or not a.boxes or not b.boxes:
        return []
    # Only a picture confidently the same on both sides can be laid over its
    # counterpart: two model drawings that merely look alike differ by lines
    # that would all read as "new ink".
    if _figure_similarity(a, b) < _FIGURE_SAME:
        return []
    use_raw = a.raw_bbox is not None and b.raw_bbox is not None
    ra = fitz.Rect(a.raw_bbox if use_raw else a.bbox)
    rb = fitz.Rect(b.raw_bbox if use_raw else b.bbox)
    if ra.width < 8 or ra.height < 8 or rb.width < 8 or rb.height < 8:
        return []
    shape_a, shape_b = ra.height / ra.width, rb.height / rb.width
    if abs(shape_a - shape_b) / shape_a > _SPOT_ASPECT_GAP:
        return []
    width = _SPOT_WIDTH
    height = max(8, round(width * shape_a))
    try:
        prod = _figure_grey(expected, a.page, ra, width, height)
        stage = _figure_grey(actual, b.page, rb, width, height)
    except Exception:
        return []
    if prod is None or stage is None:
        return []
    total = width * height
    new_ink = (stage <= _SPOT_DARK) & ~_grow(prod <= _SPOT_DARK, max(2, round(_SPOT_REACH * width)))
    if new_ink.sum() > _SPOT_MAX_NEW_INK * total:
        return []
    spots = []
    for x0, y0, x1, y1, area in _blobs(new_ink):
        if area < max(_SPOT_MIN_PX, _SPOT_MIN_AREA * total) or area > _SPOT_MAX_AREA * total:
            continue
        if area / max(1, (x1 - x0) * (y1 - y0)) < _SPOT_MIN_FILL:
            continue
        short, long_ = sorted((x1 - x0, y1 - y0))
        if short < _SPOT_MIN_THICK or short / long_ < _SPOT_MIN_SQUARENESS:
            continue  # a drawn line that shifted a little, not a blot
        patch = prod[max(0, y0 - 3): y1 + 3, max(0, x0 - 3): x1 + 3]
        if patch.size and patch.mean() < _SPOT_LIGHT:
            continue
        spots.append((x0, y0, x1, y1))
    if not spots:
        return []
    sx, sy = rb.width / width, rb.height / height
    marks = [(b.page, (rb.x0 + x0 * sx, rb.y0 + y0 * sy, rb.x0 + x1 * sx, rb.y0 + y1 * sy))
             for x0, y0, x1, y1 in spots]
    big = max(spots, key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
    count = len(spots)
    return [{
        "type": "figure-spots", "kind": KIND_FIGURE,
        "summary": (f"Black spots in Staging image — {count} dark {'mark' if count == 1 else 'marks'} "
                    f"on {name} that Production's copy of the picture does not have."),
        "detail": f"largest {(big[2] - big[0]) * sx:.0f}×{(big[3] - big[1]) * sy:.0f}pt",
        "marks": marks,
    }]


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
            if any(p == page_index and (fitz.Rect(t) & r).get_area() >= 0.5 * r.get_area() for p, t in taken):
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


# --- words printed INSIDE a figure, and its resolution -----------------------
#
# A screenshot's menu labels and a diagram's callouts are content the reader
# sees; a blurry screenshot is still read, never skipped. Each matched figure
# pair is OCR'd at print resolution, the words the PDF's own text layer already
# prints over the crop are set aside (a drawn figure's box often takes in its
# caption), and what one picture says that the other does not is reported.

_FIGURE_OCR_DPI = 300
_FIGURE_WORD_MIN = 3             # OCR fragments shorter than this are noise ("a", "if")
_FIGURE_OCR_ALIKE = 0.8          # an OCR misread of the same word ("ofyour" / "of your")
_PIXELATED_DPI = 150.0           # an embedded image below this prints visibly soft
_PIXELATED_SHARE = 0.5           # ... and at under half Production's resolution
_FIGURE_WORDS_CACHE: dict[tuple, "Counter[str]"] = {}
_FIGURE_OCR_MIN_SIDE = 40.0      # points: smaller figures are symbols, not text-bearing pictures


class _quiet_stderr:
    """Silence the C-level stderr for the duration (Tesseract writes its
    "Line cannot be recognized!!" notices straight to file descriptor 2)."""

    def __enter__(self):
        import os
        import sys
        try:
            sys.stderr.flush()
            self._saved = os.dup(2)
            self._null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._null, 2)
        except OSError:
            self._saved = None
        return self

    def __exit__(self, *exc):
        import os
        if self._saved is not None:
            os.dup2(self._saved, 2)
            os.close(self._saved)
            os.close(self._null)
        return False


def _figure_words(doc: fitz.Document, page_index: int, bbox: tuple) -> "Counter[str]":
    key = (_doc_key(doc), page_index, tuple(round(v, 1) for v in bbox))
    if key in _FIGURE_WORDS_CACHE:
        return _FIGURE_WORDS_CACHE[key]
    words: Counter = Counter()
    try:
        page = doc[page_index]
        clip = fitz.Rect(bbox) & page.rect
        # A symbol this small holds no readable words, and Tesseract complains
        # on stderr ("Image too small to scale!!") about every one it is given.
        if clip.width >= _FIGURE_OCR_MIN_SIDE and clip.height >= _FIGURE_OCR_MIN_SIDE:
            pix = page.get_pixmap(clip=clip, dpi=_FIGURE_OCR_DPI, alpha=False)
            with _quiet_stderr():
                read = fitz.open("pdf", pix.pdfocr_tobytes(language="eng", tessdata=ocr.tessdata_dir()))
            printed = Counter(ocr.normalize_word(w[4]) for w in page.get_text("words", clip=clip))
            for w in read[0].get_text("words"):
                token = ocr.normalize_word(w[4])
                if len(token) < _FIGURE_WORD_MIN or not any(ch.isalpha() for ch in token):
                    continue
                if printed[token] > 0:
                    printed[token] -= 1  # the text layer already prints it: not artwork
                    continue
                words[token] += 1
    except Exception:
        words = Counter()
    _FIGURE_WORDS_CACHE[key] = words
    return words


def _embedded_dpi(doc: fitz.Document, page_index: int, bbox: tuple) -> float | None:
    """The resolution an embedded image prints at; None for a drawing."""
    try:
        infos = doc[page_index].get_image_info()
    except Exception:
        return None
    target = fitz.Rect(bbox)
    best = None
    for info in infos:
        r = fitz.Rect(info["bbox"])
        inter = r & target
        if inter.is_empty or r.width <= 0:
            continue
        if inter.width * inter.height >= 0.6 * max(1.0, target.width * target.height):
            dpi = float(info.get("width") or 0) / (r.width / 72.0)
            best = dpi if best is None else min(best, dpi)
    return best


def figure_text_changes(a: Element, b: Element, name: str,
                        expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    out: list[dict] = []
    if ocr.available():
        wa = _figure_words(expected, a.page, a.bbox)
        wb = _figure_words(actual, b.page, b.bbox)

        def unexplained(mine: Counter, theirs: Counter) -> list[str]:
            gone = []
            for token, n in (mine - theirs).items():
                if any(difflib.SequenceMatcher(a=token, b=t).ratio() >= _FIGURE_OCR_ALIKE for t in theirs):
                    continue  # the same word, read slightly differently
                gone.extend([token] * n)
            return gone

        if sum(wa.values()) >= _FIGURE_WORD_MIN or sum(wb.values()) >= _FIGURE_WORD_MIN:
            for kind, gone, verb in (("figure-text-missing", unexplained(wa, wb), "missing in Staging"),
                                     ("figure-text-added", unexplained(wb, wa), "added in Staging")):
                if len(gone) >= 2 or any(len(t) >= 5 for t in gone):
                    shown = ", ".join(f"“{t}”" for t in gone[:8]) + (" …" if len(gone) > 8 else "")
                    out.append({"type": kind, "kind": KIND_FIGURE, "exp": a, "act": b,
                                "summary": f"Text inside the figure {verb} — {name}: {shown}.",
                                "detail": "read from the picture"})
    dpi_a = None if a.drawn else _embedded_dpi(expected, a.page, a.bbox)
    dpi_b = None if b.drawn else _embedded_dpi(actual, b.page, b.bbox)
    if dpi_b is not None and dpi_b < _PIXELATED_DPI and (a.drawn or (dpi_a and dpi_b < _PIXELATED_SHARE * dpi_a)):
        against = "drawn at full sharpness" if a.drawn else f"about {dpi_a:.0f} dpi"
        out.append({"type": "figure-pixelated", "kind": KIND_FIGURE, "exp": a, "act": b,
                    "summary": (f"Figure is pixelated in Staging — {name} is embedded at about {dpi_b:.0f} dpi "
                                f"in Staging, {against} in Production."),
                    "detail": ""})
    return out


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

    candidates = []
    for i, a in enumerate(exp_figs):
        for j, b in enumerate(act_figs):
            if a.section and b.section and a.section != b.section:
                continue  # a figure is only ever the same figure within its own topic
            look = _figure_similarity(a, b)
            labels = _labels_agree(exp_caps[i], act_caps[j])
            if not (look >= _FIGURE_MATCH or (labels and look >= _FIGURE_MATCH_WITH_LABEL)):
                continue
            if labels is False and look < _FIGURE_SAME:
                continue  # different labels, not clearly the same picture
            candidates.append((look + (_LABEL_BONUS if labels else 0.0), i, j, look, labels))
    candidates.sort(reverse=True)
    used_exp: set[int] = set()
    used_act: set[int] = set()
    rows: list[dict] = []
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
            diffs.extend(figure_label_missing(a, b, name, expected, actual))
        if look >= _FIGURE_DIFFERENT and a.drawn == b.drawn:
            # Spots are read off the rendered picture: a drawing against an
            # embedded image differs in anti-aliasing specks, not ink.
            diffs.extend(black_spots(a, b, name, expected, actual))
        diffs.extend(figure_text_changes(a, b, name, expected, actual))
        rows.append({"exp": [a], "act": [b], "differences": diffs})

    # Second chance: figures still without a partner on both sides pair on how
    # they look alone. A label beside a figure is good evidence when it agrees,
    # but two documents often label the same drawing differently - a heading
    # above it on one side, an "(a) Display (b) Wall" legend on the other.
    mixed_pairs: list[tuple[Element, Element]] = []
    leftovers = sorted(
        ((_figure_similarity(exp_figs[i], act_figs[j]), i, j)
         for i in range(len(exp_figs)) if i not in used_exp
         for j in range(len(act_figs)) if j not in used_act
         if not (exp_figs[i].section and act_figs[j].section and exp_figs[i].section != act_figs[j].section)),
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
        # Never the same figure when the shapes differ this much: Production's
        # 40×21 WEEE-plus-battery pair against Staging's 46×74 bin alone.
        ra = exp_figs[i].width / max(1.0, exp_figs[i].height)
        rb = act_figs[j].width / max(1.0, act_figs[j].height)
        if max(ra, rb) / max(1e-6, min(ra, rb)) > _FIGURE_ASPECT_LIMIT:
            continue
        used_exp.add(i)
        used_act.add(j)
        a, b = exp_figs[i], act_figs[j]
        name = _describe(a, exp_caps[i] or act_caps[j])
        diffs = list(_figure_layout(a, b, name, expected, actual, exp_texts + exp_figs, act_texts + act_figs, scale=scale))
        if mixed:
            mixed_pairs.append((a, b))
            diffs.extend(figure_label_missing(a, b, name, expected, actual))
            diffs.extend(figure_text_changes(a, b, name, expected, actual))
            rows.append({"exp": [a], "act": [b], "differences": diffs})
            continue
        if look >= _FIGURE_DIFFERENT:
            diffs.extend(black_spots(a, b, name, expected, actual))
        diffs.extend(figure_text_changes(a, b, name, expected, actual))
        if look < _FIGURE_DIFFERENT:
            diffs.append({"type": "figure-different", "kind": KIND_FIGURE,
                          "summary": f"Figure is a different picture — {name} is labelled differently in the two documents and the pictures do not match.",
                          "detail": f"appearance match {look:.0%} · Production label: “{exp_caps[i] or '—'}” · Staging label: “{act_caps[j] or '—'}”"})
        rows.append({"exp": [a], "act": [b], "differences": diffs})

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
    for i, a in enumerate(exp_figs):
        if i in used_exp or beside_mixed_partner(a, "exp"):
            continue
        name = _describe(a, exp_caps[i])
        here = _printed_in_band(a, actual, act_bands, taken_act, act_section_at) if act_bands else None
        if here is not None:
            rows.append({"exp": [a], "act": [here], "differences": []})
            continue
        found = None  # looked for in its own topic only - never elsewhere in the document
        if found is not None:
            where = heading_at(act_entries, found.page, found.bbox[1]) or "another section"
            diff = {"type": "figure-elsewhere", "kind": KIND_FIGURE, "exp": a, "act": found,
                    "summary": f"Figure is in a different section in Staging — {name} is here in Production, but Staging prints it under “{where}” (p.{found.page + 1}).",
                    "detail": ""}
        else:
            diff = {"type": "figure-missing", "kind": KIND_FIGURE, "exp": a, "act": None,
                    "summary": f"Figure missing in Staging — {name} is not printed anywhere in Staging.",
                    "detail": f"{a.width:.0f}×{a.height:.0f}pt"}
        rows.append({"exp": [a], "act": [], "differences": [diff]})

    for j, b in enumerate(act_figs):
        if j in used_act or beside_mixed_partner(b, "act"):
            continue
        name = _describe(b, act_caps[j])
        here = _printed_in_band(b, expected, exp_bands, taken_exp, exp_section_at) if exp_bands else None
        if here is not None:
            rows.append({"exp": [here], "act": [b], "differences": []})
            continue
        found = None  # looked for in its own topic only - never elsewhere in the document
        if found is not None:
            where = heading_at(exp_entries, found.page, found.bbox[1]) or "another section"
            diff = {"type": "figure-elsewhere", "kind": KIND_FIGURE, "exp": found, "act": b,
                    "summary": f"Figure is in a different section in Production — Staging prints {name} here, but Production has it under “{where}” (p.{found.page + 1}).",
                    "detail": ""}
        else:
            diff = {"type": "figure-added", "kind": KIND_FIGURE, "exp": None, "act": b,
                    "summary": f"Figure only in Staging — {name} is not printed anywhere in Production.",
                    "detail": f"{b.width:.0f}×{b.height:.0f}pt"}
        rows.append({"exp": [], "act": [b], "differences": [diff]})
    return rows


# --- the same content, laid out differently --------------------------------

_LAYOUT_MIN_CHARS = 4
_LAYOUT_SHORT_CHARS = 20          # below this, only a container that STARTS with it counts
_LAYOUT_COVERAGE = 0.9            # share of a long element's words present on the other side
_LAYOUT_COVERAGE_MIN_TOKENS = 12


def _flat(text: str) -> str:
    return _WS_RE.sub(" ", _normalise(text).replace("|", " ")).strip()


def _count_phrase(haystack: str, needle: str) -> int:
    """How many times `needle` is printed in `haystack`, as whole words."""
    return len(re.findall(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack)) if needle else 0


def _has_phrase(haystack: str, needle: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack))


def _present_elsewhere(el: Element, others: list[Element], flats: list[str],
                       blob: str, tokens: "Counter[str]", own_blob: str = "") -> tuple[bool, Element | None]:
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
    # The words printed together, as often as this side prints them. Nearly all
    # of the words scattered through the topic is NOT the text being printed:
    # that excuse hid genuinely deleted sentences made of common words.
    if _count_phrase(blob, needle) >= max(1, _count_phrase(own_blob, needle)):
        return True, None
    return False, None


def reconcile_layout(pairs: list[tuple], exp: list[Element], act: list[Element]) -> list[tuple]:
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
    exp_flats = [_flat(e.text) if e.kind != KIND_FIGURE else "" for e in exp]
    act_flats = [_flat(e.text) if e.kind != KIND_FIGURE else "" for e in act]
    exp_blob, act_blob = " ".join(f for f in exp_flats if f), " ".join(f for f in act_flats if f)
    exp_tokens, act_tokens = _tokens(exp_blob), _tokens(act_blob)

    out: list[tuple] = []
    for exp_group, act_group, note in pairs:
        container = None
        if exp_group and not act_group and len(exp_group) == 1:
            found, container = _present_elsewhere(exp_group[0], act, act_flats, act_blob, act_tokens, exp_blob)
            if found:
                note = "layout"
        elif act_group and not exp_group and len(act_group) == 1:
            found, container = _present_elsewhere(act_group[0], exp, exp_flats, exp_blob, exp_tokens, act_blob)
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
            "summary": f"{KIND_LABEL[a.kind]} moved — it is printed in a different place in Staging.",
            "detail": (a.text or "")[:200], "minor": True,
        })
    out.extend(_compare_merged(a, b, size_scale))
    return reportable(out)


def _compare_merged(a: Element | None, b: Element | None, size_scale: float) -> list[dict]:
    if a is None and b is None:
        return []
    if b is None:
        return [{
            "type": "missing",
            "kind": a.kind,
            "summary": f"{KIND_LABEL[a.kind]} in Production only — nothing answers to it in Staging.",
            "detail": a.text[:400] if a.text else f"{int(a.width)}×{int(a.height)}pt figure",
        }]
    if a is None:
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

    if a.kind == KIND_NOTE and a.label != b.label:
        out.append({
            "type": "note-label", "kind": KIND_NOTE,
            "summary": f"Callout type changed — “{a.label}” in Production, “{b.label or 'none'}” in Staging.",
            "detail": "",
        })

    # Neither direction is reported on headings or heading-like labels: their
    # styling is Staging's design (see `bold-added` below).
    lost = [] if KIND_HEADING in (a.kind, b.kind) else _bold_missing(a, b)
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
        else _bold_missing(b, a)
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


def _content_summary(kind: str, gone: list[str], extra: list[str]) -> str:
    parts = []
    if gone:
        parts.append("missing in Staging: " + "; ".join(f"“{r[:90]}”" for r in gone[:4]) + (" …" if len(gone) > 4 else ""))
    if extra:
        parts.append("extra in Staging: " + "; ".join(f"“{r[:90]}”" for r in extra[:4]) + (" …" if len(extra) > 4 else ""))
    return f"{KIND_LABEL.get(kind, 'Text')} content changed — " + " · ".join(parts) + "."


_ELSEWHERE_MIN_WORDS = 3


_SQUASHED_MIN_CHARS = 12   # below this, text found with its spaces ignored proves too little


_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_SHORT_LABEL_TOKENS = 6   # a label this short may be printed inside a longer line or a table


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


def _topic_words(exp: list[Element], act: list[Element]) -> tuple:
    """Everything each side prints in one topic, in the forms the settle checks
    read: (exp words, act words, exp units, act units, exp line keys, act line keys)."""
    def words(elements: list[Element]) -> str:
        return " " + " ".join(w for e in elements for w in _TOKEN_RE.findall(e.key)) + " "

    def keys(elements: list[Element]) -> set[str]:
        return {" ".join(_TOKEN_RE.findall(e.key)) for e in elements if e.key}

    return words(exp), words(act), _chapter_units(exp), _chapter_units(act), keys(exp), keys(act)


def _printed_in(el: Element, other_words: str, other_units: Counter | None = None,
                own_words: str | None = None) -> bool:
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
    phrase = f" {run} "
    # Printed there at least as often as here: a paragraph printed twice in
    # Production and once in Staging has lost a copy.
    need = max(1, own_words.count(phrase)) if own_words else 1
    if other_words.count(phrase) >= need:
        return True
    squashed = run.replace(" ", "")
    minimum = 4 if _CJK_RE.search(run) else _SQUASHED_MIN_CHARS
    need_solid = max(1, own_words.replace(" ", "").count(squashed)) if own_words else 1
    if len(squashed) >= minimum and other_words.replace(" ", "").count(squashed) >= need_solid:
        return True
    # A short label counts only as the same words in order - its words merely
    # scattered through the topic ("the", "display") is not the label printed.
    return False


def _elsewhere(run: str, other_words: str, own_words: str | None = None) -> bool:
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
    phrase = f" {run} "
    # Excused only when the other side prints the run at least as often as this
    # side does: boilerplate printed twice in Production and once in Staging has
    # genuinely lost a copy, even though the words are "elsewhere".
    need = max(1, own_words.count(phrase)) if own_words else 1
    if len(words) >= _ELSEWHERE_MIN_WORDS and other_words.count(phrase) >= need:
        return True
    squashed = run.replace(" ", "")
    other_solid = other_words.replace(" ", "")
    need_solid = max(1, own_words.replace(" ", "").count(squashed)) if own_words else 1
    if _CJK_RE.search(run):
        return len(squashed) >= 4 and other_solid.count(squashed) >= need_solid
    if len(words) >= _ELSEWHERE_MIN_WORDS and len(squashed) >= _SQUASHED_MIN_CHARS \
            and other_solid.count(squashed) >= need_solid:
        return True
    return len(words) <= _SHORT_LABEL_TOKENS and len(squashed) >= 6 and other_words.count(phrase) >= need


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
_HEADER_ROWS = 3  # a table's column names are printed within its first rows
_ROW_MATCH_LABELLED = 0.5      # ... or this share, when their first cell (the row's label) is the same
_CELL_SHIFT_SHARE = 0.4        # more paired rows than this with changed cells: the grid was read shifted
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
    raws = tuple(_printed_row(table, i) for i in indexes)
    text = " | ".join(_row_text(r) for r in raws)
    boxes = [table.row_boxes[i] for i in indexes if i < len(table.row_boxes)] or list(table.boxes[:1])
    return Element(kind=KIND_TABLE, text=text, key=_normalise(text), boxes=boxes, rows=len(rows),
                   cols=max((len(r) for r in rows), default=0), cells=rows, raw_cells=raws)


_VALUE_CELL_RE = re.compile(r"^[\sOoXx○●◯×✓✔\-–—*/.,:;()0-9%]*$")


def _is_header_name(text: str | None) -> bool:
    """A cell that names a column ("Pb", "铅 (Pb)", "Screen size"), not a value
    printed under one ("O", "X", "—", "43")."""
    text = " ".join((text or "").split())
    return len(text) >= 2 and not _VALUE_CELL_RE.match(text)


def _printed_row(table: Element, i: int) -> tuple:
    """Row `i` as printed; the normalised row when no printed copy lines up."""
    if len(table.raw_cells) == len(table.cells) and i < len(table.raw_cells):
        return table.raw_cells[i]
    return table.cells[i] if i < len(table.cells) else ()


def _printed_row_label(table: Element, i: int, limit: int = 40) -> str:
    """What a reader calls the row: its first filled cell as printed - or, in
    a table whose first column is merged over several rows (the cell empty in
    every row but one), the row's own printed words."""
    row = _printed_row(table, i)
    first = (row[0] or "").strip() if row else ""
    text = first if first else _row_text(row)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
    units_a = [_text_units(_row_text(r)) for r in a.cells]
    units_b = [_text_units(_row_text(r)) for r in b.cells]
    rows_a = [i for i, u in enumerate(units_a) if u]
    rows_b = [j for j, u in enumerate(units_b) if u]
    if not rows_a or not rows_b:
        return []
    pairs = _pair_rows(a, b, rows_a, rows_b, units_a, units_b)
    shorter = min(len(rows_a), len(rows_b))
    if shorter >= 2 and len(pairs) < _ROWS_RELIABLE * shorter:
        return []  # the two grids were read too differently to compare row by row

    outside_a = exp_units - sum((units_a[i] for i in rows_a), Counter())
    outside_b = act_units - sum((units_b[j] for j in rows_b), Counter())
    free_a = [i for i in rows_a if i not in pairs]
    free_b = [j for j in rows_b if j not in set(pairs.values())]
    out: list[dict] = []

    def page_of(el: Element, row: int) -> int | None:
        return el.row_boxes[row][0] if row < len(el.row_boxes) else None

    for i in list(free_a):
        for j in free_b:
            if j + 1 in free_b and _similar(units_a[i], units_b[j] + units_b[j + 1]) >= _ROW_MERGE:
                if page_of(b, j) != page_of(b, j + 1):
                    # One row continued over a page break (its first cell left
                    # empty under the repeated header) is still one row.
                    free_a.remove(i)
                    free_b.remove(j)
                    free_b.remove(j + 1)
                    break
                label = _printed_row_label(a, i)
                out.append({
                    "type": "table-merge", "kind": KIND_TABLE,
                    "summary": f"Table merge issue — the “{label}” row is one row in Production and two rows in Staging.",
                    "detail": "", "exp": _row_element(a, [i]), "act": _row_element(b, [j, j + 1]),
                    "row_label": label, "column": "", "before": "one row", "after": "split into two rows",
                })
                free_a.remove(i)
                free_b.remove(j)
                free_b.remove(j + 1)
                break
    for j in list(free_b):
        for i in free_a:
            if i + 1 in free_a and _similar(units_a[i] + units_a[i + 1], units_b[j]) >= _ROW_MERGE:
                label, label_next = _printed_row_label(a, i), _printed_row_label(a, i + 1)
                out.append({
                    "type": "table-merge", "kind": KIND_TABLE,
                    "summary": (f"Table merge issue — the “{label}” and “{label_next}” rows are two rows in "
                                f"Production and one row in Staging."),
                    "detail": "", "exp": _row_element(a, [i, i + 1]), "act": _row_element(b, [j]),
                    "row_label": f"{label} + {label_next}", "column": "", "before": "two rows", "after": "merged into one row",
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
            label, printed = _printed_row_label(a, i), " | ".join(c for c in _printed_row(a, i) if (c or "").strip())
            out.append({
                "type": "table-row-missing", "kind": KIND_TABLE,
                "summary": f"Table row missing in Staging — the “{label}” row (“{printed[:80]}”) is not in Staging's table.",
                "detail": "", "exp": _row_element(a, [i]), "act": b,
                "row_label": label, "column": "", "before": printed, "after": "",
            })
    for j in free_b:
        if not printed_outside(units_b[j], pool_a):
            label, printed = _printed_row_label(b, j), " | ".join(c for c in _printed_row(b, j) if (c or "").strip())
            out.append({
                "type": "table-row-added", "kind": KIND_TABLE,
                "summary": f"Table row added in Staging — the “{label}” row (“{printed[:80]}”) is not in Production's table.",
                "detail": "", "exp": a, "act": _row_element(b, [j]),
                "row_label": label, "column": "", "before": "", "after": printed,
            })

    cell_changes: list[tuple] = []
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
        if len(a.cells[i]) != len(b.cells[j]) or units_a[i] == units_b[j]:
            continue
        other_a = sum((units_a[x] for x in rows_a if x != i), Counter())
        other_b = sum((units_b[y] for y in rows_b if y != j), Counter())
        left_a, left_b = units_a[i] - units_b[j], units_b[j] - units_a[i]
        if (not left_a or _spilled(left_a, other_b)) and (not left_b or _spilled(left_b, other_a)):
            continue  # header or neighbouring text the extraction put in this row
        changed = [(k, x, y) for k, (x, y) in enumerate(zip(a.cells[i], b.cells[j]))
                   if _text_units(x or "") != _text_units(y or "")]
        if changed:
            cell_changes.append((i, j, changed))
    if merged_rows:
        # One issue for the table: the same merge decision printed on many rows
        # (a spanned Extension column, a status column split in two) reads as
        # one change, boxed on the whole table on both sides.
        merged = sum(1 for _, _, ca, cb in merged_rows if cb < ca)
        split = len(merged_rows) - merged
        rows_shown = ", ".join(f"“{_printed_row_label(a, i, 24)}”" for i, _, _, _ in merged_rows[:6]) + (" …" if len(merged_rows) > 6 else "")
        how = (f"{merged} row{'s' if merged != 1 else ''} with cells merged" if merged else "") + \
              (" and " if merged and split else "") + \
              (f"{split} row{'s' if split != 1 else ''} with cells split" if split else "")
        out.append({
            "type": "table-merge", "kind": KIND_TABLE,
            "summary": (f"Table merge issue — {how} in Staging compared with Production "
                        f"({'row' if len(merged_rows) == 1 else 'rows'} {rows_shown})."),
            "detail": "", "exp": a, "act": b,
            "row_label": ", ".join(_printed_row_label(a, i, 24) for i, _, _, _ in merged_rows[:6]),
            "column": "", "before": "cells as printed in Production",
            "after": how + " in Staging",
        })
    # A grid read one row out of step shows a "changed" cell in nearly every row.
    if len(pairs) >= 4 and len(cell_changes) > _CELL_SHIFT_SHARE * len(pairs):
        cell_changes = []
    def _column_plain(k: int, row: int | None = None) -> str:
        if a.headers and len(a.headers) == len(a.cells[0] if a.cells else ()) and k < len(a.headers):
            return " ".join(a.headers[k].split())
        # A header set over two or three rows ("Chemical Substance Table" above
        # "Pb | Hg | Cd …", or "产品中有害物质的名称及含量" above "铅 (Pb) | 汞 (Hg) …"):
        # the column is named by the nearest printed cell above the changed row.
        top = min(_HEADER_ROWS, len(a.cells) if row is None else row)
        for r in range(top - 1, -1, -1):
            cell = _printed_row(a, r)
            if k < len(cell) and _is_header_name(cell[k]):
                return " ".join(cell[k].split())
        return f"column {k + 1}"

    current_row: int | None = None

    def _column_name(k: int) -> str:
        plain = _column_plain(k, current_row)
        return f"“{plain}”" if not plain.startswith("column ") else plain

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
        is_marker = lambda w: w in _CELL_MARKERS  # noqa: E731
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
            return f"{where}: “{(x or '—')[:60]}” → “{(y or '—')[:60]}”"
        return f"{where}: " + "; ".join(pieces[:4]) + (" …" if len(pieces) > 4 else "")

    for i, j, changed in cell_changes:
        pa, pb = _printed_row(a, i), _printed_row(b, j)
        flat = lambda t: " ".join((t or "").split())  # noqa: E731
        printed = [(k, flat(pa[k] if k < len(pa) else x), flat(pb[k] if k < len(pb) else y)) for k, x, y in changed]
        current_row = i
        shown = "; ".join(_cell_change(k, x, y) for k, x, y in printed[:3])
        label = _printed_row_label(a, i)
        row_name = f"the “{label}” row" if label else f"row {i + 1}"
        changes = [{"row_label": label, "column": _column_plain(k, i), "before": x, "after": y}
                   for k, x, y in printed]
        out.append({
            "type": "table-cell", "kind": KIND_TABLE,
            "summary": f"Table cell changed in {row_name} — {shown}.",
            "detail": "", "exp": _row_element(a, [i]), "act": _row_element(b, [j]),
            "row_label": label, "column": changes[0]["column"], "before": changes[0]["before"], "after": changes[0]["after"],
            "changes": changes,
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
            "row_label": "", "column": "", "before": f"{cols_a} columns ({named(a)})",
            "after": f"{cols_b} columns ({named(b)})",
        })
    return out


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
                    if row_units and row_units <= act_units:
                        # Row's content found in Staging's tables, skip this false positive
                        continue
            else:  # dtype == "table-row-added"
                # Staging row added: check if it's in exp_units (entire Production pool)
                el = d.get("act")
                if el is not None:
                    row_units = _text_units(_row_text(el.cells[0]) if el.cells else "")
                    if row_units and row_units <= exp_units:
                        # Row's content found in Production's tables, skip this false positive
                        continue
        
        out.append(d)
    return out


def settle_content(diffs: list[dict], exp_words: str, act_words: str,
                   exp_el: Element | None = None, act_el: Element | None = None,
                   exp_units: Counter | None = None, act_units: Counter | None = None,
                   exp_keys: set[str] | None = None, act_keys: set[str] | None = None) -> list[dict]:
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
            words = act_words if kind == "missing" else exp_words
            units = act_units if kind == "missing" else exp_units
            if el is not None and _printed_in(el, words, units, exp_words if kind == "missing" else act_words):
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
        gone = [r for r in d.get("gone", []) if not _elsewhere(r, act_words, exp_words) and r not in (act_keys or ())
                and not _BARE_CALLOUT_RE.match(r)]
        extra = [r for r in d.get("extra", []) if not _elsewhere(r, exp_words, act_words) and r not in (exp_keys or ())
                 and not _BARE_CALLOUT_RE.match(r)]
        if not gone and not extra:
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
_BOLD_SIMILAR = 0.80    # Staging's strokes at least this share of Production's read as bold too


_BOLD_MATCH_OVERLAP = 0.5  # a Staging line must share this much of the Production line's words


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
                if not wb or not wp or wp / wb >= _BOLD_SIMILAR:
                    continue  # unmeasurable, or prints just as dark - not an issue either way
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


_ONE_SIDED_TYPES = {"missing", "added", "table-row-missing", "table-row-added"}
_CJK_SHARE = 0.9            # share of a CJK run's characters the other side's pages must print


def _solid_count(doc: fitz.Document, pages: list[int], text: str) -> int:
    """How many times `text` (spacing aside) is printed on exactly these pages."""
    solid = "".join(_TOKEN_RE.findall(_normalise(text or "")))
    if not solid:
        return 0
    stream = "".join("".join(_TOKEN_RE.findall(_normalise(doc[p].get_text("text"))))
                     for p in pages if 0 <= p < doc.page_count)
    return stream.count(solid)


def _printed_on(doc: fitz.Document, pages: list[int], text: str, heading: bool = False,
                at_least: int = 1) -> bool:
    """True when `text` is printed on `pages` (each ±1 page) of `doc`: the same
    words in order, spacing aside - or, for Chinese / Japanese text, whose
    extraction splits and reorders characters around table cells, nearly all
    of its characters. A heading counts anywhere in the document as its own line."""
    toks = _TOKEN_RE.findall(_normalise(text or ""))
    if not toks:
        return False
    if heading and _printed_as_line(doc, list(range(doc.page_count)), _normalise(text)):
        return True
    want = sorted({p + d for p in pages for d in (-1, 0, 1) if 0 <= p + d < doc.page_count})
    if not want:
        return False
    stream = "".join("".join(_TOKEN_RE.findall(_normalise(doc[p].get_text("text")))) for p in want)
    solid = "".join(toks)
    if stream.count(solid) >= max(1, at_least):
        return True
    cjk = [ch for ch in solid if _CJK_RE.match(ch)]
    if len(cjk) >= 6 and len(cjk) >= 0.5 * len(solid):
        have = Counter(stream)
        need = Counter(solid)
        found = sum(min(n, have[ch]) for ch, n in need.items())
        return found >= _CJK_SHARE * len(solid)
    return False


_PIC_MIN_SIDE = 8.0          # points: smaller marks are not pictures
_PIC_SAME = 0.8              # fingerprint similarity for "the same picture"
_PIC_ASPECT = 2.0            # width/height ratios further apart than this are different shapes
_PIC_SYMBOL_GAP = 2.0        # points of clear space splitting a drawn cluster into symbols
_PIC_LABEL_GAP = 12.0        # points below a symbol a one-word label may sit


def _symbol_parts(doc: fitz.Document, page_index: int, bbox: tuple) -> list[tuple]:
    """A drawn cluster split into the symbols printed side by side in it:
    Production draws the WEEE bin and the battery mark as one cluster, and one
    cluster compared against Staging's bin alone hid the battery. Shapes are
    grouped by clear horizontal space; a single group is the cluster itself."""
    rect = fitz.Rect(bbox)
    try:
        shapes = sorted((fitz.Rect(d["rect"]) for d in doc[page_index].get_drawings()
                         if rect.contains(fitz.Rect(d["rect"]) + (-0.5, -0.5, 0.5, 0.5))
                         and d["rect"].width < rect.width * 0.95), key=lambda r: r.x0)
    except Exception:
        return [bbox]
    groups: list[fitz.Rect] = []
    for r in shapes:
        if groups and r.x0 <= groups[-1].x1 + _PIC_SYMBOL_GAP:
            groups[-1] |= r
        else:
            groups.append(fitz.Rect(r))
    parts = [g for g in groups if g.width >= _PIC_MIN_SIDE and g.height >= _PIC_MIN_SIDE]
    return [tuple(g) for g in parts] if len(parts) >= 2 else [bbox]


_ART_GAP = 2.0               # points: drawn shapes this close belong to one piece of artwork
_ART_MIN_SIDE = 12.0         # a drawn symbol in a 5pt-type document is this small
_ART_MIN_SHAPES = 3          # fewer shapes is a rule, a box or a bullet - not artwork
_ART_TEXT_SHARE = 0.15       # words covering more of a cluster than this make it text, not artwork


def _drawn_artwork(doc: fitz.Document, page_index: int, word_rects: list) -> list[tuple]:
    """Artwork a page DRAWS - logos and symbols made of vector shapes (the
    ENERGY STAR logo, the WEEE bin and battery mark) - found from the shapes
    themselves, independent of the figure detector's size filters, which drop a
    21pt logo in a 5pt-type document. Shapes a couple of points apart form one
    piece; a piece mostly made of text (table rules around words) is not art;
    side-by-side symbols in one piece are split."""
    page = doc[page_index]
    area_page = max(1.0, page.rect.width * page.rect.height)
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    rects = []
    for d in drawings:
        r = fitz.Rect(d.get("rect") or (0, 0, 0, 0))
        if r.is_empty or min(r.width, r.height) < 0.6:
            continue  # a rule line
        if r.width * r.height > 0.25 * area_page:
            continue  # the page frame or a panel
        rects.append(r)
    rects.sort(key=lambda r: (r.y0, r.x0))
    clusters: list[list[fitz.Rect]] = []
    for r in rects:
        grown = r + (-_ART_GAP, -_ART_GAP, _ART_GAP, _ART_GAP)
        joined = [c for c in clusters if any(grown.intersects(x) for x in c)]
        if not joined:
            clusters.append([r])
            continue
        merged = [r] + [x for c in joined for x in c]
        clusters = [c for c in clusters if c not in joined] + [merged]
    out: list[tuple] = []
    for shapes in clusters:
        if len(shapes) < _ART_MIN_SHAPES:
            continue
        box = fitz.Rect(shapes[0])
        for s in shapes[1:]:
            box |= s
        if box.width < _ART_MIN_SIDE or box.height < _ART_MIN_SIDE:
            continue
        if box.width * box.height > 0.25 * area_page:
            continue
        covered = sum((box & w).get_area() for w in word_rects if box.intersects(w))
        if covered > _ART_TEXT_SHARE * box.get_area():
            continue  # text in a ruled box, not a drawing
        out += _symbol_parts(doc, page_index, tuple(box))
    return out


def _pictures(doc: fitz.Document, pages: list[int]) -> list[dict]:
    """Every picture printed on `pages`: embedded images and drawn artwork, drawn
    clusters split into their symbols, each with its fingerprint and the
    one-word label printed under it ("WEEE", "Battery")."""
    from pdfval import imagefp

    out: list[dict] = []
    for page_index in pages:
        boxes: list[tuple] = []
        try:
            for info in doc[page_index].get_image_info():
                r = fitz.Rect(info["bbox"]) & doc[page_index].rect
                if r.width >= _PIC_MIN_SIDE and r.height >= _PIC_MIN_SIDE:
                    boxes.append(tuple(r))
            words = doc[page_index].get_text("words")
            word_rects = [fitz.Rect(w[:4]) for w in words]
            boxes += [b for b in _drawn_artwork(doc, page_index, word_rects)
                      if not any(fitz.Rect(b).intersects(fitz.Rect(e)) for e in boxes)]
        except Exception:
            continue
        embedded_boxes = set(boxes[:embedded_count]) if (embedded_count := sum(
            1 for info in doc[page_index].get_image_info()
            if (fitz.Rect(info["bbox"]) & doc[page_index].rect).width >= _PIC_MIN_SIDE
            and (fitz.Rect(info["bbox"]) & doc[page_index].rect).height >= _PIC_MIN_SIDE)) else set()
        for bbox in boxes:
            r = fitz.Rect(bbox)
            if r.width < _PIC_MIN_SIDE or r.height < _PIC_MIN_SIDE:
                continue
            out.append({"page": page_index, "bbox": tuple(r), "drawn": bbox not in embedded_boxes,
                        "fp": imagefp.fingerprint(doc, page_index, tuple(r))})
    return out


_PIC_COVER_MIN = 16.0        # points: smaller marks are inline icons - icon_changes' business
_PIC_SAME_MIXED = 0.6        # similarity for the same symbol drawn on one side, embedded on the other
_PIC_LABEL_CARRIER = 0.4     # a Staging picture this alike may carry the label (checked by the words near it)
_PIC_LABEL_MAX = 20          # characters: a label under a symbol is a word or two, not a sentence


def _standalone_label(doc: fitz.Document, page_index: int, bbox: tuple) -> str:
    """The short line printed directly under a symbol ("WEEE", "Battery"), or "".
    Only a line of its own - a word or two, centred under the symbol - counts:
    the first word of the paragraph below is not a label."""
    r = fitz.Rect(bbox)
    try:
        data = doc[page_index].get_text("dict")
    except Exception:
        return ""
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            lb = fitz.Rect(line["bbox"])
            if (text and len(text) <= _PIC_LABEL_MAX and len(text.split()) <= 2
                    and r.y1 - 3 <= lb.y0 <= r.y1 + _PIC_LABEL_GAP
                    and r.x0 - 8 <= (lb.x0 + lb.x1) / 2 <= r.x1 + 8
                    # A label is about as wide as its symbol - a heading
                    # that happens to start under a logo is not its label.
                    and lb.width <= max(r.width * 1.6, 24)):
                return text
    return ""


def image_coverage_changes(chapters: list["Chapter"], expected: fitz.Document, actual: fitz.Document) -> None:
    """Every picture Production prints must be printed in Staging, and the
    reverse - checked once over both documents, on the pages themselves, by
    what the pictures look like, whatever section each landed in.

    A picture with no same-looking, same-shaped picture anywhere in the other
    document is missing (or added). A matched symbol whose own label line
    ("Battery") is not printed near its counterpart has lost its label. A figure
    issue these matches contradict - one drawn cluster of two symbols paired
    with one of them and called "oversized" - is dropped. Each new issue is
    filed under the chapter whose content sits on that page."""
    from pdfval import imagefp

    if not chapters:
        return
    # Every picture takes part in MATCHING - a 13pt logo in Production's 5pt
    # layout is the counterpart of Staging's 45pt one - but only pictures of a
    # reportable size are reported.
    all_prod = _pictures(expected, list(range(expected.page_count)))
    all_stage = _pictures(actual, list(range(actual.page_count)))

    def reportable(p: dict) -> bool:
        return min(p["bbox"][2] - p["bbox"][0], p["bbox"][3] - p["bbox"][1]) >= _PIC_COVER_MIN

    # Staging's cover and front matter are outside the comparison.
    compared_stage = {p for c in chapters for p in (c.act_pages or [])}
    prod = [p for p in all_prod if reportable(p)]
    stage = [p for p in all_stage if reportable(p) and p["page"] in compared_stage]

    def aspect_ok(a: dict, b: dict) -> bool:
        ra = (a["bbox"][2] - a["bbox"][0]) / max(1.0, a["bbox"][3] - a["bbox"][1])
        rb = (b["bbox"][2] - b["bbox"][0]) / max(1.0, b["bbox"][3] - b["bbox"][1])
        return max(ra, rb) / max(0.01, min(ra, rb)) <= _PIC_ASPECT

    def best(pic: dict, pool: list[dict], doc: fitz.Document, other_doc: fitz.Document) -> tuple[float, dict | None]:
        scored = [(imagefp.similarity(pic["fp"], q["fp"]), q) for q in pool if aspect_ok(pic, q)]
        if not scored:
            return 0.0, None
        top = max(s for s, _ in scored)
        # Look-alike marks (R Mark, G Mark) score the same: among the near-ties,
        # the one printed with the same label is the counterpart.
        ties = [q for s, q in scored if s >= top - 0.03]
        label = _standalone_label(doc, pic["page"], pic["bbox"])
        if label and len(ties) > 1:
            for q in ties:
                if label_near(other_doc, q, label):
                    return top, q
        return top, max(scored, key=lambda s: s[0])[1]

    def same_picture(sim: float, pic: dict, other: dict | None) -> bool:
        # A symbol one document DRAWS and the other EMBEDS renders differently
        # (anti-aliasing, stroke weight): a lower bar, shape still guarded.
        if other is None:
            return False
        return sim >= (_PIC_SAME_MIXED if pic.get("drawn") != other.get("drawn") else _PIC_SAME)

    def owner(side: str, page_index: int, bbox: tuple) -> "Chapter":
        """The chapter whose content is printed nearest this picture's page spot."""
        y = (bbox[1] + bbox[3]) / 2
        best_c, best_d = chapters[0], float("inf")
        for c in chapters:
            for el in (c.exp_elements if side == "prod" else c.act_elements):
                for p, b in el.boxes:
                    if p == page_index:
                        dist = 0.0 if b[1] <= y <= b[3] else min(abs(b[1] - y), abs(b[3] - y))
                        if dist < best_d:
                            best_c, best_d = c, dist
        return best_c

    def boxed(side: str, pic: dict) -> bool:
        r = fitz.Rect(pic["bbox"])
        for c in chapters:
            for d in c.differences:
                el = d.get("exp" if side == "prod" else "act")
                if el is None:
                    continue
                for p, b in el.boxes:
                    if p == pic["page"] and (r & fitz.Rect(b)).get_area() > 0.5 * max(1.0, r.get_area()):
                        return True
        return False

    def label_near(doc: fitz.Document, pic: dict, label: str) -> bool:
        r = fitz.Rect(pic["bbox"]) + (-80, -40, 80, 80)
        try:
            words = doc[pic["page"]].get_text("words", clip=r & doc[pic["page"]].rect)
        except Exception:
            return False
        near = set(_TOKEN_RE.findall(_normalise(" ".join(w[4] for w in words))))
        # By its words, not their order: Staging prints "G Mark" as "Mark G".
        return set(_TOKEN_RE.findall(_normalise(label))) <= near

    new: list[tuple["Chapter", dict]] = []
    matched_prod: list[dict] = []
    for pic in prod:
        sim, other = best(pic, all_stage, expected, actual)
        if same_picture(sim, pic, other):
            matched_prod.append(pic)
            label = _standalone_label(expected, pic["page"], pic["bbox"])
            # Missing only when NO same-looking Staging symbol carries it: the
            # WEEE bin is printed once per language, and the best-scoring copy
            # need not be the one under its label.
            # Loosely alike is enough to CARRY a label: Staging's embedded WEEE
            # bin under the "WEEE" label scores 0.44 against Production's drawn one.
            carriers = [q for q in all_stage if aspect_ok(pic, q)
                        and imagefp.similarity(pic["fp"], q["fp"]) >= _PIC_LABEL_CARRIER] if label else []
            if label and _TOKEN_RE.search(label) and not any(label_near(actual, q, label) for q in carriers):
                new.append((owner("prod", pic["page"], pic["bbox"]), {
                    "type": "figure-label-missing", "kind": KIND_FIGURE,
                    "summary": (f"Figure label missing in Staging — the “{label}” label printed under this "
                                f"symbol in Production is not printed with it in Staging."),
                    "detail": "",
                    "exp": Element(kind=KIND_FIGURE, text=label, boxes=[(pic["page"], pic["bbox"])]),
                    "act": Element(kind=KIND_FIGURE, text="", boxes=[(other["page"], other["bbox"])]),
                }))
        elif not boxed("prod", pic):
            label = _standalone_label(expected, pic["page"], pic["bbox"])
            new.append((owner("prod", pic["page"], pic["bbox"]), {
                "type": "figure-missing", "kind": KIND_FIGURE,
                "summary": ("Figure missing in Staging — "
                            + (f"the “{label}” symbol" if label else "a picture")
                            + " printed in Production is not printed anywhere in Staging."),
                "detail": "",
                "exp": Element(kind=KIND_FIGURE, text=label, boxes=[(pic["page"], pic["bbox"])]),
                "act": None,
            }))
    for pic in stage:
        sim, other = best(pic, all_prod, actual, expected)
        if same_picture(sim, pic, other) or boxed("stage", pic):
            continue
        label = _standalone_label(actual, pic["page"], pic["bbox"])
        new.append((owner("stage", pic["page"], pic["bbox"]), {
            "type": "figure-added", "kind": KIND_FIGURE,
            "summary": ("Figure only in Staging — "
                        + (f"the “{label}” symbol" if label else "a picture")
                        + " printed in Staging is not printed anywhere in Production."),
            "detail": "",
            "exp": None,
            "act": Element(kind=KIND_FIGURE, text=label, boxes=[(pic["page"], pic["bbox"])]),
        }))

    def contradicted(d: dict) -> bool:
        if d.get("type") not in ("figure-size", "figure-different", "figure-content"):
            return False
        el = d.get("exp")
        if el is None or not el.boxes:
            return False
        p, b = el.boxes[0]
        inside = [m for m in matched_prod
                  if m["page"] == p and fitz.Rect(b).contains(fitz.Rect(m["bbox"]) + (-1, -1, 1, 1))]
        return len(inside) >= 2

    touched: set[int] = set()
    for c in chapters:
        if any(contradicted(d) for d in c.differences):
            touched.add(id(c))
    for c, d in new:
        d["section"] = ""
        d["severity"] = severity_of(d)
        d["minor"] = d["severity"] > FAILING_SEVERITY
        touched.add(id(c))
    for c in chapters:
        if id(c) not in touched:
            continue
        kept = [d for d in c.differences if not contradicted(d)]
        kept += [d for owner_c, d in new if owner_c is c and len(kept) < MAX_ISSUES_PER_CHAPTER]
        old = c.differences
        position = {id(d): k for k, d in enumerate(kept)}
        for row in c.rows:
            row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
            row["differences"] = [d for d in row.get("differences", []) if id(d) in position]
        c.differences = kept


def _cjk_char_change(el: Element, other: fitz.Document, pages: list[int], lost: bool) -> dict | None:
    """For a CJK block the other side prints only NEARLY: the closest printed
    stretch of the other document's pages, and the characters that differ -
    None when it is printed exactly (or is not CJK text at all)."""
    solid = "".join(_TOKEN_RE.findall(_normalise(el.text or "")))
    cjk = [ch for ch in solid if _CJK_RE.match(ch)]
    if len(cjk) < 6 or len(cjk) < 0.5 * len(solid):
        return None
    want = sorted({p + d for p in pages for d in (-1, 0, 1) if 0 <= p + d < other.page_count})
    raw = "".join(other[p].get_text("text") for p in want)
    stream = "".join(_TOKEN_RE.findall(_normalise(raw)))
    if not stream or solid in stream:
        return None
    # The best window: where the block's opening and closing characters meet.
    n = len(solid)
    best, best_ratio = None, 0.0
    head = solid[:4]
    start = stream.find(head)
    while start != -1:
        window = stream[start:start + n + max(4, n // 10)]
        ratio = difflib.SequenceMatcher(a=solid, b=window, autojunk=False).ratio()
        if ratio > best_ratio:
            best, best_ratio = window, ratio
        start = stream.find(head, start + 1)
    if best is None or best_ratio < 0.85:
        return None
    ops = difflib.SequenceMatcher(a=solid, b=best, autojunk=False).get_opcodes()
    last_equal = max((i for i, op in enumerate(ops) if op[0] == "equal"), default=-1)
    changes = []
    for i, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag == "equal" or i > last_equal:
            continue  # the window's tail past the block is not part of it
        a_part, b_part = solid[max(0, i1 - 2):i2 + 2], best[max(0, j1 - 2):j2 + 2]
        changes.append((a_part, b_part))
    changed_chars = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops[:last_equal + 1] if tag != "equal")
    if not changes or len(changes) > 2 or changed_chars > 4:
        return None  # not a small edit - the block was only collected differently there
    exp_side, act_side = (changes[0][0], changes[0][1]) if lost else (changes[0][1], changes[0][0])
    return {
        "type": "text", "kind": KIND_TEXT,
        "summary": f"Text content changed — Production prints “{exp_side}”, Staging prints “{act_side}”.",
        "detail": "", "gone": [exp_side], "extra": [act_side],
        "exp": el if lost else None, "act": None if lost else el,
        "section": el.section,
    }


def _drop_printed_one_sided(chapter: "Chapter", expected: fitz.Document, actual: fitz.Document) -> None:
    """The last word on "in Production only" / "in Staging only": a block the
    other document prints on its own pages for this chapter (or the next page)
    is not absent - it was only collected into another block or topic there
    (a note Staging sets as a table row, a heading filed under the next
    chapter). Only what is truly not printed stays reported. The chapter's
    rows keep pointing at the right differences."""
    keep: list[dict] = []
    for d in chapter.differences:
        kind = d.get("type")
        if kind in _ONE_SIDED_TYPES and d.get("kind") != KIND_FIGURE:
            lost = kind in ("missing", "table-row-missing")
            el = d.get("exp") if lost else d.get("act")
            other, pages = (actual, chapter.act_pages) if lost else (expected, chapter.exp_pages)
            own, own_pages = (expected, chapter.exp_pages) if lost else (actual, chapter.act_pages)
            if el is not None and _printed_on(other, list(pages or []), el.text or "",
                                              heading=el.kind == KIND_HEADING,
                                              at_least=_solid_count(own, list(own_pages or []), el.text or "")):
                # Printed there - but when only nearly (Chinese / Japanese text
                # matched by its characters), say exactly which characters
                # differ: "含有基値" in Production, "含有基準値" in Staging.
                change = _cjk_char_change(el, other, list(pages or []), lost)
                if change is not None:
                    keep.append(change)
                continue
        keep.append(d)
    # Unchanged only when every difference is kept AS IS - a one-sided block
    # replaced by its precise change keeps the count but is still a change.
    if len(keep) == len(chapter.differences) and all(a is b for a, b in zip(keep, chapter.differences)):
        return
    old = chapter.differences
    position = {id(d): k for k, d in enumerate(keep)}
    for row in chapter.rows:
        row["refs"] = [position[id(old[r])] for r in row.get("refs", []) if r < len(old) and id(old[r]) in position]
        row["differences"] = [d for d in row.get("differences", []) if id(d) in position]
    chapter.differences = keep


_COVERAGE_MIN_TOKENS = 2    # shorter runs are the text checks' own business


def _topic_stream(elements: list[Element]) -> list[tuple[str, Element]]:
    """Everything a topic prints, word by word in reading order, with the
    element each word came from - paragraphs, list items, table cells, callout
    text. Callout labels and list markers are styling, not words."""
    out: list[tuple[str, Element]] = []
    for el in sorted(elements, key=lambda e: e.order):
        if el.kind == KIND_FIGURE:
            continue
        for token in _TOKEN_RE.findall(_normalise(el.text or "")):
            if _BARE_CALLOUT_RE.match(token):
                continue
            out.append((token, el))
    return out


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
    covered_text = " "
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
        covered_text += " " + " ".join(_TOKEN_RE.findall(_normalise(said))) + " "
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

    def printed_on_pages(run: list[str], kind: str, need: int = 1) -> bool:
        if kind not in printed_cache:
            printed_cache[kind] = (page_stream(actual, chapter.act_pages or []) if kind == "missing"
                                   else page_stream(expected, chapter.exp_pages or []))
        return printed_cache[kind].count(f" {' '.join(run)} ") >= need

    # Words a cell-by-cell table read scrambles: the same words inside ONE table
    # of the other side are that table's content, however the cells were read.
    tables_a = [Counter(_TOKEN_RE.findall(_normalise(e.text or ""))) for e in exp_topic if e.kind == KIND_TABLE]
    tables_b = [Counter(_TOKEN_RE.findall(_normalise(e.text or ""))) for e in act_topic if e.kind == KIND_TABLE]

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
            phrase = f" {' '.join(run)} "
            own_stream, own_solid = (stream_a, solid_a) if kind == "missing" else (stream_b, solid_b)
            need = max(1, own_stream.count(phrase))
            if other.count(phrase) >= need or phrase in covered_text:
                continue  # moved or wrapped (printed as often), or already reported
            if solid.count("".join(run)) >= max(1, own_solid.count("".join(run))):
                continue  # the same characters, only split or joined differently
            if any(all(t[w] >= n for w, n in Counter(run).items())
                   for t in (tables_b if kind == "missing" else tables_a)):
                continue  # inside one table on the other side, read cell by cell
            if printed_on_pages(run, kind, need):
                continue  # printed on the other document's pages for this chapter, or the next page
            el = source[start][1]
            printed = _printed_form(" ".join(run), [el.text or ""])
            shown = printed if len(printed) <= 150 else printed[:149] + "…"
            place = Element(kind=KIND_TEXT, text=shown, key=_normalise(printed),
                            boxes=_phrase_boxes(doc, el, printed), section=topic)
            covered |= set(run)
            covered_text += phrase
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


def _word_diff(before: str, after: str) -> list[dict]:
    """The wording change, word by word, for the report to mark up."""
    # List markers ("1.", "a.", "•") are the marker checks' business: one side
    # prints them in the text run and the other beside it, which is no change
    # in wording and must never be boxed as a missing word.
    a = [w for w in _WORD_RE.split(before.strip()) if w and not _LIST_MARKER_TOKEN_RE.fullmatch(w)]
    b = [w for w in _WORD_RE.split(after.strip()) if w and not _LIST_MARKER_TOKEN_RE.fullmatch(w)]
    ops: list[dict] = []
    # Curly and straight quotes are the same character to a reader: “display’s”
    # against "display's" is no wording change.
    fa, fb = [w.translate(_QUOTES) for w in a], [w.translate(_QUOTES) for w in b]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=fa, b=fb, autojunk=False).get_opcodes():
        # A run of punctuation or spacing alone ("." / "—") is never boxed:
        # boxes and insertion marks are for missing words.
        # "&" is a word ("og", "and"), not punctuation: "& (EU)" -> "og (EU)" is a change.
        if tag != "equal" and not _TOKEN_RE.search(" ".join(a[i1:i2] + b[j1:j2]).replace("&", "and")):
            tag = "equal"
        if tag == "equal":
            ops.append({"type": "equal", "text": " ".join(a[i1:i2] or b[j1:j2])})
        else:
            if i1 != i2:
                ops.append({"type": "del", "text": " ".join(a[i1:i2])})
            if j1 != j2:
                ops.append({"type": "ins", "text": " ".join(b[j1:j2])})
    return ops


# --- chapters --------------------------------------------------------------


def _skipped_heading(entry: TocEntry) -> bool:
    """A heading whose section is out of scope: a table of contents, a Q&A or
    FAQ section, or front matter the heading resolver already marked."""
    return entry.excluded or is_excluded_heading(entry.title or "")


def skip_spans(entries: list[TocEntry]) -> list[tuple]:
    """`(start, end, title)` for every table-of-contents / Q&A section at ANY
    level, running to the next heading at the same level or above that is not
    itself skipped - so a "Q&A" sub-section inside a chapter drops out of that
    chapter's comparison with everything under it, and the rest of the chapter
    is still compared."""
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
    if sum(1 for t in texts if looks_like_toc_listing(t)) >= _TOC_PAGE_MIN_ENTRIES:
        return "Table of contents"
    return None


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
    exp_skip = skip_spans(exp_entries)
    act_skip = skip_spans(act_entries)
    size_scale = (act_body / exp_body) if exp_body and act_body else 1.0
    if exp_body and act_body:
        biggest = max(exp_body, act_body)
        _FIGURE_MIN_BY_DOC[_doc_key(expected)] = _FIGURE_MIN_SIDE * exp_body / biggest
        _FIGURE_MIN_BY_DOC[_doc_key(actual)] = _FIGURE_MIN_SIDE * act_body / biggest

    # Topics: every heading both documents have, at every level, matched in
    # order. Content is compared only inside its own topic, straight across.
    topics: dict[str, dict] = {}
    exp_anchors: list[tuple] = []
    act_anchors: list[tuple] = []
    for k, m in enumerate(match_toc_entries(exp_entries, act_entries)):
        if m.expected_index is None or m.actual_index is None:
            continue
        e, a = exp_entries[m.expected_index], act_entries[m.actual_index]
        topic = f"t{k}"
        topics[topic] = {"title": e.title, "prod": (e.page, e.y), "stage": (a.page, a.y)}
        exp_anchors.append((e.page, e.y, topic))
        act_anchors.append((a.page, a.y, topic))
    exp_anchors.sort()
    act_anchors.sort()

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
        for topic in dict.fromkeys(e.section for e in exp_texts + act_texts):
            exp_topic = [e for e in exp_texts if e.section == topic]
            act_topic = [e for e in act_texts if e.section == topic]
            chapter.pairs += reconcile_layout(pair_elements(exp_topic, act_topic), exp_topic, act_topic)
            words[topic] = _topic_words(exp_topic, act_topic)
            act_by_topic[topic] = act_topic
            exp_by_topic[topic] = exp_topic
        repeated: dict[tuple, int] = {}
        for exp_group, act_group, note, container in chapter.pairs:
            exp_words, act_words, exp_units, act_units, exp_keys, act_keys = words[(exp_group or act_group)[0].section]
            diffs = (
                [] if note == "layout"
                else settle_content(
                    compare_group(exp_group, act_group, size_scale, moved=note == "moved"),
                    exp_words, act_words, merge_elements(exp_group), merge_elements(act_group),
                    exp_units, act_units, exp_keys, act_keys,
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
                diffs += table_row_changes(exp_table, act_table, exp_units, act_units)
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
                diffs += shading_changes(exp_group, act_group, expected, actual)
                if not any(e.kind == KIND_TABLE for e in exp_group + act_group):
                    # Table cells report their own space changes, cell by cell.
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
        # Figures, matched on their own terms (see `figure_changes`).
        for fig_row in figure_changes(exp_figs, act_figs, exp_texts, act_texts, expected, actual,
                                      exp_entries, act_entries, exp_pages, act_pages,
                                      exp_bands=_chapter_bands(expected, exp_pages, span.exp_span),
                                      act_bands=_chapter_bands(actual, act_pages, span.act_span),
                                      scale=size_scale,
                                      exp_section_at=_section_at(exp_anchors),
                                      act_section_at=_section_at(act_anchors)):
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
        ):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        for diff in table_header_repeats(actual, actual_path, act_pages) + list_structure_changes(exp_elements, act_elements, expected, actual):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        exp_links = collect_links(expected, exp_pages, span.exp_span, exp_entries)
        act_links = collect_links(actual, act_pages, span.act_span, act_entries)
        assign_sections(exp_links, exp_anchors)
        assign_sections(act_links, act_anchors)
        for diff in link_changes(exp_links, act_links, exp_elements, act_elements, expected, actual):
            if len(chapter.differences) < MAX_ISSUES_PER_CHAPTER:
                chapter.differences.append(diff)
        _drop_printed_one_sided(chapter, expected, actual)
        # Severity decides everything downstream: the order the report lists a
        # difference in, whether its box is red or orange, whether it fails.
        for diff in chapter.differences:
            diff["severity"] = severity_of(diff)
            diff["minor"] = diff["severity"] > FAILING_SEVERITY
            if "section" not in diff:
                placed = diff.get("exp") or diff.get("act")
                diff["section"] = placed.section if placed is not None else ""
        chapters.append(chapter)
    # Pictures, once over both whole documents: every picture Production prints
    # must be printed in Staging, whatever section either landed in.
    try:
        image_coverage_changes(chapters, expected, actual)
    except Exception:
        pass
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
    "text": 1, "missing": 1, "added": 1, "note-label": 1, "table-cell": 1,
    "numbering": 1, "list-marker-missing": 1, "list-marker-added": 1,
    # 2 - hyperlinks
    "link-missing": 2, "link-added": 2, "link-target": 2, "link-broken": 2,
    # 4 - images
    "figure-missing": 4, "figure-added": 4, "figure-elsewhere": 4, "figure-different": 4,
    "figure-content": 4, "figure-label-missing": 4, "figure-size": 4,
    "figure-spots": 4,
    # 5 - emphasis
    "bold-missing": 5, "bold-added": 5, "shading": 5, "marker-size": 5,
    "underline-missing": 5, "underline-added": 5,
    "icon-missing": 4, "icon-added": 4, "icon-colour": 4, "icon-changed": 4,
    "figure-text-missing": 2, "figure-text-added": 3, "figure-pixelated": 4,
    # 6 - how it is laid out, not what it says - shown, never fails the run
    "list-indent": 6, "figure-alignment": 6, "text-space": 6,
    # (table-header-fill is not reported: Staging's header colour is its design.
    # A background gone altogether is: table-fill-missing.)
    "table-fill-missing": 3,
    "table-shape": 6, "table-merge": 6, "table-header-repeat": 6,
    "table-row-missing": 6, "table-row-added": 6, "table-columns": 6, "table-as-text": 6,
}
SEVERITY_LABEL = {
    1: "Content & lists", 2: "Hyperlinks", 4: "Images", 5: "Bold", 6: "Layout",
}
# Only what is listed above is reported at all. Font size, text colour, italic,
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
    return SEVERITY_OF.get(diff.get("type"), 1)


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
             "differently (left instead of centre), with black spots, or with a label missing."},
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
]
CATEGORY_LABEL = {c["key"]: c["label"] for c in CATEGORIES}

_TYPE_CATEGORY = {
    "text": "content", "note-label": "content",
    "link-missing": "links", "link-added": "links", "link-target": "links", "link-broken": "links",
    "numbering": "lists", "list-marker-missing": "lists", "list-marker-added": "lists", "list-indent": "lists",
    "marker-size": "lists",
    "bold-missing": "bold", "bold-added": "bold", "shading": "formatting",
    "underline-missing": "formatting", "underline-added": "formatting",
    "icon-missing": "images", "icon-added": "images", "icon-colour": "images", "icon-changed": "images",
    "figure-text-missing": "images", "figure-text-added": "images", "figure-pixelated": "images",
    # Tables holds only how cells are merged: what a cell or row SAYS - a
    # changed word, a missing row, a lost space - is content like any other.
    "table-merge": "tables", "table-columns": "tables", "table-fill-missing": "tables",
    "table-shape": "content", "table-header-repeat": "content", "table-cell": "content",
    "table-row-missing": "content", "table-row-added": "content", "table-as-text": "content",
    "text-space": "content",
}


def category_of(diff: dict) -> str:
    """Which of the six categories a difference belongs to."""
    kind = diff.get("type") or ""
    if kind.startswith("figure-"):
        return "images"
    if kind in ("missing", "added"):
        # Content only one side has is filed by what it IS: a whole table
        # missing is a table issue, a whole picture an image issue.
        element = diff.get("kind")
        return "images" if element == KIND_FIGURE else "content"
    return _TYPE_CATEGORY.get(kind, "content")


# Formatting changes that are one styling decision when they repeat.
_REPEATABLE = {"size", "colour", "emphasis", "capitalisation", "punctuation", "word-order",
               "table-header-fill", "shading", "list-marker-added", "list-marker-missing",
               "icon-colour", "icon-changed", "icon-missing", "icon-added"}

_MINOR_TYPES = {
    "emphasis", "size", "colour", "figure-size", "figure-review", "moved", "table-extract",
    "punctuation", "capitalisation", "word-order", "marker-size",
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
            if b is not None:
                details["actual_page"] = b.page + 1
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
