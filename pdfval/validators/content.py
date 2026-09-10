"""2. Content validation: TOC-heading-anchored text diff (missing/added/changed text).

Sections are matched by heading title, not page number or page index, so a
document that gained or lost pages elsewhere doesn't throw off the comparison.
Diffing happens at sentence granularity (not raw PDF line) so that incidental
line-wrap differences between the two renderings don't split one logical
sentence into a false "changed" diff, and each reported item is a whole
sentence rather than a fragment. Falls back to a whole-document diff when
either PDF has no table of contents.

Each Missing/Added/Changed text issue also gets a genuine screenshot of the
relevant PDF page from both documents (prod/stage), with a red box drawn
around the exact text location, so the diff can be visually verified against
the real page layout.
"""
from __future__ import annotations

import difflib
import itertools
import re
from collections import Counter

import fitz

from pdfval import i18n
from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.toc import (
    extract_section_blocks,
    filter_toc_page_blocks,
    get_toc_entries,
    is_running_header_footer,
    looks_like_page_number,
    looks_like_toc_listing,
    match_toc_entries,
    merge_bare_marker_blocks,
    merge_wrapped_sentence_blocks,
    normalize_block_text,
    normalize_title,
)

MAX_DIFF_LINES = 40  # cap per-issue sentence dump so the report stays readable
MAX_SCREENSHOTS = 300  # cap total screenshot pairs so a huge document doesn't stall the report

# Sentence segmentation is delegated to `i18n.split_sentences`, which handles
# Western terminators (". ! ?" followed by a capitalised next sentence), CJK /
# Indic / Arabic terminators that run sentences together with no space, and the
# "1." / "2." list-marker exception. See pdfval/i18n.py.

# Tokens for word-level diffing: a single CJK/Kana/Hangul character (so a
# one-character edit in spaceless script shows as a one-character diff, not a
# whole-sentence swap), else a run of word characters, a single punctuation
# character, or a run of whitespace - kept separate so re-joining is exact.
_TOKEN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
    r"|\w+|[^\w\s]|\s+"
)

# A replaced sentence pair is only shown as an inline word-diff (like
# Diffchecker's rich-text view) when the two sentences are similar enough for
# that to be meaningful; otherwise they're too different and get the plain
# whole-sentence red/green treatment instead.
WORD_DIFF_SIMILARITY_THRESHOLD = 0.35

# When one word is replaced by a SIMILAR word, the diff drops to the letter:
# "colo[-u-]r", "bri[-g-]htness", not a whole word struck out and a whole word
# underlined. Below this character-similarity the two words are unrelated (a
# real word swap, "level" -> "settings") and the whole-word treatment is right.
LETTER_DIFF_MIN_RATIO = 0.5
LETTER_DIFF_MIN_LEN = 3  # shorter than this, a letter diff reads as noise

# A bullet glyph or inline numbered/lettered marker ("1.", "2)", "a.") sometimes
# gets glued to the following list item's text within one extracted PDF block,
# while the exact same content in the other document lands in a separate block
# per item. Splitting on the marker itself (dropping the marker text, which is
# tracked separately via `_classify_marker` for the formatting check below)
# puts both sides at the same per-item granularity before diffing, regardless
# of how each source PDF happened to group them.
#
# Multi-letter roman numerals (ii., iii., iv., ... xx.) get their own
# sub-list marker in some documents' outline style - spelled out explicitly
# (not a general roman-numeral regex) to keep the pattern bounded and avoid
# any ambiguity with ordinary short words. A lone "i."/"I." is still handled
# by the `[a-zA-Z][.)]` branch below (equally plausible as a lettered-list
# marker), so it's deliberately not repeated here.
_ROMAN_NUMERAL_MARKERS = r"ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx"
_LIST_MARKER_SPLIT_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))(?:\d{1,2}[.)]|[a-zA-Z][.)]|(?:" + _ROMAN_NUMERAL_MARKERS + r")[.)]|"
    r"[\u2022\u25e6\u25aa\u25b8\u2023\u2043])(?=\s|$)",
    re.IGNORECASE,
)

# A block that's purely a run of bare callout numbers (e.g. "1 2 3 4 5 6" or
# "1. 2. 3. ...") is a numbered-diagram legend, not prose - it's checked by
# Image Validation's label check instead of being diffed as a sentence here.
# Checked token-by-token (not one combined regex): a single `^(?:\d{1,2}...)+$`
# over the whole block catastrophically backtracks on a long unbroken digit run
# (a legend with dozens of numbers, or digits glued to a non-matching tail).
_BARE_NUMBER_TOKEN_RE = re.compile(r"^\d{1,2}[.)]?$")


def _looks_like_bare_number_list(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 2:
        return False
    return all(_BARE_NUMBER_TOKEN_RE.match(t) for t in tokens)

# A block that's purely a run of short part/plate labels for a hardware
# diagram (e.g. "R1 R2 Rt L1 L2 Lt") is the same kind of diagram legend as
# `_looks_like_bare_number_list` above, just alphanumeric instead of numeric - not
# prose, so it's excluded here too (Image Validation's label check is the
# right place for it, not Content Validation). Checked token-by-token (not one
# combined regex) to avoid catastrophic backtracking on long paragraphs.
_PART_LABEL_TOKEN_RE = re.compile(r"^[A-Za-z]{1,3}\d{0,2}$")

# A leading parenthetical qualifier - "(Applicable for models with Hotkey Puck
# G3) Apart from the control keys..." - is put on its own line by one document
# and glued to the following sentence by the other. Splitting it off (when it
# carries no sentence-ending punctuation of its own and the next sentence
# starts capitalised) puts both sides at the same granularity, so only the
# real wording difference in the sentence that follows gets reported.
_LEADING_PARENTHETICAL_RE = re.compile(r"^\(([^()]{2,110})\)\s+(?=[A-Z•\"'])")

# Words that routinely appear inside a plate/panel diagram legend alongside the
# short part labels (e.g. "L2 R2 Lt Rt top plate", "1 2 top plate") - allowed
# as filler tokens so the whole legend is still recognized as a diagram legend
# rather than leaking into the prose diff.
_DIAGRAM_LEGEND_WORDS = {"top", "bottom", "left", "right", "front", "rear", "plate", "plates", "and"}


def _looks_like_bare_part_labels(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return False
    label_like = [
        t for t in tokens
        if _PART_LABEL_TOKEN_RE.match(t) or t.isdigit() or t.lower() in _DIAGRAM_LEGEND_WORDS
    ]
    if len(label_like) != len(tokens):
        return False
    part_labels = [t for t in tokens if _PART_LABEL_TOKEN_RE.match(t) and not t.lower() in _DIAGRAM_LEGEND_WORDS]
    # A single lone token counts only when it's unmistakably a part label
    # (a letter glued to a digit, e.g. "L2"); "Rt"/"No"/"PC" on their own are
    # too ambiguous to drop.
    if len(tokens) == 1:
        return bool(re.match(r"^[A-Za-z]{1,2}\d{1,2}$", tokens[0]))
    has_digit = any(t.isdigit() for t in tokens)
    return len(part_labels) >= 1 or has_digit


# A spec-sheet value run ("10-90% 0-3000m 0-40°C"), a menu-path legend
# ("Custom Key > IR Channel Setting >"), a model-code list ("SW272 SW242 or
# or") or a Title-Case figure-label run ("How-to video Display QuicKit") is
# diagram / table / caption furniture, not flowing prose - Image and Table
# Validation own it. Diffing it here as a "sentence" produces noise like a
# lone "Rt" or a stray environmental-range row showing as "missing text".
_SPEC_VALUE_RE = re.compile(r"^[<>~+-]?\d[\d.,:/°%-]*(?:m|mm|cm|kg|g|hz|khz|w|v|a|c|f|k|db|ms|s|bit|p)?$", re.IGNORECASE)
_MENU_NAV_WORDS = {"or", "and", ">", "»", "→", "-", "/"}


def _looks_like_non_prose(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 2:
        return False
    # A stray bullet glyph from the NEXT list item sometimes rides along at
    # the end of this block's extracted text (a rendering/grouping artifact,
    # not part of this sentence) - strip it before checking for real
    # sentence-ending punctuation, or a genuine sentence like "...for 10
    # seconds. \u2022" reads as ending in "\u2022" instead of ".", defeating this
    # check and letting it fall through to the token-ratio heuristic below.
    trailing_clean = text.rstrip().rstrip("\u2022\u25e6\u25aa\u25b8\u2023\u2043").rstrip()
    has_sentence_punct = trailing_clean[-1:] in ".!?" and any(c.islower() for c in text)
    if has_sentence_punct and len(tokens) >= 5:
        return False  # a real sentence - leave it
    # A menu path drawn beside a screenshot: "A > B > C" with no verb.
    if ">" in tokens and not has_sentence_punct:
        return True
    non_prose = 0
    for t in tokens:
        core = t.strip(".,:;()[]•-")
        if (
            not core
            or core.lower() in _MENU_NAV_WORDS
            or _SPEC_VALUE_RE.match(core)
            or _PART_LABEL_TOKEN_RE.match(core)
            or (core.isupper() and any(ch.isdigit() for ch in core))  # model codes SW272
        ):
            non_prose += 1
    return non_prose / len(tokens) >= 0.6

# A callout box (TIP/NOTE/WARNING/CAUTION/etc.) is sometimes rendered with an
# icon only in one document and a literal text label in the other (e.g. "TIP:
# Applicable for models with Hotkey Puck G3." vs an icon + "Applicable for
# models with Hotkey Puck G3."). The label is a callout-style artifact, not
# real content, so it's stripped before diffing rather than causing a false
# "missing/changed text" report over an otherwise-identical sentence. When
# BOTH sides carry an explicit text label (not just one side's icon), the
# label itself is still compared separately (see `_check_callout_label`) so a
# genuine label swap (e.g. Production says NOTE, Staging says WARNING for the
# same sentence) is still reported, just as its own distinct issue.
# The label set spans the languages a hardware manual is commonly localised
# into (see `i18n._CALLOUT_WORDS`): EN/DE/FR/ES/PT/IT/NL/Nordic/JA/ZH/KO.
_match_callout_label = i18n.match_callout_label
_strip_callout_label = i18n.strip_callout_label

# A cross-reference page number ("... on page 11", "see page 42") is expected
# to differ between two documents whose page counts/layout differ - it's not
# real content, so it's stripped before diffing rather than causing a false
# "changed text" report over a sentence that otherwise reads identically. Two
# separate patterns: one strips "on page 11" wherever it appears (mid- or
# end-of-sentence) since the digit makes it unambiguous; the other strips a
# DANGLING "on page" with no number attached (which happens when a hyperlinked
# page number lands in a separate text block from the words "on page" that
# precede it, split by the link's own styling run) but only right before
# sentence-ending punctuation/end-of-string, since without a digit it's only
# safe to assume that meaning at a sentence boundary.
_strip_page_references = i18n.strip_page_references

# A character that means the text layer could not be decoded to real Unicode:
#   U+FFFD replacement char, U+FFFE/FFFF and per-plane noncharacters,
#   C0 control characters (not tab / newline / CR).
# The Private Use Area is deliberately NOT here - manuals legitimately map
# callout / UI glyphs into it; a PUA char is only an issue when it faces real
# text on the other side, which `_sentence_encoding_regressions` checks instead.
_ENCODING_ARTIFACT_RE = re.compile(
    r"[\ufffd\ufffe\uffff\U0001fffe\U0001ffff\U0002fffe\U0002ffff\U0010fffe\U0010ffff]"
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f]"
)
# PyMuPDF emits this literal token for a glyph with no ToUnicode mapping.
_CID_TOKEN_RE = re.compile(r"\(cid:\s?\d+\)")
_PUA_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")


def validate_content(
    expected: fitz.Document,
    actual: fitz.Document,
    expected_path: str | None = None,
    actual_path: str | None = None,
    output_dir: str | None = None,
) -> tuple[CheckResult, CheckResult]:
    """Returns (content_result, encoding_result) - encoding is its own check so
    the report always shows a PASS/0-count row for it rather than folding a
    rare finding into Content Validation's own issue list, invisible whenever
    there happen to be none.
    """
    result = CheckResult(name="Content Validation")
    encoding_result = CheckResult(name="Encoding Validation")

    exp_entries = get_toc_entries(expected)
    act_entries = get_toc_entries(actual)
    counter = itertools.count(1)
    exp_tables = _TableBBoxCache(expected_path)
    act_tables = _TableBBoxCache(actual_path)
    exp_images = _ImageBBoxCache(expected)
    act_images = _ImageBBoxCache(actual)

    if exp_entries and act_entries:
        _validate_by_section(
            result, encoding_result, expected, actual, exp_entries, act_entries, output_dir, counter,
            exp_tables, act_tables, exp_images, act_images,
        )
    else:
        _validate_whole_document(
            result, encoding_result, expected, actual, output_dir, counter, exp_tables, act_tables,
            exp_images, act_images,
        )

    _check_callout_icons(result, expected, actual, output_dir, counter)

    return result, encoding_result


# A word split across a line-wrap keeps its hyphen once the newline collapses
# ("lint-\nfree" -> "lint-free", "specifica-" + "tions" -> "specifica-tions"),
# but the same word in a document that didn't wrap there reads with no hyphen
# ("lintfree", "specifications") - and a real hyphenated compound may itself be
# written either way between the two documents. Neither the wrap point nor the
# hyphen style is real content, so sentences are compared with every
# letter/digit-adjacent hyphen (and any space the wrap left beside it) removed.
_COMPARE_HYPHEN_RE = re.compile(r"(\w)-\s*(?=\w)")

# A footnote-definition line often opens with the reference mark it resolves
# ("*: Charging via USB-C ..." / "†  See ...") in one document while the other
# document drops the mark and keeps just the text, or renders the note as a
# NOTE: callout. The leading mark is not content, so it is removed before the
# two sentences are compared. Kept deliberately tight: one or two mark glyphs
# (* † ‡ § ¶ or a parenthesised one), an optional : . ), then whitespace and
# real text - so it never eats an inline "5 * 3" or a bare "*".
_FOOTNOTE_LEAD_RE = re.compile(
    r"^\s*(?:\(\s*[*†‡]\s*\)|[*†‡§¶]{1,2})\s*[:.)]?\s+(?=\S)"
)

# A cross-reference's trailing page number is stripped for comparison
# ("... see Foo on page 47." -> "... see Foo ."), which can leave the sentence
# ending " ." while the other document ends "." or has no terminator at all.
# The terminal period and any space before it are never content, so they are
# folded out of the comparison key.
_TRAILING_STOP_RE = re.compile(r"[.\s]+$")


def _cmp_key(text: str) -> str:
    # Fold full-width Latin/space and non-ASCII numerals so a document typeset
    # with full-width glyphs or Arabic-Indic digits still compares equal to one
    # that isn't, then drop wrap hyphens (see above).
    text = i18n.fold_cjk_spaces(i18n.fold_width(i18n.fold_digits(text)))
    text = _FOOTNOTE_LEAD_RE.sub("", text)
    text = _TRAILING_STOP_RE.sub("", text)
    return _COMPARE_HYPHEN_RE.sub(r"\1", text)


def _validate_by_section(
    result: CheckResult, encoding_result: CheckResult, expected, actual, exp_entries, act_entries, output_dir,
    counter, exp_tables, act_tables, exp_images, act_images,
) -> None:
    matches = match_toc_entries(exp_entries, act_entries)
    matched = [m for m in matches if m.expected_index is not None and m.actual_index is not None]

    # Bound each section only by the NEXT heading that exists on BOTH sides.
    # A heading bookmarked in only one document (e.g. a sub-heading like
    # "Landscape installation" that's its own TOC entry in Staging but just
    # plain text under "Installing shading hood" in Production) must not cut
    # a section short on the side that bookmarks it - otherwise content that
    # genuinely exists there gets compared against nothing and reported as
    # entirely missing/added.
    exp_matched_entries = [exp_entries[m.expected_index] for m in matched]
    act_matched_entries = [act_entries[m.actual_index] for m in matched]
    exp_sections = extract_section_blocks(expected, exp_matched_entries)
    act_sections = extract_section_blocks(actual, act_matched_entries)
    exp_heading_titles = {normalize_title(e.title) for e in exp_entries if e.title.strip()}
    act_heading_titles = {normalize_title(e.title) for e in act_entries if e.title.strip()}
    exp_styles = _StyleCache(expected)
    act_styles = _StyleCache(actual)
    exp_blobs = _PageBlobCache(expected)
    act_blobs = _PageBlobCache(actual)
    style_reported = [0]

    for idx, m in enumerate(matched):
        exp_entry = exp_entries[m.expected_index]
        act_entry = act_entries[m.actual_index]
        exp_blocks = _attach_styles(
            _exclude_image_label_blocks(_exclude_table_blocks(exp_sections[idx], exp_tables), exp_images), exp_styles
        )
        act_blocks = _attach_styles(
            _exclude_image_label_blocks(_exclude_table_blocks(act_sections[idx], act_tables), act_images), act_styles
        )
        exp_text = "\n".join(b["text"] for b in exp_blocks)
        act_text = "\n".join(b["text"] for b in act_blocks)
        _check_encoding(encoding_result, exp_entry.title, exp_entry.page, "expected", exp_text, act_text)
        _check_encoding(encoding_result, exp_entry.title, act_entry.page, "actual", act_text, exp_text)
        _diff_text(
            result,
            heading=exp_entry.title,
            expected_page=exp_entry.page,
            actual_page=act_entry.page,
            exp_records=_to_sentence_records(exp_blocks),
            act_records=_to_sentence_records(act_blocks),
            expected_doc=expected,
            actual_doc=actual,
            output_dir=output_dir,
            counter=counter,
            exp_heading_titles=exp_heading_titles,
            act_heading_titles=act_heading_titles,
            style_reported=style_reported,
            exp_blobs=exp_blobs,
            act_blobs=act_blobs,
            encoding_result=encoding_result,
        )


def _validate_whole_document(
    result: CheckResult, encoding_result: CheckResult, expected: fitz.Document, actual: fitz.Document, output_dir,
    counter, exp_tables, act_tables, exp_images, act_images,
) -> None:
    exp_blocks = _attach_styles(
        _exclude_image_label_blocks(_exclude_table_blocks(_document_blocks(expected), exp_tables), exp_images),
        _StyleCache(expected),
    )
    act_blocks = _attach_styles(
        _exclude_image_label_blocks(_exclude_table_blocks(_document_blocks(actual), act_tables), act_images),
        _StyleCache(actual),
    )
    exp_text = "\n".join(b["text"] for b in exp_blocks)
    act_text = "\n".join(b["text"] for b in act_blocks)
    _check_encoding(encoding_result, None, None, "expected", exp_text, act_text)
    _check_encoding(encoding_result, None, None, "actual", act_text, exp_text)
    _diff_text(
        result,
        heading=None,
        expected_page=None,
        actual_page=None,
        exp_records=_to_sentence_records(exp_blocks),
        act_records=_to_sentence_records(act_blocks),
        expected_doc=expected,
        actual_doc=actual,
        output_dir=output_dir,
        counter=counter,
        exp_heading_titles=set(),
        act_heading_titles=set(),
        style_reported=[0],
        encoding_result=encoding_result,
    )


class _TableBBoxCache:
    """Lazily loads and caches each page's table bboxes (via pdfplumber) so
    Content Validation can exclude table cells from its own text diffing -
    Table Validation already compares table structure/content properly, and
    re-diffing the same cells as loose sentences here can misalign them
    against an unrelated nearby paragraph purely because of how the two
    documents order table vs. prose blocks on the page.
    """

    def __init__(self, pdf_path: str | None):
        self._path = pdf_path
        self._cache: dict[int, list[tuple]] = {}

    def bboxes(self, page: int) -> list[tuple]:
        if not self._path:
            return []
        if page not in self._cache:
            try:
                from pdfval.extractor import get_all_detected_regions

                self._cache[page] = get_all_detected_regions(self._path, page)
            except Exception:
                self._cache[page] = []
        return self._cache[page]


def _in_any_table(bbox: tuple[float, float, float, float], table_bboxes: list[tuple], threshold: float = 0.5) -> bool:
    ax0, ay0, ax1, ay1 = bbox
    a_area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    for bx0, by0, bx1, by1 in table_bboxes:
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        if inter / a_area >= threshold:
            return True
    return False


def _exclude_table_blocks(blocks: list[dict], cache: "_TableBBoxCache") -> list[dict]:
    """MARKS blocks that fall inside a detected table region rather than
    removing them.

    Removing them outright hid ~a quarter of a hardware manual's text from the
    content diff on the assumption Table Validation would compare it - but
    pdfplumber detects a ruled Production table and misses its borderless
    Staging counterpart, so Table Validation could not see the difference
    either, and a genuinely dropped table row vanished from BOTH checks. The
    blocks stay in the diff now; a finding that lands mostly on table-region
    text is shown at review confidence (see `_diff_text`), so a real change in
    a table surfaces without cell-scatter noise failing the run.
    """
    for b in blocks:
        table_bboxes = cache.bboxes(b["page"])
        b["in_table"] = bool(table_bboxes and _in_any_table(b["bbox"], table_bboxes))
    return blocks


class _ImageBBoxCache:
    """Lazily loads and caches each page's figure (raster + vector-drawn)
    bboxes so Content Validation can recognize a short diagram callout label
    ("Release button") sitting right above/below/over a figure as an image
    caption, not prose - Image Validation is where a label like that belongs,
    and diffing it here as a sentence produces a false content mismatch
    whenever the two documents' diagrams position or extract it differently.
    """

    def __init__(self, doc: fitz.Document | None):
        self._doc = doc
        self._cache: dict[int, list[tuple]] = {}

    def bboxes(self, page: int) -> list[tuple]:
        if self._doc is None:
            return []
        if page not in self._cache:
            try:
                from pdfval.extractor import get_figures

                self._cache[page] = [f.bbox for f in get_figures(self._doc, page)]
            except Exception:
                self._cache[page] = []
        return self._cache[page]


# How close (points) a short caption must sit to a figure to count as its
# label - generous vertically (captions commonly sit in the gap directly
# above or below a diagram) and tighter horizontally (a caption belongs to
# the figure it's roughly aligned with, not one further across the page).
_LABEL_H_MARGIN = 40.0
_LABEL_V_MARGIN = 30.0
_LABEL_MAX_WORDS = 5

# A bulleted/numbered/lettered list item is never a diagram caption, even a
# short one - it's real content laid out beside a figure (a spec/feature
# list next to a photo is common), not a label belonging to that figure.
_LEADING_LIST_MARKER_RE = re.compile(
    r"^(?:\d{1,2}[.)]|[a-zA-Z][.)]|(?:" + _ROMAN_NUMERAL_MARKERS + r")[.)]|"
    r"[\u2022\u25e6\u25aa\u25b8\u2023\u2043])\s",
    re.IGNORECASE,
)


def _looks_like_caption_text(text: str) -> bool:
    """A short line with no sentence-ending punctuation (e.g. "Release
    button") reads as a diagram label, not prose - a real sentence either
    ends properly or runs past this word count.
    """
    tokens = text.split()
    if not tokens or len(tokens) > _LABEL_MAX_WORDS:
        return False
    if _LEADING_LIST_MARKER_RE.match(text):
        return False
    return not text.rstrip().endswith((".", "!", "?", ":", ";"))


def _near_any_figure(bbox: tuple, figures: list[tuple]) -> bool:
    bx0, by0, bx1, by1 = bbox
    for fx0, fy0, fx1, fy1 in figures:
        if bx0 >= fx0 - _LABEL_H_MARGIN and bx1 <= fx1 + _LABEL_H_MARGIN and by0 >= fy0 - _LABEL_V_MARGIN and by1 <= fy1 + _LABEL_V_MARGIN:
            return True
    return False


def _exclude_image_label_blocks(blocks: list[dict], cache: "_ImageBBoxCache") -> list[dict]:
    filtered = []
    for b in blocks:
        if _looks_like_caption_text(b["text"]):
            figures = cache.bboxes(b["page"])
            if figures and _near_any_figure(b["bbox"], figures):
                continue
        filtered.append(b)
    return filtered


def _document_blocks(doc: fitz.Document) -> list[dict]:
    blocks_out: list[dict] = []
    for i in range(doc.page_count):
        page_blocks = doc[i].get_text("blocks")
        toc_flags = filter_toc_page_blocks([b[4] for b in page_blocks])
        for block, is_toc in zip(page_blocks, toc_flags):
            x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
            if (
                text.strip()
                and not is_toc
                and not looks_like_page_number(text)
                and not is_running_header_footer(doc, i, (x0, y0, x1, y1), text)
            ):
                blocks_out.append({"text": normalize_block_text(text), "page": i, "bbox": (x0, y0, x1, y1)})
    # get_text("blocks") is not reliably in reading order; sort before merging
    # so a stray marker glyph isn't glued onto a distant unrelated block.
    blocks_out.sort(key=lambda b: (b["page"], round(b["bbox"][1]), round(b["bbox"][0])))
    return merge_bare_marker_blocks(merge_wrapped_sentence_blocks(blocks_out))


SIDE_LABELS = {"expected": "Production", "actual": "Staging"}
MAX_ENCODING_SAMPLES = 5  # snippets of surrounding text shown per encoding finding
ENCODING_CONTEXT_CHARS = 40  # characters of text kept either side of an artifact


def _encoding_artifacts(text: str) -> tuple[dict[str, int], list[str]]:
    """Every mis-encoded character in `text`, how many times each occurs, and a
    few snippets of the surrounding words. Without the count and the context
    the finding said only "a bad character exists somewhere in this section",
    which is not actionable.
    """
    counts: Counter[str] = Counter()
    samples: list[str] = []
    for i, ch in enumerate(text):
        if not _ENCODING_ARTIFACT_RE.match(ch):
            continue
        counts[ch] += 1
        if len(samples) < MAX_ENCODING_SAMPLES:
            start = max(0, i - ENCODING_CONTEXT_CHARS)
            end = min(len(text), i + ENCODING_CONTEXT_CHARS)
            snippet = " ".join(text[start:end].split())
            if snippet:
                samples.append(f"...{snippet}...")
    for m in _CID_TOKEN_RE.finditer(text):
        counts["(cid:N)"] += 1
        if len(samples) < MAX_ENCODING_SAMPLES:
            start = max(0, m.start() - ENCODING_CONTEXT_CHARS)
            end = min(len(text), m.end() + ENCODING_CONTEXT_CHARS)
            snippet = " ".join(text[start:end].split())
            if snippet:
                samples.append(f"...{snippet}...")
    return {(ch if ch == "(cid:N)" else repr(ch)): n for ch, n in counts.most_common()}, samples


def _check_encoding(
    result: CheckResult,
    heading: str | None,
    page: int | None,
    side: str,
    text: str,
    other_text: str = "",
) -> None:
    counts, samples = _encoding_artifacts(text)
    if not counts:
        return
    other_counts, _ = _encoding_artifacts(other_text)

    details: dict = {}
    if heading is not None:
        details["heading"] = heading
    details["side"] = SIDE_LABELS.get(side, side)
    details["occurrences"] = sum(counts.values())
    details["distinct_characters"] = len(counts)
    details["characters"] = [f"{ch} x{n}" for ch, n in counts.items()]
    # An artifact present in Staging but not in Production is a regression
    # introduced by the new export, as opposed to one both documents inherited
    # from the same source font - worth separating, since only the first is
    # something the reader has to act on.
    new_chars = [ch for ch in counts if ch not in other_counts]
    if new_chars:
        details["not_present_in_other_document"] = new_chars
    if samples:
        details["text"] = samples
    result.issues.append(
        Issue(
            severity="error" if new_chars else "warning",
            page=page,
            message="Possible text encoding issue",
            details=details,
        )
    )


def _artifact_set(text: str) -> set[str]:
    """The distinct un-decodable markers in `text`: replacement / control /
    noncharacter chars, unmapped Private-Use glyphs, and "(cid:N)" tokens."""
    out = {ch for ch in text if _ENCODING_ARTIFACT_RE.match(ch) or _PUA_RE.match(ch)}
    if _CID_TOKEN_RE.search(text):
        out.add("(cid:N)")
    return out


def _check_sentence_encoding(
    result: CheckResult,
    heading: str | None,
    expected_page: int | None,
    actual_page: int | None,
    exp_records: list[dict],
    act_records: list[dict],
    expected_doc: fitz.Document,
    actual_doc: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """The same sentence on both sides, but Staging's copy carries a character
    the text layer could not decode (U+FFFD, a control char, an unmapped
    Private-Use glyph, "(cid:NN)") that Production's clean copy does not - a
    real encoding regression in the new export, not a shared source artefact."""
    for exp_r, act_r in zip(exp_records, act_records):
        new = _artifact_set(act_r["text"]) - _artifact_set(exp_r["text"])
        if not new:
            continue
        details: dict = {}
        if heading is not None:
            details["heading"] = heading
            details["expected_page"] = expected_page + 1
            details["actual_page"] = actual_page + 1
        details["side"] = "Staging"
        details["characters"] = sorted(x if x == "(cid:N)" else repr(x) for x in new)
        details["text"] = [act_r["text"]]
        _attach_screenshots(
            details, expected_doc, actual_doc, output_dir, counter,
            [exp_r], [act_r], (exp_r["page"], None), (act_r["page"], None),
        )
        result.issues.append(
            Issue(severity="error", page=expected_page,
                  message="Text encoding regression in Staging", details=details)
        )


_STYLE_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
_BOLD_FLAG = 1 << 4
_ITALIC_FLAG = 1 << 1
MAX_STYLE_ISSUES = 60  # font findings are per-sentence; cap so they can't swamp the report

# A PostScript base font name commonly appends its weight/style to the family
# with a hyphen ("Roboto-Regular", "Arial-BoldMT", "Helvetica-Oblique") - one
# exporter's plain "Roboto" and another's "Roboto-Regular" are the SAME family,
# just named to a different convention, and bold/italic are already compared
# on their own via the span's bold/italic flags. Left unstripped, this alone
# was responsible for a large share of "Text formatting differs" findings that
# had no real formatting difference at all. Only strips after an explicit `-`
# or `,` delimiter, never words glued directly onto the family with no
# separator (e.g. "Roman" in "TimesNewRomanPSMT") - the risk of a false
# suffix-match mangling a genuinely different family name isn't worth it for a
# case that's rare in practice.
_FONT_STYLE_SUFFIX_RE = re.compile(
    r"[-,](?:Bold|Italic|Oblique|Regular|Medium|Light|Black|Thin|SemiBold|"
    r"ExtraBold|ExtraLight|DemiBold|Heavy|Condensed|Narrow|MT|PS)+$",
    re.IGNORECASE,
)


def _font_family(font: str) -> str:
    """A subset-embedded font carries a random six-letter prefix that differs
    between any two exports of the same document ("ABCDEF+Arial-Bold"), so it
    has to come off before two documents' fonts can be compared at all.
    """
    name = _STYLE_SUBSET_PREFIX_RE.sub("", font or "")
    name = name.split(",")[0].strip()
    stripped = _FONT_STYLE_SUFFIX_RE.sub("", name).strip()
    return stripped or name


class _StyleCache:
    """The dominant character style (font family, size, bold, italic) of each
    text block, looked up by the block's page and bbox.

    Formatting was previously compared only for list markers and callout
    labels - a sentence set in a different font, at a different size, or that
    lost its bold in Staging read as a perfect match, because only the
    characters were ever compared. This supplies the styling so it can be
    diffed too.
    """

    def __init__(self, doc: fitz.Document):
        self._doc = doc
        self._pages: dict[int, list[tuple[tuple, dict, float]]] = {}

    def _spans(self, page: int) -> list[tuple[tuple, dict, float]]:
        if page not in self._pages:
            spans: list[tuple[tuple, dict, float]] = []
            try:
                data = self._doc[page].get_text("dict")
            except Exception:
                data = {"blocks": []}
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        flags = span.get("flags", 0)
                        style = {
                            "font": _font_family(span.get("font", "")),
                            "size": round(float(span.get("size", 0)) * 2) / 2,
                            "bold": bool(flags & _BOLD_FLAG),
                            "italic": bool(flags & _ITALIC_FLAG),
                        }
                        spans.append((tuple(span.get("bbox", (0, 0, 0, 0))), style, len(text.strip())))
            self._pages[page] = spans
        return self._pages[page]

    def style_for(self, page: int, bbox: tuple) -> dict | None:
        """The style covering most of the block's characters. A heading run or
        a bolded lead-in inside a longer paragraph shouldn't decide the whole
        block's style, so spans are weighted by how much text they carry.
        """
        weights: Counter[tuple] = Counter()
        for span_bbox, style, length in self._spans(page):
            if _overlaps(span_bbox, bbox):
                weights[(style["font"], style["size"], style["bold"], style["italic"])] += length
        if not weights:
            return None
        font, size, bold, italic = weights.most_common(1)[0][0]
        return {"font": font, "size": size, "bold": bold, "italic": italic}

    def emphasis_for(self, page: int, bbox: tuple) -> tuple[float, float]:
        """(bold_fraction, italic_fraction) of the block's characters - so a
        partly-bold sentence (a bold key term or lead-in) is caught even when
        the block's DOMINANT style is regular. Used to flag emphasis that
        Production applies and Staging drops."""
        total = bold = italic = 0
        for span_bbox, style, length in self._spans(page):
            if _overlaps(span_bbox, bbox):
                total += length
                if style["bold"]:
                    bold += length
                if style["italic"]:
                    italic += length
        if not total:
            return (0.0, 0.0)
        return (bold / total, italic / total)


def _overlaps(a: tuple, b: tuple) -> bool:
    """True when span `a` sits (mostly) inside block `b`. Compared by the
    span's own centre so a span straddling a block edge by a hair still counts.
    """
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1


def _attach_styles(blocks: list[dict], cache: _StyleCache) -> list[dict]:
    return [
        {
            **b,
            "style": cache.style_for(b["page"], b["bbox"]),
            "emphasis": cache.emphasis_for(b["page"], b["bbox"]),
        }
        for b in blocks
    ]


def _describe_style(style: dict) -> str:
    parts = [style["font"] or "(unnamed font)", f"{style['size']}pt"]
    if style["bold"]:
        parts.append("bold")
    if style["italic"]:
        parts.append("italic")
    return " · ".join(parts)


def _style_differences(exp_style: dict, act_style: dict) -> list[str]:
    diffs = []
    if exp_style["font"] != act_style["font"]:
        diffs.append("font family")
    if exp_style["size"] != act_style["size"]:
        diffs.append("font size")
    if exp_style["bold"] != act_style["bold"]:
        diffs.append("bold")
    if exp_style["italic"] != act_style["italic"]:
        diffs.append("italic")
    return diffs


# Production emphasises at least this fraction of a block's characters, and
# Staging drops it by at least this much -> the bold/italic emphasis was lost.
# A fraction rather than the dominant flag, so a bold key term or lead-in that
# lost its weight is caught even inside an otherwise-regular paragraph.
_EMPHASIS_PRESENT = 0.20
_EMPHASIS_DROP = 0.20


def _check_emphasis_lost(
    result: CheckResult,
    heading: str | None,
    expected_page: int | None,
    actual_page: int | None,
    exp_records: list[dict],
    act_records: list[dict],
    reported: list[int],
    expected_doc: fitz.Document,
    actual_doc: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """A run of text that Production sets bold or italic reads as regular in
    Staging. Directional on purpose: emphasis Staging ADDS is not flagged (the
    user asked for "bold missing", not an exact both-ways match). One finding
    per block, not per sentence in it."""
    seen: set[tuple] = set()
    for exp_r, act_r in zip(exp_records, act_records):
        pe, ae = exp_r.get("emphasis"), act_r.get("emphasis")
        if not pe or not ae:
            continue
        key = (exp_r.get("page"), tuple(round(v) for v in (exp_r.get("bbox") or ())))
        if key in seen:
            continue
        lost = []
        if pe[0] >= _EMPHASIS_PRESENT and pe[0] - ae[0] >= _EMPHASIS_DROP:
            lost.append("bold")
        if pe[1] >= _EMPHASIS_PRESENT and pe[1] - ae[1] >= _EMPHASIS_DROP:
            lost.append("italic")
        if not lost:
            continue
        seen.add(key)
        if reported[0] >= MAX_STYLE_ISSUES:
            return
        reported[0] += 1
        details: dict = {}
        if heading is not None:
            details["heading"] = heading
            details["expected_page"] = expected_page + 1
            details["actual_page"] = actual_page + 1
        details["emphasis"] = " and ".join(lost)
        details["production_bold_pct"] = f"{pe[0] * 100:.0f}%"
        details["staging_bold_pct"] = f"{ae[0] * 100:.0f}%"
        details["production_italic_pct"] = f"{pe[1] * 100:.0f}%"
        details["staging_italic_pct"] = f"{ae[1] * 100:.0f}%"
        details["text"] = [exp_r["text"]]
        _attach_screenshots(
            details, expected_doc, actual_doc, output_dir, counter,
            [exp_r], [act_r], (exp_r["page"], None), (act_r["page"], None),
        )
        result.issues.append(
            Issue(severity="warning", page=expected_page, message="Bold or italic emphasis removed", details=details)
        )


def _check_text_style(
    result: CheckResult,
    heading: str | None,
    expected_page: int | None,
    actual_page: int | None,
    exp_records: list[dict],
    act_records: list[dict],
    reported: list[int],
    expected_doc: fitz.Document,
    actual_doc: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """For sentences whose text matched exactly, report a genuine styling
    change (font swapped, size changed, bold/italic lost or gained) as its own
    formatting issue.
    """
    for exp_r, act_r in zip(exp_records, act_records):
        exp_style, act_style = exp_r.get("style"), act_r.get("style")
        if not exp_style or not act_style:
            continue
        diffs = _style_differences(exp_style, act_style)
        # A font-family / font-size-only difference is almost always the whole
        # document rendered in a substitute face at a slightly different size
        # (a global export setting), not a per-sentence restyle - it was the
        # single largest source of noise in the report and the user asked for
        # it gone. Only a bold / italic change (or one alongside a font change)
        # is a real formatting difference worth a finding here; partial-run
        # emphasis loss is handled by `_check_emphasis_lost`.
        if not diffs or set(diffs) <= {"font family", "font size"}:
            continue
        if reported[0] >= MAX_STYLE_ISSUES:
            return
        reported[0] += 1
        details: dict = {}
        if heading is not None:
            details["heading"] = heading
            details["expected_page"] = expected_page + 1
            details["actual_page"] = actual_page + 1
        details["changed"] = ", ".join(diffs)
        details["expected_style"] = _describe_style(exp_style)
        details["actual_style"] = _describe_style(act_style)
        details["text"] = [exp_r["text"]]
        # Anchored on the SENTENCE's own page/bbox (`exp_r`/`act_r`), not the
        # section heading's page - a styling change is otherwise impossible to
        # verify visually, since nothing before this pointed at the actual
        # sentence at all. This was the single largest source of screenshot-
        # free findings in the report (every "Text formatting differs" issue).
        _attach_screenshots(
            details, expected_doc, actual_doc, output_dir, counter,
            [exp_r], [act_r], (exp_r["page"], None), (act_r["page"], None),
        )
        result.issues.append(
            Issue(severity="warning", page=expected_page, message="Text formatting differs", details=details)
        )


def _check_callout_labels(
    result: CheckResult,
    heading: str | None,
    expected_page: int | None,
    actual_page: int | None,
    exp_records: list[dict],
    act_records: list[dict],
) -> None:
    """For sentences whose text matched (equal), flag a genuine callout-type
    swap (e.g. Production says NOTE, Staging says WARNING) as its own issue.
    Only compared when BOTH sides carry an explicit text label - one side
    using an icon only (no label text) can't be reliably compared, so it's
    left alone rather than risk a false report.
    """
    for exp_r, act_r in zip(exp_records, act_records):
        exp_label, act_label = exp_r.get("callout"), act_r.get("callout")
        if exp_label and act_label and exp_label != act_label:
            details: dict = {}
            if heading is not None:
                details["heading"] = heading
                details["expected_page"] = expected_page + 1
                details["actual_page"] = actual_page + 1
            details["expected_label"] = exp_label
            details["actual_label"] = act_label
            details["text"] = [exp_r["text"]]
            result.issues.append(Issue(severity="warning", page=expected_page, message="Callout label differs", details=details))


def _check_marker_style(
    result: CheckResult,
    heading: str | None,
    expected_page: int | None,
    actual_page: int | None,
    exp_records: list[dict],
    act_records: list[dict],
) -> None:
    """A genuine list-style swap (Production numbers a list 1,2,3 but Staging
    uses letters a,b,c or a bullet for the same items) is a LAYOUT finding, and
    now belongs to Alignment Validation, which reports it as "List marker
    changed" alongside the indent and text-alignment checks it sits naturally
    with. Kept as a no-op so Content Validation reports content only.
    """
    return


_NEARBY_PAGE_SPILLOVER = 1  # pages either side of the section's own span to also scan
_NEARBY_MATCH_MIN_CHARS = 14  # a fragment shorter than this is too generic to trust


def _norm_blob(text: str) -> str:
    """Letters and digits only, lower-cased - survives any re-wrapping,
    re-spacing, sentence re-segmentation or prose/cell reflow between the two
    documents."""
    return re.sub(r"[^a-z0-9]+", "", i18n.fold_width(i18n.fold_digits(text)).lower())


class _PageBlobCache:
    """Normalised full-page text, memoised per page, plus a helper that returns
    one blob for a run of pages."""

    def __init__(self, doc: fitz.Document):
        self._doc = doc
        self._pages: dict[int, str] = {}

    def _page(self, index: int) -> str:
        if index not in self._pages:
            try:
                self._pages[index] = _norm_blob(self._doc[index].get_text())
            except Exception:
                self._pages[index] = ""
        return self._pages[index]

    def range_blob(self, first: int, last: int) -> str:
        first = max(0, first)
        last = min(self._doc.page_count - 1, last)
        return "".join(self._page(p) for p in range(first, last + 1))


def _record_page_span(records: list[dict], fallback_page: int | None) -> tuple[int, int]:
    pages = [r["page"] for r in records if isinstance(r.get("page"), int)]
    if not pages:
        base = fallback_page or 0
        return base, base
    return min(pages), max(pages)


def _present_near(text: str, blob: str) -> bool:
    """Is this sentence's text sitting, contiguously, somewhere in `blob`?

    `blob` is the OTHER document's text for the same run of pages this section
    occupies (plus a page of spillover). A whole sentence appearing there
    verbatim - even though the sentence-by-sentence diff didn't pair it - means
    the content is present and merely landed on an adjacent page, got glued to
    its neighbour, or was read as a table cell on one side and prose on the
    other. That is what a human reviewer checks before calling text missing,
    and it is not the same as the "is this wording anywhere in the whole
    document" test, which is too blunt for boilerplate repeated across
    unrelated sections.
    """
    key = _norm_blob(text)
    if len(key) < _NEARBY_MATCH_MIN_CHARS:
        return False
    return key in blob


def _diff_text(
    result: CheckResult,
    heading: str | None,
    expected_page: int | None,
    actual_page: int | None,
    exp_records: list[dict],
    act_records: list[dict],
    expected_doc: fitz.Document,
    actual_doc: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
    exp_heading_titles: set[str],
    act_heading_titles: set[str],
    style_reported: list[int],
    exp_blobs: "_PageBlobCache | None" = None,
    act_blobs: "_PageBlobCache | None" = None,
    encoding_result: "CheckResult | None" = None,
) -> None:
    # Diff on a hyphen-insensitive comparison key (see `_cmp_key`) so a word
    # that only wrapped differently between the two documents doesn't read as a
    # changed sentence; the opcodes are positional, so the original records are
    # still what gets sliced and reported.
    exp_keys = [_cmp_key(r["text"]) for r in exp_records]
    act_keys = [_cmp_key(r["text"]) for r in act_records]
    exp_line_set = set(exp_keys)
    act_line_set = set(act_keys)

    # The counterpart document's text over the pages this section spans, plus a
    # page either side - so a sentence that merely reflowed onto the next page
    # (which is fine, and which a reviewer would check for) isn't reported as
    # missing or added. Built lazily and only when there's a finding to test.
    exp_blobs = exp_blobs or _PageBlobCache(expected_doc)
    act_blobs = act_blobs or _PageBlobCache(actual_doc)
    exp_span = _record_page_span(exp_records, expected_page)
    act_span = _record_page_span(act_records, actual_page)
    _act_nearby: list[str] = []
    _exp_nearby: list[str] = []

    def act_nearby_blob() -> str:
        if not _act_nearby:
            _act_nearby.append(
                act_blobs.range_blob(
                    act_span[0] - _NEARBY_PAGE_SPILLOVER, act_span[1] + _NEARBY_PAGE_SPILLOVER
                )
            )
        return _act_nearby[0]

    def exp_nearby_blob() -> str:
        if not _exp_nearby:
            _exp_nearby.append(
                exp_blobs.range_blob(
                    exp_span[0] - _NEARBY_PAGE_SPILLOVER, exp_span[1] + _NEARBY_PAGE_SPILLOVER
                )
            )
        return _exp_nearby[0]

    matcher = difflib.SequenceMatcher(a=exp_keys, b=act_keys, autojunk=False)
    opcodes = matcher.get_opcodes()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            _check_callout_labels(
                result, heading, expected_page, actual_page, exp_records[i1:i2], act_records[j1:j2]
            )
            _check_marker_style(
                result, heading, expected_page, actual_page, exp_records[i1:i2], act_records[j1:j2]
            )
            _check_text_style(
                result, heading, expected_page, actual_page, exp_records[i1:i2], act_records[j1:j2],
                style_reported, expected_doc, actual_doc, output_dir, counter,
            )
            _check_emphasis_lost(
                result, heading, expected_page, actual_page, exp_records[i1:i2], act_records[j1:j2],
                style_reported, expected_doc, actual_doc, output_dir, counter,
            )
            _check_sentence_encoding(
                encoding_result if encoding_result is not None else result,
                heading, expected_page, actual_page, exp_records[i1:i2], act_records[j1:j2],
                expected_doc, actual_doc, output_dir, counter,
            )
            continue

        details: dict = {}
        if heading is not None:
            details["heading"] = heading
            details["expected_page"] = expected_page + 1
            details["actual_page"] = actual_page + 1

        # Text sitting inside a detected table region is kept in the diff (it is
        # no longer removed - see `_exclude_table_blocks`) but any finding on it
        # is advisory: Table Validation compares that content structurally, and
        # cell text scattered across the page can pair against a stray
        # neighbour here. A genuine change (a fixed typo, a reworded
        # description) still surfaces - just folded into review, never blocking.
        _span_records = exp_records[i1:i2] + act_records[j1:j2]
        _in_table = _span_records and sum(1 for r in _span_records if r.get("in_table")) * 2 >= len(_span_records)
        # A run where one side collapses to a bare fragment ("HDMI video cable
        # connection type." -> "HDMI") or reads as a value list rather than a
        # sentence is table/menu cell text the sentence splitter caught mid-row,
        # not prose - advisory, same as `_in_table`.
        _fragmentary = any(
            _looks_like_non_prose(r["text"]) for r in _span_records
        ) or _one_sided_stub(exp_records[i1:i2], act_records[j1:j2])
        if _in_table or _fragmentary:
            details["confidence"] = "review"
            details["note"] = (
                "this text is inside a table - Table Validation compares table content directly"
                if _in_table
                else "this looks like table / menu cell text caught mid-row, not flowing prose"
            )

        # A sentence that also appears verbatim elsewhere on the other side
        # isn't actually missing/added/changed - it just landed in a different
        # position within the section (e.g. a table's intro sentence printed
        # before the table in one document, after it in the other). A
        # sentence that's ALSO one of that side's own heading titles (e.g. a
        # parent heading reprinted where a sibling/child heading doesn't exist
        # on the other side, or a caption like "More about HB27" landing next
        # to a heading like "Landscape installation" only on one side) is
        # noise, not real prose. Deliberately does NOT check the rest of the
        # document for a coincidental match elsewhere (tried and reverted -
        # boilerplate/generic sentences repeated in unrelated sections caused
        # genuinely missing/added content under THIS heading to be silently
        # dropped just because the same wording also happened to exist
        # somewhere else entirely).
        exp_slice_records = [
            r for r in exp_records[i1:i2]
            if _cmp_key(r["text"]) not in act_line_set
            and normalize_title(r["text"]) not in exp_heading_titles
            and normalize_title(r["text"]) not in act_heading_titles
            and not _present_near(r["text"], act_nearby_blob())
        ]
        act_slice_records = [
            r for r in act_records[j1:j2]
            if _cmp_key(r["text"]) not in exp_line_set
            and normalize_title(r["text"]) not in act_heading_titles
            and normalize_title(r["text"]) not in exp_heading_titles
            and not _present_near(r["text"], exp_nearby_blob())
        ]

        if tag == "delete":
            if not exp_slice_records:
                continue
            details["expected"] = _cap([r["text"] for r in exp_slice_records])
            # Missing text: the sentence exists only in Production. Show the
            # Production location (its own text, boxed, with a little context).
            # The Staging side has NOTHING that corresponds to it - a "context"
            # crop of whatever sentences the diff happened to align either side
            # of the gap is a picture of a different, unrelated place, which is
            # worse than showing nothing.
            _attach_screenshots(
                details, expected_doc, actual_doc, output_dir, counter,
                None, None,
                _context_span(exp_records, i1, i2, expected_doc, expected_page),
                (None, None),
            )
            result.issues.append(Issue(severity="error", page=expected_page, message="Missing text", details=details))
        elif tag == "insert":
            if not act_slice_records:
                continue
            details["actual"] = _cap([r["text"] for r in act_slice_records])
            # Added text: the sentence exists only in Staging - show only that.
            _attach_screenshots(
                details, expected_doc, actual_doc, output_dir, counter,
                None, None,
                (None, None),
                _context_span(act_records, j1, j2, actual_doc, actual_page),
            )
            result.issues.append(Issue(severity="error", page=actual_page, message="Added text", details=details))
        elif tag == "replace":
            # Apply the same reordering / heading-title allowance to both
            # sides of a replace. When only one side has anything left after
            # filtering, this "change" is really just a one-sided add or
            # delete (the other side's text was a reordered duplicate or a
            # heading).
            if not exp_slice_records and not act_slice_records:
                # `_present_near` emptied both sides: the two sentences are
                # near-identical and each is still present in the other
                # document. Not a content edit - a trailing period, a stray
                # space, a list-marker glyph one document's extraction caught
                # as text and the other's didn't. Still a prod/stage
                # difference the reviewer asked to see, so it is shown at
                # review confidence (folded away, never fails the run) rather
                # than silently dropped.
                exp_lax = [
                    r for r in exp_records[i1:i2]
                    if normalize_title(r["text"]) not in exp_heading_titles
                    and normalize_title(r["text"]) not in act_heading_titles
                ]
                act_lax = [
                    r for r in act_records[j1:j2]
                    if normalize_title(r["text"]) not in act_heading_titles
                    and normalize_title(r["text"]) not in exp_heading_titles
                ]
                if exp_lax and act_lax:
                    wd = _word_diff_pairs([r["text"] for r in exp_lax], [r["text"] for r in act_lax])
                    if wd is not None:
                        details["word_diff"] = wd
                    else:
                        details["expected"] = _cap([r["text"] for r in exp_lax])
                        details["actual"] = _cap([r["text"] for r in act_lax])
                    details["confidence"] = "review"
                    _attach_screenshots(
                        details, expected_doc, actual_doc, output_dir, counter,
                        None, None,
                        _context_span(exp_records, i1, i2, expected_doc, expected_page),
                        _context_span(act_records, j1, j2, actual_doc, actual_page),
                    )
                    result.issues.append(
                        Issue(
                            severity="warning", page=expected_page,
                            message="Minor text difference", details=details,
                        )
                    )
                continue
            if not exp_slice_records:
                details["actual"] = _cap([r["text"] for r in act_slice_records])
                _attach_screenshots(
                    details, expected_doc, actual_doc, output_dir, counter,
                    None, None,
                    (None, None),
                    _context_span(act_records, j1, j2, actual_doc, actual_page),
                )
                result.issues.append(Issue(severity="error", page=actual_page, message="Added text", details=details))
                continue
            if not act_slice_records:
                details["expected"] = _cap([r["text"] for r in exp_slice_records])
                _attach_screenshots(
                    details, expected_doc, actual_doc, output_dir, counter,
                    None, None,
                    _context_span(exp_records, i1, i2, expected_doc, expected_page),
                    (None, None),
                )
                result.issues.append(Issue(severity="error", page=expected_page, message="Missing text", details=details))
                continue
            exp_slice = [r["text"] for r in exp_slice_records]
            act_slice = [r["text"] for r in act_slice_records]
            word_diff = _word_diff_pairs(exp_slice, act_slice)
            if word_diff is not None:
                details["word_diff"] = word_diff
            else:
                details["expected"] = _cap(exp_slice)
                details["actual"] = _cap(act_slice)
            # Changed text: both sides carry the reworded sentence, so both
            # screenshots box a real, corresponding location.
            _attach_screenshots(
                details, expected_doc, actual_doc, output_dir, counter,
                None, None,
                _context_span(exp_records, i1, i2, expected_doc, expected_page),
                _context_span(act_records, j1, j2, actual_doc, actual_page),
            )
            result.issues.append(Issue(severity="error", page=expected_page, message="Changed text", details=details))


def _one_sided_stub(exp_recs: list[dict], act_recs: list[dict]) -> bool:
    """True when one side of a replace is a 1-3 word fragment while the other
    carries a real sentence - the tell of a table cell ("Large") lined up
    against a run-together neighbour ("Large the PIP mode.")."""
    ew = sum(len(r["text"].split()) for r in exp_recs)
    aw = sum(len(r["text"].split()) for r in act_recs)
    if not ew or not aw:
        return False
    lo, hi = min(ew, aw), max(ew, aw)
    return lo <= 3 and hi >= lo * 3


def _attach_screenshots(
    details: dict,
    expected_doc: fitz.Document,
    actual_doc: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
    exp_slice: list[dict] | None,
    act_slice: list[dict] | None,
    exp_fallback: tuple[int | None, tuple | None],
    act_fallback: tuple[int | None, tuple | None],
) -> None:
    """Render a genuine, zoomed screenshot of the relevant page from each
    document, with a red box around the diffed text (or, on the side that has
    no diffed text of its own - a Missing/Added text finding - around the
    nearest neighboring sentence instead), so the issue can be verified
    against the real prod/stage page layout on BOTH sides. Uses
    `capture_region`, not `capture_page`, so the crop is zoomed in on the
    highlighted area rather than handing back a full, hard-to-read page.
    """
    if not output_dir:
        return
    seq = next(counter)
    if seq > MAX_SCREENSHOTS:
        return

    exp_page, exp_bbox = _slice_anchor(exp_slice, exp_fallback)
    act_page, act_bbox = _slice_anchor(act_slice, act_fallback)

    prod_path = screenshots.capture_region(expected_doc, exp_page, output_dir, f"content_{seq}_prod", exp_bbox)
    stage_path = screenshots.capture_region(actual_doc, act_page, output_dir, f"content_{seq}_stage", act_bbox)

    if prod_path:
        details["prod_screenshot"] = prod_path
    elif exp_page is None:
        details["prod_screenshot_note"] = (
            "this text is only in Staging - there is no corresponding place in Production to show"
        )
    if stage_path:
        details["stage_screenshot"] = stage_path
    elif act_page is None:
        details["stage_screenshot_note"] = (
            "this text is only in Production - there is no corresponding place in Staging to show"
        )


# ---- callout pictogram check ---------------------------------------------
# Production flags a NOTE / TIP / WARNING / CAUTION with a small FILLED,
# COLOURED badge to the left of the callout body. Staging's DITA build
# replaces that badge with a flat ~13pt black hairline glyph - or drops it -
# so the "this is a note" cue is visually lost even though the callout text
# is still there. Colour is the reliable tell: Production's badge always
# renders with a real coloured-pixel fraction (~0.38-0.46 here), Staging's
# flat glyph is ~0.0.
_CALLOUT_ICON_MIN_PT = 10.0  # Staging's flat callout glyphs render at ~13pt
_CALLOUT_ICON_MAX_PT = 34.0
_CALLOUT_ICON_GAP_PT = 48.0
_CALLOUT_BADGE_MIN_COLOUR = 0.18
_CALLOUT_GLYPH_MAX_COLOUR = 0.12
_CALLOUT_MATCH_MIN_OVERLAP = 0.7
_CALLOUT_LABEL_STRIP_RE = re.compile(
    r"^\s*(?:note|tip|warning|caution|important|danger|attention)\s*[:.–-]?\s*",
    re.IGNORECASE,
)


def _colour_fraction(doc: fitz.Document, page: int, rect) -> float:
    try:
        pix = doc[page].get_pixmap(matrix=fitz.Matrix(4, 4), clip=fitz.Rect(rect), alpha=False)
    except Exception:
        return 0.0
    s, step = pix.samples, pix.n
    total = pix.width * pix.height or 1
    if step < 3 or not s:
        return 0.0
    coloured = 0
    for k in range(0, len(s) - 2, step):
        r, g, b = s[k], s[k + 1], s[k + 2]
        if max(r, g, b) - min(r, g, b) > 28:
            coloured += 1
    return coloured / total


def _callout_key(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", _cmp_key(text).lower())[:16])


def _icon_images(page: "fitz.Page") -> list[tuple]:
    out = []
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return out
    for b in blocks:
        if b.get("type") != 1:
            continue
        ix0, iy0, ix1, iy1 = b["bbox"]
        if _CALLOUT_ICON_MIN_PT <= ix1 - ix0 <= _CALLOUT_ICON_MAX_PT and _CALLOUT_ICON_MIN_PT <= iy1 - iy0 <= _CALLOUT_ICON_MAX_PT:
            out.append((ix0, iy0, ix1, iy1))
    return out


def _icon_left_of(block_bbox: tuple, icons: list[tuple], max_gap: float = _CALLOUT_ICON_GAP_PT + 12) -> tuple | None:
    """The icon image sitting left of (and roughly aligned with the top of) a
    callout body block - the pictogram may hang off the label line, a separate
    block a little above the first bullet, so reach upward a bit. `max_gap` is
    widened for the Staging lookup (its legend/DITA layout spaces the icon
    further from the text) since a false "no icon" there is worse than a near
    miss.
    """
    bx0, by0, bx1, by1 = block_bbox
    for ix0, iy0, ix1, iy1 in icons:
        if ix1 <= bx0 + 12 and bx0 - ix0 <= max_gap and iy0 >= by0 - 46 and iy1 <= by1 + 12:
            return (ix0, iy0, ix1, iy1)
    return None


def _prod_callouts(doc: fitz.Document) -> list[dict]:
    """Every Production callout that carries a coloured pictogram badge, one
    entry per callout (its blocks merged), with the badge bbox + colour."""
    merged: dict[tuple, dict] = {}
    for pi in range(doc.page_count):
        page = doc[pi]
        icons = _icon_images(page)
        if not icons:
            continue
        try:
            blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        except Exception:
            continue
        for b in blocks:
            text = " ".join(sp["text"] for ln in b.get("lines", []) for sp in ln.get("spans", [])).strip()
            if len(text) < 12:
                continue
            icon = _icon_left_of(b["bbox"], icons)
            if icon is None:
                continue
            colour = _colour_fraction(doc, pi, icon)
            if colour < _CALLOUT_BADGE_MIN_COLOUR:
                continue
            gk = (pi, round(icon[1] / 6))
            e = merged.setdefault(gk, {
                "page": pi, "icon_bbox": icon, "colour": colour,
                "body_bbox": list(b["bbox"]), "text": "", "key": set(),
            })
            e["body_bbox"] = [
                min(e["body_bbox"][0], b["bbox"][0]), min(e["body_bbox"][1], b["bbox"][1]),
                max(e["body_bbox"][2], b["bbox"][2]), max(e["body_bbox"][3], b["bbox"][3]),
            ]
            clean = _CALLOUT_LABEL_STRIP_RE.sub("", text)
            e["text"] = (e["text"] + " " + clean).strip()[:240]
            e["key"] |= _callout_key(clean)
    return [e for e in merged.values() if len(e["key"]) >= 3]


def _callout_union_bbox(icon, body) -> tuple:
    if not icon:
        return tuple(body)
    return (min(icon[0], body[0]), min(icon[1], body[1]), max(icon[2], body[2]), max(icon[3], body[3]))


def _check_callout_icons(
    result: CheckResult,
    expected_doc: fitz.Document,
    actual_doc: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """Report a NOTE/TIP/WARNING/CAUTION whose coloured Production badge is
    missing or flattened to a colourless outline glyph in Staging.
    """
    prod = _prod_callouts(expected_doc)
    if not prod:
        return

    # Every Staging text block + that page's icon images, so the matching
    # callout can be located and its pictogram (if any) inspected.
    act_pages: list[list[dict]] = []
    act_icons: list[list[tuple]] = []
    for pi in range(actual_doc.page_count):
        page = actual_doc[pi]
        act_icons.append(_icon_images(page))
        rows = []
        try:
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                t = " ".join(sp["text"] for ln in b.get("lines", []) for sp in ln.get("spans", [])).strip()
                if len(t) < 12:
                    continue
                rows.append({"bbox": tuple(b["bbox"]), "key": _callout_key(_CALLOUT_LABEL_STRIP_RE.sub("", t))})
        except Exception:
            pass
        act_pages.append(rows)

    seen: set = set()
    for c in prod:
        prod_id = (c["page"], round(c["icon_bbox"][1] / 6))
        if prod_id in seen:
            continue
        best = None  # (overlap, page, bbox)
        for pi, rows in enumerate(act_pages):
            for r in rows:
                if not r["key"]:
                    continue
                ov = len(c["key"] & r["key"]) / len(c["key"])
                if ov >= _CALLOUT_MATCH_MIN_OVERLAP and (best is None or ov > best[0]
                        or (ov == best[0] and r["bbox"][1] < best[2][1])):
                    best = (ov, pi, r["bbox"])
        if best is None:
            continue
        _, spage, sbbox = best
        icon = _icon_left_of(sbbox, act_icons[spage], max_gap=102.0)
        if icon is not None and _colour_fraction(actual_doc, spage, icon) > _CALLOUT_GLYPH_MAX_COLOUR:
            continue  # Staging still shows a coloured badge - fine

        dedup_key = (spage, round(sbbox[1] / 20))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        seen.add(prod_id)

        details: dict = {
            "expected_page": c["page"] + 1,
            "actual_page": spage + 1,
            "text": [c["text"][:200]],
            "production": "coloured callout badge",
            "staging": "no callout icon" if icon is None else "flat outline glyph, no colour",
        }
        _attach_screenshots(
            details, expected_doc, actual_doc, output_dir, counter,
            None, None,
            (c["page"], _callout_union_bbox(c["icon_bbox"], c["body_bbox"])),
            (spage, _callout_union_bbox(icon, sbbox)),
        )
        result.issues.append(Issue(
            severity="warning", page=c["page"],
            message="Callout icon missing in Staging", details=details,
        ))


# How far past the last matched sentence to reach for the screenshot when the
# gap where the missing/added content belongs runs off the bottom (or starts
# above the top) of a page - enough to bring the figure or table that follows
# into frame.
_GAP_EXTEND_PT = 260.0


def _context_span(
    records: list[dict], lo: int, hi: int, doc: fitz.Document, fallback: int | None
) -> tuple[int | None, tuple[float, float, float, float] | None]:
    """(page, bbox) of the region this content occupies on THIS side, WITH the
    matched sentence immediately before and after it - so the same region is
    framed on both documents and the two screenshots are directly comparable.

    Boxing only the changed sentences (what this used to do) meant a
    Missing-text finding showed a tight crop of a diagram fragment on the
    Production side and, on the Staging side, an unrelated paragraph - the
    reader had two pictures of different things. `records[lo:hi]` is the
    changed run (empty, i.e. lo == hi, on the side that lost/gained it: then
    this is the gap where the content belongs); the neighbours pin it to a
    real, matchable place on the page.
    """
    span = records[lo:hi]
    before = records[lo - 1] if lo - 1 >= 0 else None
    after = records[hi] if hi < len(records) else None
    bits = [r for r in [before, *span, after] if r]
    if not bits:
        return fallback, None

    page = Counter(r["page"] for r in bits).most_common(1)[0][0]
    on_page = [r["bbox"] for r in bits if r["page"] == page]
    x0, y0 = min(b[0] for b in on_page), min(b[1] for b in on_page)
    x1, y1 = max(b[2] for b in on_page), max(b[3] for b in on_page)

    rect = _page_rect(doc, page)
    # The gap continues off the bottom of this page (the "after" sentence is on
    # a later page or there is none) - reach down the page so a following
    # figure or table is in the frame.
    if rect and before and before["page"] == page and (after is None or after["page"] != page):
        x0, x1 = rect[0] + 2, rect[2] - 2
        y1 = min(rect[3] - 2, y1 + _GAP_EXTEND_PT)
    # ...or starts above the top of this page.
    if rect and after and after["page"] == page and (before is None or before["page"] != page):
        x0, x1 = rect[0] + 2, rect[2] - 2
        y0 = max(rect[1] + 2, y0 - _GAP_EXTEND_PT)
    return page, (x0, y0, x1, y1)


def _page_rect(doc: fitz.Document, page: int) -> tuple[float, float, float, float] | None:
    try:
        r = doc[page].rect
        return (r.x0, r.y0, r.x1, r.y1)
    except Exception:
        return None


def _slice_anchor(
    records: list[dict] | None, fallback: tuple[int | None, tuple | None]
) -> tuple[int | None, tuple[float, float, float, float] | None]:
    """The (page, bbox) to highlight for one side of a finding: the union bbox
    of the actual diffed sentences when there are any, otherwise the
    context-region span passed in as `fallback` (see `_context_span`) so a
    side with nothing of its own to point at still frames the same region as
    the other document.
    """
    if not records:
        return fallback
    page = records[0]["page"]
    bboxes = [r["bbox"] for r in records if r["page"] == page]
    if not bboxes:
        return fallback
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return page, (x0, y0, x1, y1)


def _classify_marker(marker: str) -> str:
    if re.match(r"^\d{1,2}[.)]$", marker):
        return "number"
    if re.match(rf"^(?:{_ROMAN_NUMERAL_MARKERS})[.)]$", marker, re.IGNORECASE):
        return "roman"
    if re.match(r"^[a-zA-Z][.)]$", marker):
        return "letter"
    return "bullet"


def _to_sentence_records(blocks: list[dict]) -> list[dict]:
    """Split each block's text into whole sentences for diffing, keeping the
    originating block's page/bbox so a sentence can be located on the page.
    The block's own leading callout label (TIP/NOTE/...) and list marker
    style (number/letter/bullet), if any, are kept on the first resulting
    sentence as `callout`/`marker` so a genuine label or list-style swap
    between documents can still be reported even though the label/marker text
    itself is stripped from the sentence before content-diffing.
    """
    records: list[dict] = []
    for block in blocks:
        text = block["text"].strip()
        if not text:
            continue
        if _looks_like_bare_number_list(text):
            continue
        if _looks_like_bare_part_labels(text) or _looks_like_non_prose(text):
            continue
        callout = _match_callout_label(text)
        text = _strip_callout_label(text)

        markers = [m.group(0) for m in _LIST_MARKER_SPLIT_RE.finditer(text)]
        chunks_raw = _LIST_MARKER_SPLIT_RE.split(text)

        first_in_block = True
        for idx, chunk in enumerate(chunks_raw):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk = _strip_callout_label(chunk)
            if not chunk:
                continue
            marker_kind = _classify_marker(markers[idx - 1]) if 0 < idx <= len(markers) else None
            first_in_chunk = True
            pieces: list[str] = []
            pm = _LEADING_PARENTHETICAL_RE.match(chunk)
            if pm:
                pieces.append(chunk[: pm.end()].strip())
                chunk = chunk[pm.end():].strip()
            pieces.extend(i18n.split_sentences(chunk))
            for sentence in pieces:
                sentence = _strip_page_references(sentence).strip()
                if sentence:
                    records.append(
                        {
                            "text": sentence,
                            "page": block["page"],
                            "bbox": block["bbox"],
                            "callout": callout if first_in_block else None,
                            "marker": marker_kind if first_in_chunk else None,
                            "style": block.get("style"),
                            "emphasis": block.get("emphasis"),
                            "in_table": block.get("in_table", False),
                        }
                    )
                    first_in_block = False
                    first_in_chunk = False
    return _merge_punctuation_only_records(records)


def _merge_punctuation_only_records(records: list[dict]) -> list[dict]:
    """An inline circled-number reference marker (e.g. a step-callout glyph
    like "\u2776" embedded mid-sentence, "...lock into place (\u2776).") sometimes
    extracts as its own tiny block with no visible/extractable content other
    than the surrounding punctuation ("( )."), landing as a lone sentence with
    no letters or digits at all - while the other document keeps it attached
    to the end of the preceding sentence. Merging any such punctuation-only
    record onto the end of the previous one keeps both documents at the same
    granularity instead of reporting a spurious "changed text" over content
    that's otherwise identical.
    """
    merged: list[dict] = []
    for r in records:
        if merged and not any(ch.isalnum() for ch in r["text"]):
            prev = merged[-1]
            merged[-1] = {**prev, "text": f"{prev['text']} {r['text']}".strip()}
        else:
            merged.append(r)
    return merged


def _cap(sentences: list[str]) -> list[str]:
    if len(sentences) > MAX_DIFF_LINES:
        return sentences[:MAX_DIFF_LINES] + [f"... (+{len(sentences) - MAX_DIFF_LINES} more sentences)"]
    return sentences


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _word_diff_pairs(exp_slice: list[str], act_slice: list[str]) -> list[list[dict]] | None:
    """The word/letter-level rich-text view of a change - unchanged words plain,
    removed struck through, added underlined, and a changed word broken down to
    the letter.

    Sentences are aligned first (`difflib` on the two sentence lists), so this
    still produces a useful view when the two sides don't have the same number
    of sentences - a paragraph that gained or lost a sentence shows that
    sentence added/removed in full and word-diffs the ones that pair up, rather
    than falling back to two opaque red/green blocks. Returns None only when a
    paired sentence turns out to be a genuine wholesale rewrite (too little in
    common for a word diff to read as anything but noise).
    """
    matcher = difflib.SequenceMatcher(a=exp_slice, b=act_slice, autojunk=False)
    pairs: list[list[dict]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for s in exp_slice[i1:i2]:
                pairs.append([{"type": "equal", "text": s}])
        elif tag == "delete":
            for s in exp_slice[i1:i2]:
                pairs.append([{"type": "del", "text": s}])
        elif tag == "insert":
            for s in act_slice[j1:j2]:
                pairs.append([{"type": "ins", "text": s}])
        elif tag == "replace":
            exp_run, act_run = exp_slice[i1:i2], act_slice[j1:j2]
            # Pair the replaced sentences up as far as they go, word-diff each
            # pair, and show whatever is left over on one side as a pure
            # add/remove.
            for k in range(max(len(exp_run), len(act_run))):
                if k < len(exp_run) and k < len(act_run):
                    ops = _word_diff(exp_run[k], act_run[k])
                    total = sum(len(op["text"]) for op in ops) or 1
                    equal = sum(len(op["text"]) for op in ops if op["type"] == "equal")
                    if equal / total < WORD_DIFF_SIMILARITY_THRESHOLD:
                        return None
                    pairs.append(ops)
                elif k < len(exp_run):
                    pairs.append([{"type": "del", "text": exp_run[k]}])
                else:
                    pairs.append([{"type": "ins", "text": act_run[k]}])
    if not any(op["type"] != "equal" for ops in pairs for op in ops):
        return None
    if len(pairs) > MAX_DIFF_LINES:
        return pairs[:MAX_DIFF_LINES]
    return pairs


def _word_diff(exp_sentence: str, act_sentence: str) -> list[dict]:
    """Token-level diff of two sentences, dropping to LETTER level inside a word
    that was replaced by a similar word.

    Each op is {"type": equal|del|ins, "text": str} and, for the letter-level
    ops emitted inside a changed word, also "level": "char" so the report can
    render them tight against the surrounding letters instead of as spaced-out
    separate words.
    """
    exp_tokens = _tokenize(exp_sentence)
    act_tokens = _tokenize(act_sentence)
    matcher = difflib.SequenceMatcher(a=exp_tokens, b=act_tokens, autojunk=False)
    ops: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            _push_op(ops, "equal", "".join(exp_tokens[i1:i2]))
        elif tag == "delete":
            _push_op(ops, "del", "".join(exp_tokens[i1:i2]))
        elif tag == "insert":
            _push_op(ops, "ins", "".join(act_tokens[j1:j2]))
        elif tag == "replace":
            _emit_replace(ops, exp_tokens[i1:i2], act_tokens[j1:j2])
    return ops


def _push_op(ops: list[dict], typ: str, text: str, level: str | None = None) -> None:
    if not text:
        return
    if ops and ops[-1]["type"] == typ and ops[-1].get("level") == level:
        ops[-1]["text"] += text
    else:
        op = {"type": typ, "text": text}
        if level:
            op["level"] = level
        ops.append(op)


def _emit_replace(ops: list[dict], exp_tokens: list[str], act_tokens: list[str]) -> None:
    """A replaced run of tokens. When it is ONE word for ONE similar word, show
    which letters changed; otherwise strike the whole run and insert the new."""
    exp_text = "".join(exp_tokens)
    act_text = "".join(act_tokens)
    if (
        len(exp_tokens) == 1
        and len(act_tokens) == 1
        and len(exp_text) >= LETTER_DIFF_MIN_LEN
        and len(act_text) >= LETTER_DIFF_MIN_LEN
        and difflib.SequenceMatcher(a=exp_text, b=act_text, autojunk=False).ratio() >= LETTER_DIFF_MIN_RATIO
    ):
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=exp_text, b=act_text, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                _push_op(ops, "equal", exp_text[i1:i2], level="char")
            elif tag == "delete":
                _push_op(ops, "del", exp_text[i1:i2], level="char")
            elif tag == "insert":
                _push_op(ops, "ins", act_text[j1:j2], level="char")
            elif tag == "replace":
                _push_op(ops, "del", exp_text[i1:i2], level="char")
                _push_op(ops, "ins", act_text[j1:j2], level="char")
        return
    _push_op(ops, "del", exp_text)
    _push_op(ops, "ins", act_text)
