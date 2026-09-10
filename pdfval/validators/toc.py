"""TOC (bookmarks/outline) status, and heading-anchored section extraction.

Page numbers are expected to differ between the two documents (insertions/
deletions shift later pages), so headings are matched by title text only.
"""
from __future__ import annotations

import difflib
import re
import weakref
from collections import Counter
from dataclasses import dataclass

import fitz

from pdfval import i18n
from pdfval.models import CheckResult, Issue

TITLE_MATCH_THRESHOLD = 0.6  # minimum similarity to pair up two non-identical headings

# Sections that are a printed NAVIGATION listing - a table of contents, an
# index, a Q&A/FAQ topic-jump page - not prose. Their whole body is
# "heading ... page-number" entries; the embedded outline is compared directly
# by validate_toc / toc.html, so comparing this text line-by-line in the
# section browser just floods it with red. Matched by title in English and CJK.
_EXCLUDED_HEADING_RE = re.compile(
    r"\bq\s*&\s*a\b|\bfaq\b|\btable of contents\b|\bq\s*&\s*a\s*index\b"
    r"|目\s*录|索\s*引|问\s*答|常见问题",
    re.IGNORECASE,
)

# A printed "Table of Contents" listing page (dot-leader "Title .... N" lines)
# isn't itself bookmarked, so it would otherwise get swallowed into whichever
# heading happens to precede it. Its content is meta/pagination, not prose, and
# the embedded outline is already compared directly by validate_toc.
_TOC_DOT_LEADER_RE = re.compile(r"\.{4,}")


def is_excluded_heading(title: str) -> bool:
    return bool(_EXCLUDED_HEADING_RE.search(title))


def looks_like_toc_listing(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.lower().rstrip(":") == "table of contents":
        return True
    return bool(_TOC_DOT_LEADER_RE.search(stripped))


# A top-level TOC entry is sometimes styled with a solid rule/underline instead
# of a dot leader, which renders with no dots at all - just "Title<space>N".
# Undetected, a block like this reads as ordinary prose, gets swallowed into
# whichever heading's section happens to span that page, and is then diffed as
# real content: on one real manual this put a line from the printed table of
# contents into a "Missing text" finding, and pointed that finding's screenshot
# at the TOC page instead of the heading's own content. Only trusted on a page
# that ALSO contains at least one block matching the strict dot-leader test
# (see `filter_toc_page_blocks`), so it can't misfire on an ordinary page whose
# last line of prose happens to end in a number.
_BARE_TOC_ENTRY_RE = re.compile(r"^(.{8,}?)\s+(\d{1,4})$")


def _looks_like_bare_toc_entry(text: str) -> bool:
    stripped = text.strip()
    if "\n" in stripped or stripped.endswith((".", "!", "?", ":", ",", ";")):
        return False
    m = _BARE_TOC_ENTRY_RE.match(stripped)
    if not m:
        return False
    return not m.group(1).rstrip().endswith((".", "!", "?", ":", ";"))


# A printed "Q&A index" page (a topic-jump listing: mixed question/heading/
# page-number lines, not itself bookmarked in either document) reads nothing
# like a real dot-leader TOC page, so `looks_like_toc_listing` never fires on
# it - yet several of its lines DON'T match the strict bare "Title N" pattern
# either (a single block often merges multiple entries, e.g. "Safety
# precautions\n6\nHow to assemble your monitor hardware (for models with
# stand)\n19"), so `_looks_like_bare_toc_entry` misses those too, and the
# leftover fragments survive into whichever heading's section happens to span
# that page as garbled false content. The listing can also spill onto a
# CONTINUATION page with no repeated title of its own - so this is detected
# by SHAPE, not just the title line: every block on a real Q&A-index page is
# either a bare question ("...turn it on?"), a heading name, or a heading
# name + trailing bare page number - NONE end in ordinary sentence
# punctuation (".", "!", ":", ";", ",") the way real prose almost always
# does somewhere on the page, and at least one is a genuine question.
# Title of a Q&A / topic-jump index page, EN or CJK ("Q&A index", "问答索引").
_Q_AND_A_INDEX_TITLE_RE = re.compile(r"^\s*(?:q\s*&\s*a\s*index|问\s*答\s*索\s*引)\s*$", re.IGNORECASE)
# Sentence-ending punctuation that real prose has SOMEWHERE on a page - Latin
# and CJK. A colon is deliberately NOT here: a listing often opens with one
# ("Start with the topics that interest you:" / "从您感兴趣的主题开始:").
_PROSE_TERMINATORS = (".", "!", ";", ",", "。", "！", "；", "，")
_QUESTION_MARKS = ("?", "？")  # ASCII and full-width
# A listing line: something, then a bare page number, optionally repeated
# ("选择位置 25 开机和初始设置 30"). Works with or without a space before the
# number, so it fits CJK ("微调图像清晰度43") and Latin alike.
_LISTING_ENTRY_RE = re.compile(r"\S\s*\d{1,3}(?:\s|$)")


def _looks_like_qa_index_page(texts: list[str]) -> bool:
    stripped = [t.strip() for t in texts if t.strip()]
    if any(_Q_AND_A_INDEX_TITLE_RE.match(t) for t in stripped):
        return True
    if len(stripped) < 4:
        return False
    has_question = any(any(q in t for q in _QUESTION_MARKS) for t in stripped)
    entry_like = sum(1 for t in stripped if _LISTING_ENTRY_RE.search(t) or any(q in t for q in _QUESTION_MARKS))
    prose_like = any(t.rstrip().endswith(_PROSE_TERMINATORS) for t in stripped)
    # A topic-jump index: has at least one question, most lines are
    # entry/question shaped, and NO line reads as a finished prose sentence.
    return has_question and not prose_like and entry_like >= max(3, len(stripped) * 0.5)


def filter_toc_page_blocks(texts: list[str]) -> list[bool]:
    """Which of a PAGE's blocks (in extraction order) are TOC-listing meta
    content rather than prose - used instead of calling `looks_like_toc_listing`
    block-by-block so the loose "Title N" signal only ever activates on a page
    already positively identified as a TOC page by the strict dot-leader/
    "Table of Contents" heading signal.
    """
    if _looks_like_qa_index_page(texts):
        return [True] * len(texts)
    strict = [looks_like_toc_listing(t) for t in texts]
    if not any(strict):
        return strict
    return [s or _looks_like_bare_toc_entry(t) for s, t in zip(strict, texts)]


# A word hyphenated at a line-wrap ("lint-\nfree") must not become "lint- free"
# once the newline is collapsed to a space - that stray space is a rendering
# artifact of where the line happened to wrap, not a real content difference,
# and the same word in a document that didn't wrap there would just read
# "lint-free" with no space, causing a false "changed text" report otherwise.
# (Whether the hyphen itself is a real hyphen or a soft syllable-break is
# resolved later, at comparison time, by diffing hyphen-insensitively.)
_HYPHEN_LINEBREAK_RE = re.compile(r"-\s*\n\s*")

# A plain ASCII "- "/"* " at the START of a line is a bullet marker, same as
# "•" - canonicalize it to a real bullet glyph BEFORE collapsing newlines, so
# two separate bulleted lines ("- SD / SDHC / SDXC\n- MMC") don't collapse
# into one run-on sentence with a stray mid-string "- " that reads as a single
# changed line instead of two list items. Only a line-START hyphen qualifies -
# an ordinary hyphen inside a word/phrase (e.g. "USB-C") is never preceded by
# a line break, so it's left untouched.
_LINE_START_BULLET_RE = re.compile(r"(\A|\n)[-*]\s+")

# PDF text extraction can yield different Unicode code points for what's
# visually the same character depending on the font/renderer in use (curly
# vs straight quotes, en/em dash vs hyphen, non-breaking/thin spaces,
# ligatures like "fi"/"fl") - canonicalize these so a rendering-only
# difference between the two documents doesn't surface as a false "changed
# text" issue.
_UNICODE_NORMALIZE_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
        "\u200b": "", "\ufeff": "",
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
        # Superscript digits (unit exponents like "cd/m\u00b2") sometimes
        # extract as the plain digit in one PDF's font and the proper
        # superscript code point in the other's - fold to plain so "cd/m2"
        # and "cd/m\u00b2" compare equal.
        "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
        "\u2070": "0", "\u2074": "4", "\u2075": "5", "\u2076": "6",
        "\u2077": "7", "\u2078": "8", "\u2079": "9",
    }
)


def normalize_unicode_variants(text: str) -> str:
    return text.translate(_UNICODE_NORMALIZE_MAP)


# A sentence terminator with the following space missing ("...completely.If it
# is still not clean...") is a typesetting/extraction artifact, and it is a
# damaging one: sentence splitting keys off the space, so ONE document losing it
# makes that document keep two sentences glued together while the other splits
# them - and the diff then reports the second sentence as deleted even though
# both documents plainly contain it.
#
# Only applied to a whitespace-delimited token that holds exactly one dot, which
# is what keeps "Support.BenQ.com", "U.S.A.", version numbers and filenames out
# of it - a domain or an abbreviation carries more than one.
_MISSING_SPACE_AFTER_TERMINATOR_RE = re.compile(r"^(.*[a-z]{2})([.!?])([A-Z][a-z].*)$", re.DOTALL)


def _restore_sentence_spacing(text: str) -> str:
    out = []
    for token in text.split(" "):
        if token.count(".") + token.count("!") + token.count("?") == 1:
            m = _MISSING_SPACE_AFTER_TERMINATOR_RE.match(token)
            if m:
                token = f"{m.group(1)}{m.group(2)} {m.group(3)}"
        out.append(token)
    return " ".join(out)


def normalize_block_text(text: str) -> str:
    """Collapse a PDF text block's own internal line wrapping into one line,
    de-hyphenating words that were split across the wrap, canonicalizing
    ASCII line-start bullets so list items don't get glued into one sentence,
    restoring a space dropped after a sentence terminator, and canonicalizing
    Unicode look-alike characters (quotes/dashes/spaces/ligatures).
    """
    text = _LINE_START_BULLET_RE.sub(lambda m: f"{m.group(1)}\u2022 ", text)
    text = " ".join(_HYPHEN_LINEBREAK_RE.sub("-", text).split())
    text = _restore_sentence_spacing(text)
    text = normalize_unicode_variants(text)
    # Fold full-width Latin glyphs, wide spaces and non-ASCII numerals (Arabic-
    # Indic, Devanagari, ...) so a document typeset in one convention compares
    # equal to one typeset in another.
    text = i18n.fold_width(i18n.fold_digits(text))
    # Then drop any space the extractor injected mid-run inside CJK text - it
    # sits at different offsets in Production vs Staging, so without this two
    # identical Chinese/Japanese/Korean sentences read as entirely different.
    return i18n.fold_cjk_spaces(text)


# A block that is only a page number (running header/footer) isn't real
# content - comparing "2" against "4" reads as a false "changed text" issue.
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

# A block that is only a bullet/number marker with no text of its own (PDFs
# sometimes extract "•" and its text as separate blocks) gets merged into the
# next block so the bullet is diffed as one whole item, same as it reads in
# the source document.
_BARE_MARKER_RE = re.compile(r"^([\u2022\u25e6\u25aa\u25b8*\-]|\d{1,2}[.)]|[a-zA-Z][.)])$")


def looks_like_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(i18n.fold_digits(text.strip())))


# --- running headers / footers -----------------------------------------
#
# A page's running header (a section/doc title repeated at the top of every
# page) and its footer (the page number, plus a copyright / version / doc-title
# strip at the bottom) are page chrome, not body content. They never belong to
# a section's prose, and because the two documents paginate differently,
# comparing them yields nothing but noise - so they are dropped before any
# comparison, exactly as bare page numbers already are.
_HF_BAND = 0.12          # top/bottom fraction of page height that is the header/footer margin
_HF_SHORT_WORDS = 16     # a header/footer line is short; a real paragraph at the margin is not
_HF_SHORT_LINES = 3
_HF_REPEAT_MIN = 3       # a short line in the margin band on >= this many pages is chrome for sure

# Keyed by id() - PyMuPDF Documents are not weak-referenceable. Cleared by
# `reset_paragraph_cache()` at the end of every run so ids can't collide.
_hf_cache: dict[int, set] = {}


def _hf_key(text: str) -> str:
    # Digits masked so "12"/"34" page numbers and "V 1.02" collapse to one key.
    return re.sub(r"\d+", "#", " ".join(text.split())).strip().casefold()


# A CJK script has almost no spaces, so `len(s.split())` counts 1-2 "words" for
# a whole 20-character Chinese sentence - which made every Chinese line near the
# page bottom (a note box, a warning) read as a footer and vanish. Count CJK
# characters as words, and never call a line that ends like a real sentence a
# footer (a page number / version / copyright strip never does).
_CJK_RANGE_RE = re.compile(r"[\u3040-\u33ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_SENTENCE_END = ("。", "！", "？", ".", "!", "?", "；", ";", "，", "、")


def _is_short_line(text: str) -> bool:
    s = text.strip()
    if not s or s.count("\n") + 1 > _HF_SHORT_LINES:
        return False
    if s.rstrip().endswith(_SENTENCE_END):
        return False  # a finished sentence is content, not chrome
    # Count every CJK character as its own word - a spaceless script otherwise
    # collapses a whole 20-character sentence to 1-2 tokens and reads as chrome.
    n = len(s.split()) + len(_CJK_RANGE_RE.findall(s))
    return n <= _HF_SHORT_WORDS


def _scan_running_header_footer(doc: "fitz.Document") -> set:
    """{("top"|"bot", key), ...} for every short line that recurs in the top or
    bottom margin band across the document - i.e. every running header/footer."""
    seen: "Counter[tuple[str, str]]" = Counter()
    for i in range(doc.page_count):
        try:
            page = doc[i]
            rect = page.rect
            h = rect.height
            blocks = page.get_text("blocks")
        except Exception:
            continue
        if h <= 1:
            continue
        top_edge = rect.y0 + h * _HF_BAND
        bot_edge = rect.y1 - h * _HF_BAND
        for b in blocks:
            y0, y1, text = b[1], b[3], (b[4] or "")
            if not _is_short_line(text):
                continue
            k = _hf_key(text)
            if not k:
                continue
            if y0 <= top_edge:
                seen[("top", k)] += 1
            elif y1 >= bot_edge:
                seen[("bot", k)] += 1
    return {sig for sig, n in seen.items() if n >= _HF_REPEAT_MIN}


def is_running_header_footer(doc: "fitz.Document", page_index: int, bbox, text: str) -> bool:
    """True when this block is a running header/footer - a line that recurs in
    the top or bottom margin band, a bare page number, or (footer band only) any
    short strip hugging the page bottom, e.g. a back-page copyright/version line
    that appears just once.
    """
    sigs = _hf_cache.get(id(doc))
    if sigs is None:
        sigs = _scan_running_header_footer(doc)
        _hf_cache[id(doc)] = sigs
    try:
        rect = doc[page_index].rect
        h = rect.height or 1.0
    except Exception:
        return False
    y0, y1 = bbox[1], bbox[3]
    in_top = y0 <= rect.y0 + h * _HF_BAND
    in_bot = y1 >= rect.y1 - h * _HF_BAND
    if not (in_top or in_bot):
        return False
    stripped = text.strip()
    if looks_like_page_number(stripped):
        return True
    key = _hf_key(stripped)
    if in_top and ("top", key) in sigs:
        return True
    if in_bot and (("bot", key) in sigs or _is_short_line(stripped)):
        # A short line in the footer band is a footer whether or not it repeats
        # (the back-page copyright strip only appears once). The header band is
        # left to the repeat test alone - real section content legitimately
        # begins at the very top of a page.
        return True
    return False


def is_bare_marker(text: str) -> bool:
    return bool(_BARE_MARKER_RE.match(text.strip()))


def merge_bare_markers(blocks: list[str]) -> list[str]:
    """Merge a lone bullet/number marker block into the block that follows it."""
    merged: list[str] = []
    pending_marker: str | None = None
    for text in blocks:
        if is_bare_marker(text):
            pending_marker = f"{pending_marker} {text}" if pending_marker else text
            continue
        if pending_marker:
            merged.append(f"{pending_marker} {text}")
            pending_marker = None
        else:
            merged.append(text)
    if pending_marker:
        merged.append(pending_marker)
    return merged


# A block boundary doesn't always line up with a sentence boundary - PyMuPDF
# sometimes splits one wrapped paragraph into two separate blocks at an
# ordinary word-wrap point (no hyphen, no punctuation at all around the
# split). If a block doesn't end in sentence-ending punctuation and the next
# block starts with a lowercase letter, it's almost certainly the same
# sentence continuing rather than two distinct sentences - merge them so both
# documents reach the same granularity before diffing, instead of a false
# "Missing text"/"Added text" pair for identical content split differently.
# Scripts without letter case (CJK, Arabic, Hebrew, Thai, ...) give no
# "starts lowercase" signal, so there a longer-than-a-heading previous block
# that doesn't end in a terminator is taken as the continuation cue instead.
_CASELESS_CONTINUATION_MIN_PREV_LEN = 20


def _is_wrap_continuation(prev_text: str, next_text: str) -> bool:
    if not prev_text or not next_text or i18n.ends_with_terminator(prev_text):
        return False
    first = next_text[0]
    if first.islower():
        return True
    if i18n.is_caseless_letter(first):
        return len(prev_text.strip()) >= _CASELESS_CONTINUATION_MIN_PREV_LEN
    return False


def merge_wrapped_sentence_blocks(blocks: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for block in blocks:
        if merged:
            prev = merged[-1]
            if _is_wrap_continuation(prev["text"], block["text"]):
                # A wrap that fell on a hyphen ("...specifica-" + "tions...")
                # joins with no space so the word isn't broken by a stray
                # space; an ordinary word-wrap joins with a space. Whether the
                # hyphen is real or a soft syllable-break is settled later by
                # diffing hyphen-insensitively.
                sep = "" if re.search(r"\w-$", prev["text"]) else " "
                merged[-1] = {
                    "text": f"{prev['text']}{sep}{block['text']}",
                    "page": prev["page"],
                    "bbox": _union_bbox(prev["bbox"], block["bbox"]),
                }
                continue
        merged.append(dict(block))
    return merged


# A lone bullet/number glyph belongs to the text that sits right next to it.
# If the next block is far below or well to the left of the marker, they are
# not a pair - the marker is a table's own stray glyph and the block is
# unrelated prose - so the marker is dropped rather than glued on.
_MARKER_MAX_VGAP = 24.0  # points below the marker the paired text may start
_MARKER_MAX_LEFT_OF = 8.0  # points the paired text may sit left of the marker


def _marker_pairs_with(marker: dict, block: dict) -> bool:
    if marker["page"] != block["page"]:
        return False
    mx0, my0, _, my1 = marker["bbox"]
    bx0, by0, _, _ = block["bbox"]
    return by0 <= my1 + _MARKER_MAX_VGAP and bx0 >= mx0 - _MARKER_MAX_LEFT_OF


def merge_bare_marker_blocks(blocks: list[dict]) -> list[dict]:
    """Block-dict version of `merge_bare_markers` that also unions the bbox of
    the merged marker into the block it's attached to. A marker only merges
    into the next block when that block actually sits beside/below it; an
    orphan marker with no adjacent text is discarded.
    """
    merged: list[dict] = []
    pending: dict | None = None
    for block in blocks:
        if is_bare_marker(block["text"]):
            if pending is None or _marker_pairs_with(pending, block):
                if pending is None:
                    pending = dict(block)
                else:
                    pending["text"] = f"{pending['text']} {block['text']}"
                    pending["bbox"] = _union_bbox(pending["bbox"], block["bbox"])
            continue
        if pending is not None and _marker_pairs_with(pending, block):
            merged.append(
                {
                    "text": f"{pending['text']} {block['text']}",
                    "page": block["page"],
                    "bbox": _union_bbox(pending["bbox"], block["bbox"]),
                }
            )
            pending = None
        else:
            pending = None  # orphan marker with no adjacent text - drop it
            merged.append(block)
    return merged


def _union_bbox(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


@dataclass
class TocEntry:
    level: int
    title: str
    page: int  # 0-based page index
    y: float  # heading's top-left-origin y-coordinate on that page
    excluded: bool = False  # out-of-scope section (e.g. Q&A) - kept only for section boundaries


@dataclass
class TocMatch:
    expected_index: int | None
    actual_index: int | None


def get_toc_entries(doc: fitz.Document) -> list[TocEntry]:
    """Read the embedded outline and resolve each heading's on-page y position.

    PDF outlines don't use one consistent coordinate convention for the
    bookmark destination point across documents (some are bottom-left/native,
    some are already top-left, depending on how the destination was encoded).
    Rather than trust a single formula, we ground the position in the actual
    rendered page: look for a text block whose content matches the heading
    title, and use its real bbox. The destination point's y (flipped to
    top-left origin) is only a fallback when no matching block is found.
    """
    entries: list[TocEntry] = []
    for level, title, page_1based, dest in doc.get_toc(simple=False):
        page_index = max(0, min(page_1based - 1, doc.page_count - 1))
        title = " ".join(title.split())
        y = _locate_heading_y(doc, page_index, title, dest)
        entries.append(TocEntry(level=level, title=title, page=page_index, y=y, excluded=is_excluded_heading(title)))
    return entries


def _locate_heading_y(doc: fitz.Document, page_index: int, title: str, dest: object) -> float:
    page = doc[page_index]
    normalized_title = normalize_title(title)
    if normalized_title:
        candidates = []
        for block in page.get_text("blocks"):
            text = normalize_title(block[4])
            if not text:
                continue
            if text == normalized_title or text.startswith(normalized_title) or normalized_title.startswith(text):
                candidates.append(block[1])
        if candidates:
            return min(candidates)

    to = dest.get("to") if isinstance(dest, dict) else None
    if to is None:
        return 0.0
    # Fall back to the native (bottom-left-origin) PDF destination, flipped.
    return page.rect.height - to.y


def normalize_title(title: str) -> str:
    return " ".join(title.split()).lower()


def match_toc_entries(
    expected_entries: list[TocEntry],
    actual_entries: list[TocEntry],
    include_excluded: bool = False,
) -> list[TocMatch]:
    """Align headings between the two TOCs by title text; page numbers are ignored.

    Excluded headings (e.g. a Q&A / contents index) are left out of the match
    by default - Content Validation must not diff their listing text. Pass
    `include_excluded=True` to align them too (the section browser does this so
    it can still LIST the heading, just with a "not compared" note); the
    returned indices always refer to positions in the original entry lists.
    """
    keep = (lambda e: True) if include_excluded else (lambda e: not e.excluded)
    exp_visible = [i for i, e in enumerate(expected_entries) if keep(e)]
    act_visible = [i for i, e in enumerate(actual_entries) if keep(e)]
    exp_titles = [normalize_title(expected_entries[i].title) for i in exp_visible]
    act_titles = [normalize_title(actual_entries[i].title) for i in act_visible]
    matcher = difflib.SequenceMatcher(a=exp_titles, b=act_titles, autojunk=False)

    matches: list[TocMatch] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                matches.append(TocMatch(exp_visible[i], act_visible[j]))
        elif tag == "delete":
            matches.extend(TocMatch(exp_visible[i], None) for i in range(i1, i2))
        elif tag == "insert":
            matches.extend(TocMatch(None, act_visible[j]) for j in range(j1, j2))
        elif tag == "replace":
            used_act: set[int] = set()
            for i in range(i1, i2):
                best_j, best_ratio = None, 0.0
                for j in range(j1, j2):
                    if j in used_act:
                        continue
                    ratio = difflib.SequenceMatcher(None, exp_titles[i], act_titles[j], autojunk=False).ratio()
                    if ratio > best_ratio:
                        best_ratio, best_j = ratio, j
                if best_j is not None and best_ratio >= TITLE_MATCH_THRESHOLD:
                    used_act.add(best_j)
                    matches.append(TocMatch(exp_visible[i], act_visible[best_j]))
                else:
                    matches.append(TocMatch(exp_visible[i], None))
            matches.extend(TocMatch(None, act_visible[j]) for j in range(j1, j2) if j not in used_act)
    return matches


@dataclass
class SectionBounds:
    """A heading's section span, precise to the y-position on its start/end
    pages (not just the page number) - needed because multiple headings
    often share the same page, and a page-only range would double-count
    that page's content across every heading that touches it.
    """

    start_page: int
    start_y: float
    end_page: int
    end_y: float


def in_section_bounds(page: int, y: float, bounds: SectionBounds, tolerance: float = 3.0) -> bool:
    if page < bounds.start_page or page > bounds.end_page:
        return False
    if page == bounds.start_page and y < bounds.start_y - tolerance:
        return False
    if page == bounds.end_page and y >= bounds.end_y - tolerance:
        return False
    return True


def _section_bounds_list(entries: list[TocEntry], page_count: int) -> list[SectionBounds]:
    """Each entry's `SectionBounds`, bounded only by the NEXT entry in true
    reading order (page, then y) within this same list - so callers can pass
    a matched-only subset and an extra/missing heading on one side won't
    prematurely cut a section's range short there.
    """
    reading_order = sorted(range(len(entries)), key=lambda i: (entries[i].page, entries[i].y))
    next_index: dict[int, int | None] = {}
    for pos, idx in enumerate(reading_order):
        next_index[idx] = reading_order[pos + 1] if pos + 1 < len(reading_order) else None

    bounds: list[SectionBounds] = []
    for idx, entry in enumerate(entries):
        next_idx = next_index[idx]
        if next_idx is not None:
            end_page, end_y = entries[next_idx].page, entries[next_idx].y
        else:
            end_page, end_y = page_count - 1, float("inf")
        bounds.append(SectionBounds(entry.page, entry.y, end_page, end_y))
    return bounds


def matched_heading_ranges(
    exp_entries: list[TocEntry], act_entries: list[TocEntry], exp_page_count: int, act_page_count: int
) -> list[tuple[TocEntry, TocEntry, SectionBounds, SectionBounds]]:
    """For each heading that exists on BOTH sides, return
    (exp_entry, act_entry, exp_bounds, act_bounds) - the section span that
    heading covers on each side, precise to the y-position (see
    `SectionBounds`/`in_section_bounds`).

    Page-index-based matching (comparing page N on one document against page
    N on the other) silently compares the wrong pages once the two documents'
    page counts diverge - this anchors the comparison to the heading itself
    instead, the same way Content Validation is heading-anchored. Bounded
    only by the NEXT MUTUALLY-MATCHED heading, so an extra/missing heading on
    one side doesn't cut a section short there.
    """
    matches = match_toc_entries(exp_entries, act_entries)
    matched = [m for m in matches if m.expected_index is not None and m.actual_index is not None]

    exp_matched = [exp_entries[m.expected_index] for m in matched]
    act_matched = [act_entries[m.actual_index] for m in matched]
    exp_bounds = _section_bounds_list(exp_matched, exp_page_count)
    act_bounds = _section_bounds_list(act_matched, act_page_count)

    return [
        (exp_matched[i], act_matched[i], exp_bounds[i], act_bounds[i])
        for i in range(len(matched))
    ]


def heading_at(entries: list[TocEntry], page: int, y: float) -> str | None:
    """Return the (non-excluded) heading title whose section contains a page/y position."""
    ordered = sorted((e for e in entries if not e.excluded), key=lambda e: (e.page, e.y))
    result = None
    for e in ordered:
        if (e.page, e.y) <= (page, y):
            result = e.title
        else:
            break
    return result


def extract_sections(doc: fitz.Document, entries: list[TocEntry]) -> list[str]:
    """For each TOC entry, return the text from that heading up to the next heading (any level).

    The "next heading" is resolved by true reading order (page, then y) rather
    than bookmark-list order: some PDFs' outline trees list a short callout/
    sidebar heading after a heading that is spatially lower on the same page,
    which would otherwise make a section swallow unrelated later content.
    """
    return ["\n".join(b["text"] for b in blocks) for blocks in extract_section_blocks(doc, entries)]


def extract_section_blocks(doc: fitz.Document, entries: list[TocEntry]) -> list[list[dict]]:
    """Same section boundaries as `extract_sections`, but keep each text block's
    own page/bbox so callers can locate a specific sentence on the rendered
    page (e.g. to draw a highlight box in a screenshot).
    """
    reading_order = sorted(range(len(entries)), key=lambda i: (entries[i].page, entries[i].y))
    next_index: dict[int, int | None] = {}
    for pos, idx in enumerate(reading_order):
        next_index[idx] = reading_order[pos + 1] if pos + 1 < len(reading_order) else None

    sections: list[list[dict]] = []
    for idx, entry in enumerate(entries):
        next_idx = next_index[idx]
        if next_idx is not None:
            end_page, end_y = entries[next_idx].page, entries[next_idx].y
        else:
            end_page, end_y = doc.page_count - 1, float("inf")
        sections.append(_extract_blocks_range(doc, entry.page, entry.y, end_page, end_y, entry.title))
    return sections


# --- paragraph reconstruction --------------------------------------------
#
# PyMuPDF's get_text("blocks") grouping is what keeps a page's TABLE cells,
# multi-column layouts and diagram callouts apart - so it stays the base unit.
# But two PDFs from different producers do NOT get grouped the same: the very
# same body paragraph is ONE block in Production and TWO blocks (split at a
# line-wrap) in Staging, and the text diff then reports the tail as a
# missing/added sentence. `_merge_wrapped_blocks` fixes exactly that: it re-
# joins consecutive blocks the geometry proves are one wrapped paragraph
# (vertically adjacent, same left edge, previous block's last line reached the
# column's right edge so it *wrapped* rather than ended deliberately, same font
# size, next block not a fresh list item). Everything else - a deliberately
# short line, a table cell, a heading, a callout - is left exactly where the
# block extractor put it.

_BULLET_CHARS = "\u2022\u25cf\u25e6\u25aa\u25b6\u25b8\u2023\u2043\u00b7\u2219*\u25cb"
_LIST_ITEM_START_RE = re.compile(
    r"^\s*(?:[" + re.escape(_BULLET_CHARS) + r"]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])(?:\s|$)"
)
_PARA_MERGE_MAX_VGAP_RATIO = 1.4   # blocks farther apart than this * line-height stay separate
_PARA_MERGE_LEFT_TOL = 8.0         # left edges must line up within this many points
_PARA_MERGE_RIGHT_SLACK = 12.0     # prev block's last line must end within this of the column edge
_PARA_MERGE_SIZE_TOL = 1.0         # font-size difference that still counts as one paragraph


def _block_line_geometry_from(data: dict) -> dict:
    """Per-block first-line-x0, last-line-x1, size and line-height, keyed by the
    block's rounded bbox - the plain block extractor doesn't hand these back and
    the merge test needs them. Takes an already-fetched get_text("dict")."""
    out = {}
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        rows = [
            ln for ln in block.get("lines", [])
            if "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
        ]
        if not rows:
            continue
        text_spans = [
            s for ln in rows for s in ln.get("spans", []) if s.get("text", "").strip()
        ]
        sizes = [float(s.get("size", 0.0)) for s in text_spans]
        # A span is bold if its flag bit 4 is set or its font name says so.
        bold_chars = sum(
            len(s.get("text", "")) for s in text_spans
            if (int(s.get("flags", 0)) & 16) or "bold" in str(s.get("font", "")).lower()
        )
        total_chars = sum(len(s.get("text", "")) for s in text_spans) or 1
        first, last = rows[0]["bbox"], rows[-1]["bbox"]
        out[_round_bbox(block.get("bbox", (0, 0, 0, 0)))] = {
            "first_x0": float(first[0]),
            "last_x1": float(last[2]),
            "size": max(sizes) if sizes else 10.0,
            "line_h": max(float(first[3] - first[1]), 1.0),
            "n_lines": len(rows),
            "mostly_bold": bold_chars / total_chars >= 0.7,
        }
    return out


def _round_bbox(b):
    return tuple(round(float(v)) for v in b)


def _page_column_right_from(data: dict, page) -> float:
    """Body column right edge = the most common line-end x on the page."""
    try:
        rights = {}
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for ln in block.get("lines", []):
                if "".join(s.get("text", "") for s in ln.get("spans", [])).strip():
                    key = round(ln["bbox"][2])
                    rights[key] = rights.get(key, 0) + 1
        if rights:
            return float(max(rights.items(), key=lambda kv: kv[1])[0])
    except Exception:
        pass
    return float(page.rect.x1)


def _merge_wrapped_blocks(page, page_index):
    """PyMuPDF blocks with a paragraph split across a line-wrap re-joined."""
    raw = [b for b in page.get_text("blocks") if b[4].strip()]
    if not raw:
        return []
    raw.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    try:
        data = page.get_text("dict")  # one call, shared by both helpers below
    except Exception:
        data = {"blocks": []}
    geom = _block_line_geometry_from(data)
    col_right = _page_column_right_from(data, page)

    out = []
    prev_g = None
    for b in raw:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
        g = geom.get(_round_bbox((x0, y0, x1, y1)))
        merged = False
        if out and prev_g and g:
            p = out[-1]
            vgap = y0 - p["bbox"][3]
            line_h = max(g["line_h"], prev_g["line_h"])
            same_left = abs(g["first_x0"] - prev_g["first_x0"]) <= _PARA_MERGE_LEFT_TOL
            same_size = abs(g["size"] - prev_g["size"]) <= _PARA_MERGE_SIZE_TOL
            # The definitive "this continues onto the next block" signal: the
            # previous block's LAST line physically reached the column's right
            # edge, i.e. it wrapped. A short heading, a caption, or a
            # deliberately short sentence-ending line never does - so none of
            # them get glued to the block below.
            prev_wrapped = prev_g["last_x1"] >= col_right - _PARA_MERGE_RIGHT_SLACK
            next_is_item = bool(_LIST_ITEM_START_RE.match(text))
            # A single bold line that doesn't end a sentence is a run-in
            # sub-heading ("Enabling HDR function") - never glue the body to it.
            prev_is_heading = (
                prev_g.get("n_lines", 1) == 1
                and prev_g.get("mostly_bold")
                and not i18n.ends_with_terminator(p["text"])
                and not g.get("mostly_bold")
            )
            # A bare marker ("1.", "a)") the PDF placed to the RIGHT of its own
            # list item (so it sorts just after the item text) belongs at the
            # FRONT of that item - which is where the other document, and a
            # reader, sees it.
            same_row = abs(y0 - p["bbox"][1]) <= 1.5 * line_h
            if (
                is_bare_marker(text)
                and same_row
                and not _LIST_ITEM_START_RE.match(p["text"])
                and not is_bare_marker(p["text"].split("\n", 1)[0])
            ):
                p["text"] = f"{text} {p['text']}"
                p["bbox"] = (
                    min(p["bbox"][0], x0), min(p["bbox"][1], y0),
                    max(p["bbox"][2], x1), max(p["bbox"][3], y1),
                )
                merged = True
            elif (
                0 <= vgap <= _PARA_MERGE_MAX_VGAP_RATIO * line_h
                and same_left and same_size and prev_wrapped
                and not next_is_item and not prev_is_heading
            ):
                p["text"] = p["text"] + "\n" + text
                p["bbox"] = (
                    min(p["bbox"][0], x0), min(p["bbox"][1], y0),
                    max(p["bbox"][2], x1), max(p["bbox"][3], y1),
                )
                merged = True
        if not merged:
            out.append({"text": text, "page": page_index, "bbox": (x0, y0, x1, y1)})
        prev_g = g
    return out


# `extract_section_blocks` is called several times per run (all entries, the
# matched-wide subset, then again from the section browser), and each call
# re-walks the same pages - so `_merge_wrapped_blocks` (two get_text calls a
# page) was being run thousands of times on a large manual and the report
# stopped finishing. Memoise per (document-id, page). Keyed by id() because
# PyMuPDF Documents are not weak-referenceable; `reset_paragraph_cache()` is
# called at the end of every run so ids can't collide across runs.
_PARA_CACHE: dict[tuple[int, int], list[dict]] = {}


def reset_paragraph_cache() -> None:
    _PARA_CACHE.clear()
    _hf_cache.clear()


def _reconstruct_paragraphs(page, page_index):
    key = (id(page.parent), page_index)
    cached = _PARA_CACHE.get(key)
    if cached is None:
        cached = _merge_wrapped_blocks(page, page_index)
        _PARA_CACHE[key] = cached
    # Hand back copies - callers mutate the block dicts (normalize_block_text,
    # bbox unions) and must not corrupt the shared cache entry.
    return [dict(b) for b in cached]


def _extract_blocks_range(
    doc: fitz.Document, start_page: int, start_y: float, end_page: int, end_y: float, heading_title: str = ""
) -> list[dict]:
    tolerance = 3.0
    normalized_heading = normalize_title(heading_title)
    blocks_out: list[dict] = []
    last_page = min(end_page, doc.page_count - 1)
    for page_index in range(start_page, last_page + 1):
        page = doc[page_index]
        paragraphs = _reconstruct_paragraphs(page, page_index)
        toc_flags = filter_toc_page_blocks([p["text"] for p in paragraphs])
        for para, is_toc in zip(paragraphs, toc_flags):
            x0, y0, x1, y1 = para["bbox"]
            text = para["text"]
            if not text.strip():
                continue
            if page_index == start_page and y0 < start_y - tolerance:
                continue
            if page_index == end_page and y0 >= end_y - tolerance:
                continue
            if is_toc or looks_like_page_number(text):
                continue
            if is_running_header_footer(doc, page_index, para["bbox"], text):
                continue
            # The heading's own title line is already validated by the TOC
            # comparison table - skip it here so it isn't also reported as a
            # spurious "missing"/"added" content diff.
            if normalized_heading and normalize_title(text) == normalized_heading:
                continue
            blocks_out.append({"text": normalize_block_text(text), "page": page_index, "bbox": (x0, y0, x1, y1)})
    blocks_out.sort(key=lambda b: (b["page"], round(b["bbox"][1]), round(b["bbox"][0])))
    # merge_wrapped_sentence_blocks is now a light safety net (paragraphs are
    # already whole); merge_bare_marker_blocks still attaches stray bullet
    # glyphs that PyMuPDF emits as their own line.
    return merge_bare_marker_blocks(merge_wrapped_sentence_blocks(blocks_out))


def validate_toc(expected: fitz.Document, actual: fitz.Document) -> CheckResult:
    result = CheckResult(name="TOC Validation")
    exp_entries = get_toc_entries(expected)
    act_entries = get_toc_entries(actual)

    if not exp_entries and not act_entries:
        result.issues.append(
            Issue(severity="warning", page=None, message="No table of contents found in either document")
        )
        return result
    if not exp_entries or not act_entries:
        side = "expected" if not exp_entries else "actual"
        result.issues.append(
            Issue(severity="warning", page=None, message=f"Table of contents is missing from the {side} document")
        )
        return result

    matches = match_toc_entries(exp_entries, act_entries)
    matched = sum(1 for m in matches if m.expected_index is not None and m.actual_index is not None)
    missing = sum(1 for m in matches if m.expected_index is not None and m.actual_index is None)
    added = sum(1 for m in matches if m.actual_index is not None and m.expected_index is None)
    visible_exp = [e for e in exp_entries if not e.excluded]
    visible_act = [e for e in act_entries if not e.excluded]

    result.issues.append(
        Issue(
            severity="info",
            page=None,
            message="TOC status",
            details={
                "expected_headings": len(visible_exp),
                "actual_headings": len(visible_act),
                "matched": matched,
                "missing": missing,
                "added": added,
            },
        )
    )

    return result


def build_toc_comparison(expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A clean, per-heading table: prod page vs stage page, and whether the
    heading matched, is missing (only in prod), or is extra (only in stage).
    """
    exp_entries = get_toc_entries(expected)
    act_entries = get_toc_entries(actual)
    if not exp_entries or not act_entries:
        return []

    rows: list[dict] = []
    for m in match_toc_entries(exp_entries, act_entries):
        if m.expected_index is not None and m.actual_index is not None:
            exp_e, act_e = exp_entries[m.expected_index], act_entries[m.actual_index]
            rows.append(
                {
                    "heading": exp_e.title,
                    "expected_level": exp_e.level,
                    "actual_level": act_e.level,
                    "expected_page": exp_e.page + 1,
                    "actual_page": act_e.page + 1,
                    "status": "Matched",
                }
            )
        elif m.expected_index is not None:
            exp_e = exp_entries[m.expected_index]
            rows.append(
                {
                    "heading": exp_e.title,
                    "expected_level": exp_e.level,
                    "actual_level": None,
                    "expected_page": exp_e.page + 1,
                    "actual_page": None,
                    "status": "Missing",
                }
            )
        else:
            act_e = act_entries[m.actual_index]
            rows.append(
                {
                    "heading": act_e.title,
                    "expected_level": None,
                    "actual_level": act_e.level,
                    "expected_page": None,
                    "actual_page": act_e.page + 1,
                    "status": "Extra",
                }
            )
    return rows


def build_toc_report(expected: fitz.Document, actual: fitz.Document) -> dict:
    """The full standalone bookmark/TOC comparison behind `toc.html`.

    Every bookmark from either document, in reading order, with its status and
    the specific way it differs (dropped, added, renumbered to a different
    nesting level, moved to a different position, or landed on a different page
    number). Plus a one-line count of each so the page can lead with the tally.
    """
    # This IS the dedicated bookmark comparison, so it lists EVERY bookmark -
    # including a Q&A/contents-index heading that the section-by-section content
    # browser skips. Clear the `excluded` flag so those still take part.
    from dataclasses import replace as _replace

    exp_entries = [_replace(e, excluded=False) for e in get_toc_entries(expected)]
    act_entries = [_replace(e, excluded=False) for e in get_toc_entries(actual)]
    have_exp, have_act = bool(exp_entries), bool(act_entries)

    matches = match_toc_entries(exp_entries, act_entries)

    rows: list[dict] = []
    counts = {"matched": 0, "missing": 0, "extra": 0, "level_changed": 0, "page_shifted": 0}

    for m in matches:
        if m.expected_index is not None and m.actual_index is not None:
            e, a = exp_entries[m.expected_index], act_entries[m.actual_index]
            level_changed = e.level != a.level
            page_delta = a.page - e.page
            counts["matched"] += 1
            if level_changed:
                counts["level_changed"] += 1
            if page_delta != 0:
                counts["page_shifted"] += 1
            notes = []
            if level_changed:
                notes.append(f"nesting level {e.level} → {a.level}")
            if page_delta != 0:
                notes.append(f"page {e.page + 1} → {a.page + 1} ({page_delta:+d})")
            rows.append({
                "heading": e.title,
                "expected_level": e.level,
                "actual_level": a.level,
                "expected_page": e.page + 1,
                "actual_page": a.page + 1,
                "status": "Changed" if level_changed else "Matched",
                "level_changed": level_changed,
                "page_delta": page_delta,
                "note": "; ".join(notes),
            })
        elif m.expected_index is not None:
            e = exp_entries[m.expected_index]
            counts["missing"] += 1
            rows.append({
                "heading": e.title,
                "expected_level": e.level,
                "actual_level": None,
                "expected_page": e.page + 1,
                "actual_page": None,
                "status": "Missing",
                "level_changed": False,
                "page_delta": None,
                "note": "in Production only — this bookmark is not in Staging",
            })
        else:
            a = act_entries[m.actual_index]
            counts["extra"] += 1
            rows.append({
                "heading": a.title,
                "expected_level": None,
                "actual_level": a.level,
                "expected_page": None,
                "actual_page": a.page + 1,
                "status": "Extra",
                "level_changed": False,
                "page_delta": None,
                "note": "in Staging only — Staging adds this bookmark",
            })

    return {
        "have_expected_toc": have_exp,
        "have_actual_toc": have_act,
        "expected_headings": sum(1 for e in exp_entries if not e.excluded),
        "actual_headings": sum(1 for e in act_entries if not e.excluded),
        "counts": counts,
        "rows": rows,
    }
