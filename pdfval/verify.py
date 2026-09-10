"""Second-pass verification: every finding is re-checked against the FULL text
of both PDFs before it is allowed into the report as a confirmed issue.

The individual validators are heuristic and section-anchored, so they produce a
predictable family of false positives: a sentence that only moved to another
page reads as "missing", a table row reformatted as prose reads as "lost", a
font that one PDF calls ``Roboto`` and the other ``Roboto-Regular`` reads as a
formatting change. Almost all of them share one tell - the thing claimed to be
missing/changed is still right there somewhere in the other document.

This pass asks exactly that question, document-wide, and:
  * drops a finding whose content is demonstrably still present unchanged, and
  * marks the rest `confidence = "confirmed"` or `"review"`.

The HTML report shows confirmed findings by default; "review" ones are kept but
folded away, so nothing valid is ever silently lost.
"""
from __future__ import annotations

import re
import unicodedata

import fitz

from pdfval.extractor import get_figures, get_tables
from pdfval.models import Issue, ValidationReport

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Words too generic for "these exact words appear over there" to mean anything.
_STOPWORDS = frozenset(
    "the a an and or of to for in on at is are was were be been being with by from as "
    "it this that these those you your we our they their he she его not no yes if then "
    "will can may should must would could into out up down over under more most less "
    "see refer press select use using set click menu option options item items page".split()
)


def _norm(text: str) -> str:
    # NFKD + drop combining marks folds accents ("Česky" == "Cesky",
    # "română" == "romana"), ligatures and full-width glyphs, so a diacritic
    # the two exports disagree on isn't read as a lost word.
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return "".join(
        ch for ch in decomposed if ch.isalnum() and not unicodedata.combining(ch)
    )


def _content_words(text: str) -> list[str]:
    out = []
    for w in _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold()):
        if len(w) >= 3 and w not in _STOPWORDS and not w.isdigit():
            out.append(w)
    return out


class _DocIndex:
    """Whole-document text, in the two shapes the checks below need: a set of
    normalised words, and one big normalised (letters+digits only) string that
    survives any amount of re-wrapping / re-spacing / cell-vs-prose reflow.
    """

    def __init__(self, doc: fitz.Document):
        parts: list[str] = []
        words: set[str] = set()
        for i in range(doc.page_count):
            try:
                t = doc[i].get_text()
            except Exception:
                continue
            parts.append(t)
            words.update(_WORD_RE.findall(unicodedata.normalize("NFKC", t).casefold()))
        self.blob = _norm(" ".join(parts))
        self.words = words
        self.norm_words = {_norm(w) for w in words}
        self.norm_words.discard("")

    def has_phrase(self, text: str, min_len: int = 12) -> bool:
        key = _norm(text)
        if len(key) < min_len:
            return False
        return key in self.blob

    def has_all_content_words(self, text: str, need_ratio: float = 1.0) -> bool:
        cw = _content_words(text)
        if not cw:
            return False
        hit = sum(1 for w in cw if w in self.words)
        return hit / len(cw) >= need_ratio


# --- font-name normalisation (Text formatting differs) --------------------

_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")
_FONT_STYLE_RE = re.compile(
    r"[\s,_-]*(regular|italic|oblique|bold|semibold|demibold|demi|medium|light|thin|"
    r"black|heavy|book|normal|condensed|narrow|display|text|mt|ms)+$",
    re.IGNORECASE,
)


def _font_core(style_desc: str) -> str:
    """From "Roboto-Regular · 12.0pt" keep just the comparable family key."""
    name = style_desc.split("·")[0].split("·")[0]
    name = _SUBSET_RE.sub("", name).split(",")[0].strip()
    prev = None
    while name and name != prev:
        prev, name = name, _FONT_STYLE_RE.sub("", name).strip()
    return re.sub(r"[\s_-]+", "", name).casefold()


def _size_pt(style_desc: str) -> float | None:
    m = re.search(r"([\d.]+)\s*pt", style_desc)
    return float(m.group(1)) if m else None


# --- per-message verdicts ------------------------------------------------

_LOW_CONFIDENCE_MESSAGES = frozenset(
    {
        "Image could not be confirmed identical",
        "Image alignment changed",
        "Image highlight box missing",
        "Table column layout differs",
        "Table cell layout differs",
        "Table split across pages",
        "Possible text encoding issue",
        # Indent / alignment is measured in points and drifts between two
        # re-exports of the same source; only a wholesale change is meaningful,
        # and even then it is advisory, never a blocking failure.
        "List alignment/indent changed",
        "Text alignment changed",
        "List indent changed",
        "Alignment changed",
    }
)

_VOWEL_RE = re.compile(r"[aeiouyàâäéèêëíìîïóòôöúùûü]", re.IGNORECASE)


def _looks_like_real_word(w: str) -> bool:
    return len(w) >= 4 and bool(_VOWEL_RE.search(w))


def _is_prose_line(s: str) -> bool:
    """A line that reads as a real sentence: several words and, unless it is
    long, sentence-ending punctuation. A lone part label ("Rt"), a callout name
    ("Release button"), a caption stub ("For detailed connection methods -"),
    a menu-path legend or a spec run is not prose - those belong to Image /
    Table Validation, not the text diff."""
    try:
        from pdfval.validators.content import _looks_like_non_prose
    except Exception:
        _looks_like_non_prose = lambda _t: False  # noqa: E731
    toks = s.split()
    if len(toks) < 3 or _looks_like_non_prose(s):
        return False
    ends_sentence = s.rstrip().endswith((".", "!", "?", ":", ";"))
    return ends_sentence or len(toks) >= 8


def _all_non_prose(sentences: list[str]) -> bool:
    """No line in the list reads as real flowing prose."""
    return not any(_is_prose_line(s) for s in sentences)


_INDEX_LINE_RE = re.compile(r"\b\d{1,3}\s*$|\?\s+[A-Z]|\b[A-Z][a-z].{4,}\s+\d{1,3}\b")


def _looks_like_index_dump(sentences: list[str]) -> bool:
    """A long run of lines that each end in a bare page number or pack several
    "Heading 24" entries together is a table-of-contents / FAQ index that the
    section extractor swept in - not a real prose change."""
    if len(sentences) <= 6:
        return False
    hits = sum(1 for s in sentences if _INDEX_LINE_RE.search(s))
    return hits / len(sentences) >= 0.35


def _fragments_present_elsewhere(word_diff: list, exp_idx: "_DocIndex", act_idx: "_DocIndex") -> bool:
    """A "Changed text" diff whose deleted run is still present verbatim in the
    Staging document AND whose inserted run is still present verbatim in the
    Production document is not an edit - it is a sentence-boundary disagreement
    (one document glued two sentences together, or a paragraph reflowed onto
    the next page), so the diff pairs a short sentence against a long one and
    calls the leftover "changed" while a reader sees the same words on both.
    """
    deleted = " ".join(
        op.get("text", "")
        for ops in word_diff
        for op in ops
        if op.get("type") == "del" and op.get("text", "").strip()
    ).strip()
    inserted = " ".join(
        op.get("text", "")
        for ops in word_diff
        for op in ops
        if op.get("type") == "ins" and op.get("text", "").strip()
    ).strip()
    if not deleted or not inserted:
        return False
    return act_idx.has_phrase(deleted, min_len=16) and exp_idx.has_phrase(inserted, min_len=16)

_DROP = object()  # sentinel: remove this issue entirely

# A short diagram-callout caption ("On the main menu, the KVM status is
# displayed.") is sometimes kept as real, selectable text in one PDF export
# but flattened into the diagram's raster image in the other - the words are
# still visually on the page, just no longer in that side's text layer, so a
# pure text diff reports it "missing" when nothing was actually deleted. Two
# signals make this plausible rather than a real loss: the claimed content is
# short (a caption, not a paragraph) and the page it would appear on has a
# substantial image sitting right there to have absorbed it.
_CAPTION_LIKE_MAX_WORDS = 14
_SUBSTANTIAL_IMAGE_AREA = 20000.0  # ~140x140pt - a real diagram/mockup, not a small icon
_PAGE_LOOKAHEAD = 3  # the stored page is the section's start, not necessarily where the image sits


def _has_substantial_image_nearby(doc: fitz.Document | None, page_index: int | None) -> bool:
    if doc is None or page_index is None:
        return False
    for p in range(max(0, page_index), min(doc.page_count, page_index + _PAGE_LOOKAHEAD)):
        try:
            figures = get_figures(doc, p)
        except Exception:
            continue
        if any(f.display_area >= _SUBSTANTIAL_IMAGE_AREA for f in figures):
            return True
    return False


# A real table that one document's PDF export renders cleanly can come out
# malformed (a phantom merged cell, garbled geometry) on the other side -
# extraction then can't trust its shape as a table at all, so its OWN row/
# header text ("OSD icon 5-way controller operation Function") leaks into
# Content Validation as if it were prose, while the clean side correctly kept
# it out as table content. That reads as "missing", though nothing was lost -
# it's the same table, just classified differently on each side. A match is
# only trusted when MOST of the claimed sentence's real words show up in one
# table's cells, not merely a couple of common ones.
_TABLE_ROW_WORD_OVERLAP = 0.7


def _has_matching_table_row_nearby(path: str | None, page_index: int | None, sentences: list[str]) -> bool:
    if not path or page_index is None:
        return False
    words = {w for s in sentences for w in _content_words(s)}
    if not words:
        return False
    for p in range(max(0, page_index), page_index + _PAGE_LOOKAHEAD):
        try:
            tables = get_tables(path, p)
        except Exception:
            continue
        for table in tables:
            blob = _norm(" ".join(cell for row in (table.get("rows") or []) for cell in row if cell))
            if not blob:
                continue
            hits = sum(1 for w in words if _norm(w) in blob)
            if hits / len(words) >= _TABLE_ROW_WORD_OVERLAP:
                return True
    return False


def _verdict(
    issue: Issue,
    exp_idx: _DocIndex,
    act_idx: _DocIndex,
    exp_doc: fitz.Document | None = None,
    act_doc: fitz.Document | None = None,
    exp_path: str | None = None,
    act_path: str | None = None,
):
    d = issue.details or {}
    msg = issue.message

    if msg == "Minor text difference":
        # Already gated to "near-identical, still present in the other doc" by
        # Content Validation itself, and explicitly a review-only finding.
        # Drop it only when there is literally nothing textual to show.
        wd = d.get("word_diff")
        changed = "".join(
            op.get("text", "")
            for ops in (wd or [])
            for op in ops
            if op.get("type") in ("del", "ins")
        ).strip()
        if not wd and not d.get("expected") and not d.get("actual"):
            return _DROP
        return "review"

    if msg == "Missing text":
        sents = [s for s in (d.get("expected") or []) if isinstance(s, str) and not s.startswith("... (+")]
        if not sents or _all_non_prose(sents) or _looks_like_index_dump(sents):
            return _DROP
        # Deliberately does NOT drop just because this exact wording also
        # exists somewhere else in the whole document - Content Validation is
        # already heading-anchored and its own local reorder check already
        # allows for a sentence that moved within its own section. A whole-
        # document "is this text anywhere at all" check is too blunt once a
        # document repeats boilerplate under several headings (e.g. the same
        # OSD-menu disclaimer paragraph reprinted before every menu section):
        # it was found to silently drop a genuine loss under one specific
        # heading just because the identical boilerplate still existed under
        # a different, unrelated heading.
        act_page = d.get("actual_page")
        page_idx = (act_page - 1) if isinstance(act_page, int) else None
        if sum(len(s.split()) for s in sents) <= _CAPTION_LIKE_MAX_WORDS and _has_substantial_image_nearby(act_doc, page_idx):
            return "review"  # likely rasterized into Staging's image, not deleted
        if _has_matching_table_row_nearby(act_path, page_idx, sents):
            return "review"  # likely a real table on this side, misread as prose on the other
        return "confirmed"

    if msg == "Added text":
        sents = [s for s in (d.get("actual") or []) if isinstance(s, str) and not s.startswith("... (+")]
        if not sents or _all_non_prose(sents):
            return _DROP
        exp_page = d.get("expected_page")
        page_idx = (exp_page - 1) if isinstance(exp_page, int) else None
        if sum(len(s.split()) for s in sents) <= _CAPTION_LIKE_MAX_WORDS and _has_substantial_image_nearby(exp_doc, page_idx):
            return "review"  # likely rasterized into Production's image, not added
        if _has_matching_table_row_nearby(exp_path, page_idx, sents):
            return "review"  # likely a real table on this side, misread as prose on the other
        return "confirmed"

    if msg == "Changed text":
        wd = d.get("word_diff")
        if wd:
            changed = "".join(
                op["text"]
                for ops in wd
                for op in ops
                if op.get("type") in ("del", "ins") and op.get("text", "").strip()
            )
            if not _norm(changed):  # only whitespace / punctuation moved
                return _DROP
            # The same question the Missing/Added rules ask, asked per fragment:
            # is the text claimed to have been deleted still somewhere in the
            # other document, and vice versa? This is the tell of a sentence
            # BOUNDARY disagreement rather than an edit - one document glues two
            # sentences together (a dropped space after a full stop, a paragraph
            # that reflowed onto the next page) so the diff pairs a short
            # sentence against a long one and calls the remainder deleted, while
            # a reader looking at the two pages sees the same words on both.
            if _fragments_present_elsewhere(wd, exp_idx, act_idx):
                return _DROP
            return "confirmed"
        exp_list = [x for x in (d.get("expected") or []) if isinstance(x, str)]
        act_list = [x for x in (d.get("actual") or []) if isinstance(x, str)]
        if (exp_list or act_list) and _all_non_prose(exp_list) and _all_non_prose(act_list):
            return _DROP  # both sides are diagram / legend fragments
        if _looks_like_index_dump(exp_list) or _looks_like_index_dump(act_list):
            return _DROP  # a table-of-contents / FAQ index swept into the section
        exp, act = " ".join(exp_list), " ".join(act_list)
        if exp and act and act_idx.has_phrase(exp) and exp_idx.has_phrase(act):
            return _DROP  # both wordings exist in both docs => reordering
        return "confirmed"

    if msg == "Text formatting differs":
        exp_s, act_s = d.get("expected_style", ""), d.get("actual_style", "")
        same_family = _font_core(exp_s) == _font_core(act_s)
        e_pt, a_pt = _size_pt(exp_s), _size_pt(act_s)
        same_size = e_pt is None or a_pt is None or abs(e_pt - a_pt) < 1.0
        changed = d.get("changed", "")
        if same_family and same_size:
            return _DROP
        if same_family and "font family" in changed and "size" not in changed:
            return _DROP
        return "review"  # real formatting deltas are worth showing but rarely blocking

    if msg == "Image label missing":
        labels = d.get("missing_labels") or []
        if not labels:
            return _DROP
        if all(_norm(l) in act_idx.norm_words or _norm(l) in act_idx.blob for l in labels):
            return _DROP  # the label text is present elsewhere in Staging
        real = [l for l in labels if _looks_like_real_word(l)]
        if not real:
            return _DROP  # every "missing label" is an OCR fragment, not a word
        # The validator has already OCR-checked the Staging figure's pixels and
        # set its own confidence (review when it could not verify). Two or more
        # real words genuinely gone is worth confirming; a single one stays
        # review.
        return "confirmed" if len(real) >= 2 else "review"

    if msg == "Diagram callout number missing":
        # Already gated to "partial loss only" in the validator; keep as review -
        # OCR of leader-line numbers is not reliable enough to block on.
        return "review"

    if msg in ("Table cell missing", "Table heading missing"):
        toks = d.get("missing_tokens") or []
        exp_cell = " ".join(d.get("expected") or [])
        # The whole expected cell is still present verbatim in Staging => the
        # cell wasn't really lost, the row just paired imperfectly.
        if exp_cell and act_idx.has_phrase(exp_cell):
            return _DROP
        # A SINGLE short missing token that also appears somewhere in Staging is
        # almost always an extraction/OCR hiccup, not a real loss - but two or
        # more missing words is a genuine content difference even if each word
        # exists elsewhere in the manual.
        if len(toks) <= 1 and all(t.casefold() in act_idx.words or _norm(t) in act_idx.blob for t in toks):
            return _DROP
        return "confirmed"

    if msg in ("Table row missing",):
        rows = [r for r in (d.get("expected") or []) if isinstance(r, str)]
        # Drop only when the row's text is STILL PRESENT CONTIGUOUSLY somewhere
        # in Staging - i.e. the row really is there, it was just detected on
        # only one side (a borderless table, a definition list). A "the
        # individual words each appear somewhere in the document" test was here
        # and was far too loose: every ordinary sentence's words scatter across
        # a 13,000-word manual, so it silently dropped genuine lost rows.
        if rows and all(act_idx.has_phrase(r, min_len=10) for r in rows):
            return _DROP
        return "confirmed"

    if msg == "Missing table":
        return "review"

    if msg == "Table columns differ":
        e, a = d.get("expected_columns"), d.get("actual_columns")
        # One side reporting 7+ columns against a small count on the other is
        # pdfplumber reading a diagram / OSD mock-up as a grid, not a real
        # column-count change.
        if isinstance(e, int) and isinstance(a, int) and max(e, a) >= 7 and min(e, a) <= 4:
            return "review"
        return "confirmed"

    if msg in _LOW_CONFIDENCE_MESSAGES:
        return "review"

    # Page-count, broken image, missing image, columns differ, links, encoding of
    # genuine control chars, callout/list swaps - these are structural facts the
    # two PDFs cannot disagree about by accident.
    return "confirmed"


def verify_report(
    report: ValidationReport,
    expected: fitz.Document,
    actual: fitz.Document,
    expected_path: str | None = None,
    actual_path: str | None = None,
) -> None:
    """Re-check every issue against both full documents; drop the demonstrably
    false ones and tag the rest with `confidence` in their details.
    """
    exp_idx = _DocIndex(expected)
    act_idx = _DocIndex(actual)
    for check in report.checks:
        kept: list[Issue] = []
        for issue in check.issues:
            try:
                verdict = _verdict(issue, exp_idx, act_idx, expected, actual, expected_path, actual_path)
            except Exception:
                verdict = "confirmed"  # never lose a finding to a verifier bug
            if verdict is _DROP:
                continue
            # A validator that already marked its own finding "review" knows
            # something the verifier doesn't (it is advisory table/menu cell
            # text, say) - the verifier may still DROP it, but never promote it
            # back to a blocking "confirmed".
            if (issue.details or {}).get("confidence") == "review":
                verdict = "review"
            issue.details = {**(issue.details or {}), "confidence": verdict}
            kept.append(issue)
        check.issues = kept
