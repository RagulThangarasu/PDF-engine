"""Data for the standalone TOC-navigated side-by-side content browser.

`report.html` is the validation findings; this is the other view the user asked
for - pick a heading on the left, read Production and Staging for that section
next to each other on the right, every sentence lined up so a difference is
obvious. It reuses Content Validation's own section matching and sentence
splitting so the two views agree on what "a section" and "a sentence" are.
"""
from __future__ import annotations

import difflib
import html
import os
import re

import fitz

from pdfval.validators import content as _content
from pdfval import ocr as _ocr
from pdfval.report import screenshots as _shots
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import (
    TocEntry,
    extract_section_blocks,
    is_excluded_heading,
    match_toc_entries,
    normalize_block_text,
    normalize_title,
)

_WHOLE_DOC_TITLE = "Whole document"
# A stand-in span is there to show the reader WHERE, not to re-render a
# chapter beside a two-line section.
_FALLBACK_MAX_PAGES = 2


def _section_ends(entries: list[TocEntry]) -> list[tuple[int, float]]:
    """Each entry's END `(page, y)` - the position of the next heading in
    reading order, or the end of the document."""
    order = sorted(range(len(entries)), key=lambda i: (entries[i].page, entries[i].y))
    ends: list[tuple[int, float]] = [(10**9, float("inf"))] * len(entries)
    for place, idx in enumerate(order):
        if place + 1 < len(order):
            nxt = entries[order[place + 1]]
            ends[idx] = (nxt.page, nxt.y)
    return ends


def _heading_band(
    doc: fitz.Document, entry: TocEntry | None, end: tuple[int, float] | None
) -> list[dict]:
    """A geometric stand-in span for a section whose text blocks came back
    empty - a container heading, or a page whose content is all table/figure.
    The snapshot then still shows the reader what is actually printed under
    that heading instead of the column reading "Not in Production"."""
    if entry is None or not (0 <= entry.page < doc.page_count):
        return []
    end_page, end_y = end or (10**9, float("inf"))
    if end_y <= 0:  # the next heading starts at the very top of its page
        end_page -= 1
    last = min(doc.page_count - 1, end_page, entry.page + _FALLBACK_MAX_PAGES - 1)
    out: list[dict] = []
    for p in range(entry.page, max(entry.page, last) + 1):
        rect = doc[p].rect
        y0 = max(rect.y0, entry.y - _SNAP_MARGIN) if p == entry.page else rect.y0
        y1 = min(rect.y1, end_y) if p == end_page else rect.y1
        if y1 - y0 < 6:
            continue
        out.append({"page": p, "bbox": (rect.x0, y0, rect.x1, y1)})
    return out


def _capped_span(blocks: list[dict] | None, max_pages: int = _FALLBACK_MAX_PAGES) -> list[dict]:
    """The first `max_pages` pages of a stand-in span - a fallback is there to
    show the reader WHERE in the other document to look, not to re-render a
    whole chapter beside a two-line section."""
    pages: list[int] = []
    out: list[dict] = []
    for b in blocks or []:
        p = b.get("page")
        if not isinstance(p, int):
            continue
        if p not in pages:
            if len(pages) >= max_pages:
                continue
            pages.append(p)
        out.append(b)
    return out


# Separators a document puts between the entries of a "see also" list of
# section links - a dash, a bullet, a comma - when the extractor hands the
# whole list back as one line.
_HEADING_RUN_SEP_RE = re.compile(r"^[\s\-\u2013\u2014\u2022\u00b7,;:>/]{0,3}")


def _is_heading_title_run(text: str, heading_titles: set[str]) -> bool:
    """True when `text` is nothing but a run of two or more known heading
    titles - a cross-reference list ("Configuring startup and shutdown settings
    -Configuring power off and sleep settings -Setting a power schedule"), or a
    parent title glued to its first child ("投影机概述概述").

    One document lays such a list out as three separate lines, each of which
    the single-title test above recognizes and drops as navigation; the other
    emits all three as one block, which matches no single title and survives -
    so the report showed the whole list as content only Production has.
    """
    nt = normalize_title(text)
    if not nt:
        return False
    titles = sorted((t for t in heading_titles if t), key=len, reverse=True)
    pos = found = 0
    while pos < len(nt):
        pos += _HEADING_RUN_SEP_RE.match(nt[pos:]).end()
        if pos >= len(nt):
            break
        for t in titles:
            if nt.startswith(t, pos):
                pos += len(t)
                found += 1
                break
        else:
            return False
    return found >= 2


def _is_heading_line(text: str, heading_titles: set[str]) -> bool:
    """A sub-heading's own title line, extracted as a line of the parent
    section's body, is navigation not content."""
    nt = normalize_title(text)
    if not nt:
        return False
    if nt in heading_titles:
        return True
    # A bare "目录" / "索引" / "Table of contents" line is a running header the
    # section extractor swept in, never body text.
    if len(nt) <= 24 and is_excluded_heading(nt):
        return True
    return _is_heading_title_run(nt, heading_titles)


def _section_records(section_blocks: list[dict], heading_titles: set[str]) -> list[dict]:
    for b in section_blocks:
        b["text"] = normalize_block_text(b.get("text", ""))
    out = []
    for r in _content._to_sentence_records(section_blocks):
        if _is_heading_line(r["text"], heading_titles):
            continue
        out.append(r)
    return out


def _prose_only(
    blocks: list[dict], tables: "_content._TableBBoxCache", images: "_content._ImageBBoxCache"
) -> list[dict]:
    """Same exclusions Content Validation applies before diffing: a diagram
    callout label ("Release button", "USB peripherals" pointing at a port
    photo) is navigation for a figure, not a sentence, and a table's own
    cells are Table Validation's job - left unfiltered, both show up here as
    broken/duplicated fake "sentences" (a label split across lines reads as
    a sentence fragment, and the same label repeated at each callout arrow
    reads as a duplicate one) that have nothing to do with a genuine content
    difference between the two documents.
    """
    blocks = _content._exclude_table_blocks(blocks, tables)
    blocks = _content._exclude_image_label_blocks(blocks, images)
    return [b for b in blocks if not b.get("in_table")]


def _key(text: str) -> str:
    return _content._cmp_key(text)


def _align(exp: list[str], act: list[str]) -> list[dict]:
    """Line the two sentence lists up. Each row is one of:
      equal   - same sentence on both sides
      change  - a sentence reworded (carries an inline word/letter diff)
      prod    - a sentence only in Production
      stage   - a sentence only in Staging
    """
    matcher = difflib.SequenceMatcher(a=[_key(s) for s in exp], b=[_key(s) for s in act], autojunk=False)
    rows: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # An "equal" run can still pair sentences that differ only in a
            # leading footnote mark ("*: ..."), a wrap hyphen or spacing - the
            # comparison key folds those, the display must not, so show each
            # side's own exact text.
            for ps, as_ in zip(exp[i1:i2], act[j1:j2]):
                rows.append({"op": "equal", "prod": ps, "stage": as_})
        elif tag == "delete":
            for s in exp[i1:i2]:
                rows.append({"op": "prod", "prod": s, "stage": ""})
        elif tag == "insert":
            for s in act[j1:j2]:
                rows.append({"op": "stage", "prod": "", "stage": s})
        else:  # replace
            e, a = exp[i1:i2], act[j1:j2]
            for k in range(max(len(e), len(a))):
                if k < len(e) and k < len(a):
                    # difflib pairs by position inside a replace block; when the
                    # two sentences share almost nothing they are not a reword,
                    # just two unrelated lines that landed opposite each other
                    # (Production's "Useful tools ..." vs Staging's "EPREL
                    # Registration Number:") - show them as their own rows, not
                    # a garbled word-diff.
                    et, at = _tok_set(e[k]), _tok_set(a[k])
                    overlap = len(et & at) / max(len(et | at), 1)
                    if overlap < 0.2:
                        rows.append({"op": "prod", "prod": e[k], "stage": ""})
                        rows.append({"op": "stage", "prod": "", "stage": a[k]})
                    else:
                        rows.append(
                            {
                                "op": "change",
                                "prod": e[k],
                                "stage": a[k],
                                "word_diff": _content._word_diff(e[k], a[k]),
                            }
                        )
                elif k < len(e):
                    rows.append({"op": "prod", "prod": e[k], "stage": ""})
                else:
                    rows.append({"op": "stage", "prod": "", "stage": a[k]})
    return rows


_CJK_CHAR_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")


def _tok_set(text: str) -> set[str]:
    # Words in ANY script, plus every CJK character on its own - a spaceless
    # script has no word tokens for a `[a-z0-9]{3,}` pass to find, which left
    # every reworded Chinese sentence pair looking like two unrelated lines
    # (zero overlap) instead of one change with an inline diff. Restricted to
    # a-z it did exactly the same to Cyrillic and Greek: two Russian renderings
    # of one directive showed as a prod-only row and a stage-only row rather
    # than one reworded line with the difference marked inside it.
    return {
        w for w in _alnum_blob(_key(text or "")).split()
        if len(w) >= 3 or _CJK_CHAR_RE.match(w)
    }


def _alnum_blob(text: str) -> str:
    # Every CJK character is kept but space-isolated so it becomes its own
    # token in the word-set comparisons below. Without this a whole Chinese
    # sentence collapsed to a single unusable token and the "is this really
    # present in the other document" reconciliation never fired for CJK - so
    # relocated Chinese content stayed flagged prod/stage only with an empty
    # opposite column.
    #
    # Everything else is kept as words of LETTERS in any script, not just
    # a-z. Restricted to ASCII, this threw away every Cyrillic, Greek, Thai and
    # accented-Latin word outright, so a multilingual regulatory page - a WEEE
    # notice repeated in 20 languages - had nothing left to reconcile with:
    # "Русский" came back as the empty string on both sides, and a list whose
    # language label the two documents order differently reported the very same
    # word as missing from Production AND added in Staging, three lines apart.
    s = _CJK_CHAR_RE.sub(lambda m: f" {m.group(0)} ", (text or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", s, flags=re.UNICODE)).strip()


def _sig_words(text: str) -> set[str]:
    """Meaningful tokens of `text` for presence checks: Latin words >= 3 chars
    and every CJK character (each is a full morpheme, so length doesn't apply)."""
    return {
        w for w in _alnum_blob(text).split()
        if (len(w) >= 3 or _CJK_CHAR_RE.match(w)) and w not in _RECONCILE_STOPWORDS
    }


def _blob_len(text: str) -> int:
    return len(_alnum_blob(text).split())


def _blocks_blob(blocks: list[dict]) -> str:
    return _alnum_blob(" ".join(normalize_block_text(b.get("text", "")) for b in blocks))


_RECONCILE_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "by", "with", "from", "as", "at", "it", "this", "that", "your", "you",
}


_RUN_MAX_TOKENS = 5   # tokens - above this, match on characters instead (below)
_RUN_MIN_CHARS = 12   # characters - a shorter run proves too little on its own


def _contains_run(blob: str, text: str) -> bool:
    """Is `text` itself printed somewhere in `blob` - its own tokens, in order
    and unbroken?

    Word-for-word for a short fragment, and character-for-character with
    spacing discounted for anything longer. The second form is what settles a
    line the two producers tokenise differently: Production sets a phone number
    as "Cell# +880-18-47052070" plus "or +880-17-14077062" on the next line and
    Staging runs the two together as "...47052070or +880-17...", and a Japanese
    sentence one document wraps mid-word comes back with the break in a
    different place. Order and adjacency make this a far stricter test than the
    word-set ones, so a genuinely dropped paragraph is never swallowed by it.
    """
    frag = _alnum_blob(text)
    if not frag:
        return False
    if len(frag.split()) <= _RUN_MAX_TOKENS and f" {frag} " in f" {blob} ":
        return True
    squashed = frag.replace(" ", "")
    return len(squashed) >= _RUN_MIN_CHARS and squashed in blob.replace(" ", "")


def _reconcile_cross_present(rows: list[dict], exp_full_blob: str, act_full_blob: str) -> None:
    """A short sentence that shows as prod-only / stage-only but whose every
    meaningful word appears in the OTHER document's full section text is not
    missing content - it was extracted differently (Production wraps it narrow
    inside a two-column figure box so the reading order jumbles it, or a
    figure-baked label vs live text). Re-label the row "equal" so it isn't
    flagged or highlighted. Kept tight: <= 14 words and >= 3 distinct
    meaningful words, all of which must be present, so a genuinely dropped
    paragraph is never swallowed.
    """
    exp_words = set(exp_full_blob.split())
    act_words = set(act_full_blob.split())

    def present(text: str, other_words: set[str], other_blob: str) -> bool:
        n = _blob_len(text)
        words = _sig_words(text)
        if words and words <= other_words:
            # a short fragment (the reading order split a figure-box label)
            # needs only its words present; a longer line must be a real
            # sentence-length match, all meaningful words accounted for. CJK
            # counts characters, not words, so its caps are higher.
            return (n <= 5) or (n <= 14 and len(words) >= 3) or (n <= 45 and len(words) >= 6)
        # The word-set test can't see two kinds of real match: a fragment with
        # no word of three characters at all ("4K", "16:9", "Wi-Fi", a spec
        # value), and a line the two producers tokenise differently. Both are
        # settled by looking for the row's own text, verbatim, on the other
        # side - see `_contains_run`.
        return _contains_run(other_blob, text)

    for row in rows:
        if row["op"] == "prod" and row.get("prod") and present(row["prod"], act_words, act_full_blob):
            row["op"] = "equal"
            row["stage"] = row["prod"]
        elif row["op"] == "stage" and row.get("stage") and present(row["stage"], exp_words, exp_full_blob):
            row["op"] = "equal"
            row["prod"] = row["stage"]


def _mark_whole_doc_matches(rows: list[dict], exp_all_blob: str, act_all_blob: str) -> None:
    """A row still prod-only/stage-only after the section/parent-scoped
    reconciliation above may still be genuinely present elsewhere in the
    OTHER document - just filed under a differently-organized heading, not
    within this section's own immediate counterpart. Too broad a check to
    auto-apply (a whole-document match is a much weaker signal than a
    same-section one), so this only FLAGS the row for the "Match content"
    button in the UI to apply on demand, it never changes `op` itself.
    """
    def sig(text: str) -> set[str]:
        return {w for w in _alnum_blob(text).split() if len(w) >= 3 and w not in _RECONCILE_STOPWORDS}

    exp_words = set(exp_all_blob.split())
    act_words = set(act_all_blob.split())

    def candidate(text: str, other_words: set[str]) -> bool:
        words = sig(text)
        n = len(_alnum_blob(text).split())
        return bool(words) and words <= other_words and (n <= 5 or (n <= 14 and len(words) >= 3))

    for row in rows:
        if row["op"] == "prod" and row.get("prod") and candidate(row["prod"], act_words):
            row["content_match_text"] = row["prod"]
        elif row["op"] == "stage" and row.get("stage") and candidate(row["stage"], exp_words):
            row["content_match_text"] = row["stage"]


# A one-sided row is paired with its best match anywhere in the OTHER document
# once the two sentences are this alike - below it, the "closest" sentence is a
# different sentence that merely shares boilerplate, so the row is left alone.
_RELOCATED_EQUAL_RATIO = 0.94   # near-identical: same sentence, moved section -> "=" row
_RELOCATED_CHANGE_RATIO = 0.68  # clearly the same sentence, reworded -> "≠" row with inline diff
_RELOCATED_MIN_LEN = 12         # chars; below this a fuzzy ratio is too noisy to trust


def _relocated_index(sentences: list[str]) -> list[tuple[str, str]]:
    """(key, original) for every distinct sentence, long enough to match on."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for s in sentences:
        k = _key(s)
        if len(k) < _RELOCATED_MIN_LEN or k in seen:
            continue
        seen.add(k)
        out.append((k, s))
    return out


def _best_relocated_match(key: str, index: list[tuple[str, str]]) -> tuple[float, str]:
    best_ratio, best_text = 0.0, ""
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(key)
    for k, original in index:
        if abs(len(k) - len(key)) > max(len(key), len(k)) * 0.5:
            continue
        sm.set_seq1(k)
        if sm.quick_ratio() <= best_ratio:
            continue
        r = sm.ratio()
        if r > best_ratio:
            best_ratio, best_text = r, original
    return best_ratio, best_text


def _fill_relocated_counterparts(
    sections: list[dict], exp_sentences: list[str], act_sentences: list[str]
) -> None:
    """The section-scoped and whole-doc word-set checks above only ever turn a
    row into a bare "=" (no opposite text shown) or flag it for a button. This
    does the last mile the user asked for: for every row still Production-only
    or Staging-only, find that sentence's best match ANYWHERE in the other
    document and, when they are clearly the same sentence, fill the empty
    column - as an "=" row if near-identical, or a "≠" row with the inline
    word diff if it was reworded - so no comparison cell is ever left blank
    for content that genuinely exists on both sides, just under a different
    heading.
    """
    exp_index = _relocated_index(exp_sentences)
    act_index = _relocated_index(act_sentences)

    for sec in sections:
        for row in sec["rows"]:
            op = row["op"]
            if op == "prod" and row.get("prod") and not row.get("stage"):
                text, index, side = row["prod"], act_index, "stage"
            elif op == "stage" and row.get("stage") and not row.get("prod"):
                text, index, side = row["stage"], exp_index, "prod"
            else:
                continue
            key = _key(text)
            if len(key) < _RELOCATED_MIN_LEN:
                continue
            ratio, match = _best_relocated_match(key, index)
            if ratio < _RELOCATED_CHANGE_RATIO or not match:
                continue
            row[side] = match
            row["relocated"] = side  # which column was filled from elsewhere
            if ratio >= _RELOCATED_EQUAL_RATIO:
                row["op"] = "equal"
                row.pop("word_diff", None)
            else:
                row["op"] = "change"
                prod_text = row["prod"] if side == "stage" else match
                stage_text = row["stage"] if side == "prod" else match
                row["prod"], row["stage"] = prod_text, stage_text
                row["word_diff"] = _content._word_diff(prod_text, stage_text)
        sec["counts"] = _counts(sec["rows"])


def _reconcile_ocr_present(
    rows: list[dict],
    exp_words: "_ocr.PageWords", exp_pages: list[int],
    act_words: "_ocr.PageWords", act_pages: list[int],
) -> None:
    """A short sentence that still shows as prod-only/stage-only after the
    text-layer reconciliation above is genuinely missing from the OTHER
    side's text layer - but it may still be visually present there, baked
    into a diagram/screenshot as artwork instead of live text (a callout
    caption is common to render this way). `pdfval.ocr.PageWords` reads what
    Tesseract finds on the rendered page that the text layer doesn't already
    explain - if this row's own significant words are all found there, the
    content isn't actually missing, just not machine-readable on that side.
    Only runs OCR (expensive) when there's at least one row left to check.
    """
    if not any(row["op"] in ("prod", "stage") for row in rows):
        return

    def sig(text: str) -> set[str]:
        return {w for w in _alnum_blob(text).split() if len(w) >= 3 and w not in _RECONCILE_STOPWORDS}

    def artwork_words(words_obj: "_ocr.PageWords", doc, pages: list[int]) -> set[str]:
        out: set[str] = set()
        for p in pages:
            # OCR is the run's slowest step; only pay it for a page that
            # carries a raster image - a screenshot / photo a caption could be
            # baked into. A pure-prose page has nothing for this check to find.
            try:
                if not doc[p].get_images(full=False):
                    continue
            except Exception:
                pass
            for w in words_obj.ocr_words(p):
                nw = _ocr.normalize_word(w[4])
                if nw:
                    out.add(nw)
        return out

    def contained(word: str, other_words: set[str]) -> bool:
        # OCR sometimes merges adjacent words with no rendered space between
        # them (seen with a trademark symbol immediately followed by the next
        # word, e.g. "USB-C\u2122port"), so an exact-token match can miss a
        # significant word that is really there, just glued onto a neighbour.
        if word in other_words:
            return True
        return any(len(aw) > len(word) and word in aw for aw in other_words)

    def present(text: str, other_words: set[str]) -> bool:
        n = len(_alnum_blob(text).split())
        words = sig(text)
        if not words or not other_words or not all(contained(w, other_words) for w in words):
            return False
        return (n <= 5) or (n <= 14 and len(words) >= 3)

    act_artwork = artwork_words(act_words, act_words._doc, act_pages) if act_pages else set()
    exp_artwork = artwork_words(exp_words, exp_words._doc, exp_pages) if exp_pages else set()

    for row in rows:
        if row["op"] == "prod" and row.get("prod") and present(row["prod"], act_artwork):
            row["op"] = "equal"
            row["stage"] = row["prod"]
            row["ocr_note"] = "Present in Staging's artwork/diagram (found via OCR) - not in the text layer, so not a real content difference."
        elif row["op"] == "stage" and row.get("stage") and present(row["stage"], exp_artwork):
            row["op"] = "equal"
            row["prod"] = row["stage"]
            row["ocr_note"] = "Present in Production's artwork/diagram (found via OCR) - not in the text layer, so not a real content difference."


def _row_locations(rows: list[dict], exp_recs: list[dict], act_recs: list[dict]) -> None:
    """Attach `prod_loc` / `stage_loc` = {"page", "bbox"} to each row, looked
    up from the sentence records by comparison key, so a difference can be
    boxed on the page snapshot."""
    exp_loc = {_key(r["text"]): {"page": r.get("page"), "bbox": r.get("bbox")}
               for r in exp_recs if r.get("bbox") is not None}
    act_loc = {_key(r["text"]): {"page": r.get("page"), "bbox": r.get("bbox")}
               for r in act_recs if r.get("bbox") is not None}
    for row in rows:
        if row.get("prod"):
            row["prod_loc"] = exp_loc.get(_key(row["prod"]))
        if row.get("stage"):
            row["stage_loc"] = act_loc.get(_key(row["stage"]))


def _number_label(nums: list[int]) -> str:
    """A compact label for the difference numbers a box carries: "3", "3-6",
    "3-5, 9". Plain ASCII - the badge is drawn with Pillow's built-in face,
    which has no en dash and renders one as a tofu box."""
    parts: list[str] = []
    start = prev = nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is None:
            break
        start = prev = n
    return ", ".join(parts)


def _merge_boxes(boxes: list[dict]) -> list[dict]:
    """Consecutive differences very often anchor to the SAME sentence (a
    paragraph rewritten line by line, or a run of dropped lines all pointing at
    one surviving neighbour on the other side) - stacking a dozen identical
    rectangles there just paints a solid block and hides which numbers are in
    it. Collapse boxes sharing a page and rectangle into one, keeping the bold
    "diff" styling if any of them was a real difference on this side.
    """
    merged: dict[tuple, dict] = {}
    for b in boxes:
        key = (b.get("page"), tuple(round(v, 1) for v in b["bbox"]))
        slot = merged.get(key)
        if slot is None:
            merged[key] = {**b, "numbers": [b["n"]]}
            continue
        slot["numbers"].append(b["n"])
        if b.get("kind") == "diff":
            slot["kind"] = "diff"
    out: list[dict] = []
    for b in merged.values():
        b["label"] = _number_label(sorted(set(b.pop("numbers"))))
        b.pop("n", None)
        out.append(b)
    return out


def _diff_highlight_boxes(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(prod_boxes, stage_boxes) for the snapshots - PAIRED and NUMBERED.

    Every difference is boxed on BOTH sides under the same number, so box 3 in
    the Production snapshot points straight at the spot box 3 marks in the
    Staging one. Whichever side carries the differing text gets the bold red
    box; the side that doesn't gets a thin orange box on the nearest matched
    sentence - exactly where the missing content belongs. Boxing only the side
    that changed left the reader hunting the other document by eye for the
    place a left-hand highlight was talking about.
    """
    prod_boxes: list[dict] = []
    stage_boxes: list[dict] = []

    def nearest(idx: int, side: str) -> dict | None:
        """The closest row above or below that does have a location on `side`."""
        for step in range(1, max(2, len(rows))):
            for j in (idx - step, idx + step):
                if 0 <= j < len(rows):
                    loc = rows[j].get(side)
                    if loc and loc.get("bbox"):
                        return loc
        return None

    number = 0
    for i, row in enumerate(rows):
        if row["op"] == "equal":
            continue
        pl, sl = row.get("prod_loc"), row.get("stage_loc")
        prod_hit = bool(row.get("prod") and pl and pl.get("bbox"))
        stage_hit = bool(row.get("stage") and sl and sl.get("bbox"))
        prod_at = pl if prod_hit else nearest(i, "prod_loc")
        stage_at = sl if stage_hit else nearest(i, "stage_loc")
        if not prod_at and not stage_at:
            continue
        number += 1
        if prod_at:
            prod_boxes.append({**prod_at, "kind": "diff" if prod_hit else "context", "n": number})
        if stage_at:
            stage_boxes.append({**stage_at, "kind": "diff" if stage_hit else "context", "n": number})
    return _merge_boxes(prod_boxes), _merge_boxes(stage_boxes)


def _annotate_callouts(
    rows: list[dict], exp_recs: list[dict], act_recs: list[dict], icon_dropped: list[set[str]]
) -> None:
    """Tag each row that starts a NOTE / TIP / WARNING / CAUTION callout with
    the callout type on each side and whether Content Validation flagged its
    Production icon as missing/flattened in Staging - so the browser shows
    where and how a note's flag differs."""
    exp_c = {_key(r["text"]): r.get("callout") for r in exp_recs if r.get("callout")}
    act_c = {_key(r["text"]): r.get("callout") for r in act_recs if r.get("callout")}
    for row in rows:
        pc = exp_c.get(_key(row["prod"])) if row.get("prod") else None
        sc = act_c.get(_key(row["stage"])) if row.get("stage") else None
        dropped = False
        # Only a callout HEAD row (Staging labels it NOTE:/TIP:/...) is tied to
        # an icon-drop finding, so the marker lands once per callout, not on
        # every body line whose words happen to sit in the finding's blob.
        if icon_dropped and sc:
            rt = _tok_set(row.get("prod") or row.get("stage"))
            if len(rt) >= 3:
                dropped = any(len(rt & ft) / len(rt) >= 0.5 for ft in icon_dropped)
        if pc or sc or dropped:
            row["callout"] = {
                "prod": pc, "stage": sc,
                "icon_dropped": dropped,
                # A note row only for a real divergence: the icon was dropped,
                # or both sides carry a text label and the two disagree.
                "differ": dropped or bool(pc and sc and pc != sc),
            }


_LINK_PAGE_REF_RE = re.compile(r"\bon page\s+\d+\b", re.IGNORECASE)


def _norm_sub(text: str) -> str:
    text = _LINK_PAGE_REF_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def _shared_run(a: str, b: str) -> int:
    """Longest contiguous run of characters common to both strings."""
    if not a or not b:
        return 0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


def _linked_records(doc: fitz.Document, recs: list[dict]) -> dict[str, list[str]]:
    """key(sentence) -> link targets, for every record a hyperlink hotspot
    covers - matched by the phrase the link's own text spells out (precise) or,
    for a broad hotspot (DITA often makes a whole paragraph one link, and
    Production's wrapped-xref rects extract as garbled text), by the link
    rectangle covering a real part of the record's block."""
    page_links: dict[int, list[tuple[str, "fitz.Rect", str]]] = {}
    out: dict[str, list[str]] = {}
    for r in recs:
        p = r.get("page")
        if not isinstance(p, int) or not (0 <= p < doc.page_count):
            continue
        if p not in page_links:
            got: list[tuple[str, "fitz.Rect", str]] = []
            try:
                for l in doc[p].get_links():
                    fr = l.get("from")
                    if fr is None:
                        continue
                    rect = fitz.Rect(fr)
                    label = _norm_sub(doc[p].get_textbox(rect))
                    if l.get("uri"):
                        target = l["uri"]
                    elif isinstance(l.get("page"), int) and l["page"] >= 0:
                        target = f"page {l['page'] + 1}"
                    else:
                        target = "internal link"
                    got.append((label, rect, target))
            except Exception:
                got = []
            page_links[p] = got
        rt = _norm_sub(r["text"])
        if len(rt) < 8:
            continue
        hits = []
        for label, rect, t in page_links[p]:
            # The link's own covered text and this sentence share a long
            # contiguous phrase - robust to the link rect grabbing a few
            # neighbouring words or the xref's "on page N" (already stripped)
            # and to Production's wrapped-xref rects extracting out of order.
            if len(label) >= 12 and _shared_run(label, rt) >= 18:
                hits.append((t, label))
        if hits:
            k = _key(r["text"])
            bucket = out.setdefault(k, [])
            for h in hits:
                if h not in bucket:
                    bucket.append(h)
    return out


# The PDF renders a hyperlink as coloured, underlined text, not as a word with
# a marker beside it - so the browser does the same, and a row reads the way
# the page reads. The phrase the link actually covers is matched back into the
# sentence tolerantly: `get_textbox` on a link rect returns the words with the
# PDF's own line breaks and spacing, which rarely survive sentence
# reconstruction character for character.
_LINK_MIN_PHRASE = 8


# A printed web address, for the last-resort match below.
_URL_TEXT_RE = re.compile(r"(?:https?://|www\.)[^\s,;)\]]+", re.I)


def _link_anchor(text: str, target: str, label: str) -> tuple[int, int] | None:
    """Where this link's clickable text sits in the sentence.

    Three tries, because a link rect's extracted text is the least reliable
    thing in a PDF: the phrase the link covers; the target spelled out as text
    (a printed URL is normally its own link); and any web address in the
    sentence. Without the fallbacks, a document whose link rect extracts
    garbled - which is exactly what the compact Production layout does - loses
    its link in the report and reads as if the link were not there at all.
    """
    label = " ".join((label or "").split())
    if len(label) >= _LINK_MIN_PHRASE:
        at = _find_loose(text, label)
        if at is not None:
            return at
    printed = re.sub(r"^https?://", "", (target or "")).rstrip("/")
    if len(printed) >= _LINK_MIN_PHRASE:
        at = _find_loose(text, printed)
        if at is not None:
            return _snap_to_url(text, *at)
    if (target or "").lower().startswith(("http://", "https://")):
        m = _URL_TEXT_RE.search(text)
        if m:
            return m.start(), m.end()
    return None


def _snap_to_url(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a match to the whole printed web address it lands in, so the
    scheme isn't left sitting outside the link ("Visit https://<a>example.com
    </a>") when the match came from a target with its scheme stripped."""
    for m in _URL_TEXT_RE.finditer(text):
        if m.start() <= start and end <= m.end():
            return m.start(), m.end()
    return start, end


def _link_html(text: str, links: list[tuple[str, str]] | None) -> str | None:
    """`text` with each hyperlinked phrase wrapped in an <a>, HTML-escaped, or
    None when nothing could be matched (the caller then renders plain text)."""
    if not text or not links:
        return None
    spans: list[tuple[int, int, str]] = []
    for target, label in links:
        at = _link_anchor(text, target, label)
        if at is None:
            continue
        start, end = at
        if any(start < e and s < end for s, e, _ in spans):
            continue  # overlaps a link already placed
        spans.append((start, end, target))
    if not spans:
        return None
    spans.sort()
    out: list[str] = []
    at = 0
    for start, end, target in spans:
        out.append(html.escape(text[at:start]))
        out.append(
            f'<a class="pdflink" href="{html.escape(target, quote=True)}"'
            f' title="{html.escape(target, quote=True)}" target="_blank" rel="noopener">'
            f"{html.escape(text[start:end])}</a>"
        )
        at = end
    out.append(html.escape(text[at:]))
    return "".join(out)


def _find_loose(text: str, phrase: str) -> tuple[int, int] | None:
    """Where `phrase` sits in `text`, ignoring how whitespace was broken."""
    lowered = text.lower()
    needle = phrase.lower()
    at = lowered.find(needle)
    if at >= 0:
        return at, at + len(needle)
    # Whitespace-insensitive: walk the sentence skipping spaces the PDF's own
    # line break put in (or took out) of the link's extracted text.
    squashed = [(c, i) for i, c in enumerate(lowered) if not c.isspace()]
    flat = "".join(c for c, _ in squashed)
    target = "".join(c for c in needle if not c.isspace())
    if not target:
        return None
    hit = flat.find(target)
    if hit < 0:
        return None
    return squashed[hit][1], squashed[hit + len(target) - 1][1] + 1


# A contents / FAQ-index line ("... Displaying two sources ... 54", a dot-leader
# row) is navigation, and every one of them is a link - annotating those just
# floods the view, so a row that reads like one is left unmarked.
_TOC_LINE_RE = re.compile(r"(\.\s?){4,}|\s\d{1,3}\s*$|\d{1,3}\s+[A-Z][a-z]")


def _annotate_links(
    rows: list[dict], exp_recs: list[dict], act_recs: list[dict],
    expected: fitz.Document, actual: fitz.Document,
) -> None:
    """Mark every row whose Production and/or Staging sentence carries an
    embedded hyperlink, with the link target(s), so the browser shows where
    Production links text and whether Staging kept the link."""
    exp_lk = _linked_records(expected, exp_recs)
    act_lk = _linked_records(actual, act_recs)
    if not exp_lk and not act_lk:
        return
    for row in rows:
        text = row.get("prod") or row.get("stage") or ""
        if _TOC_LINE_RE.search(text):
            continue
        pl = exp_lk.get(_key(row["prod"])) if row.get("prod") else None
        sl = act_lk.get(_key(row["stage"])) if row.get("stage") else None
        if pl or sl:
            row["link"] = {
                "prod": bool(pl), "stage": bool(sl),
                "prod_targets": [t for t, _ in (pl or [])],
                "stage_targets": [t for t, _ in (sl or [])],
                "differ": bool(pl) != bool(sl),
            }
            # The row's own text with the linked phrase marked up the way the
            # PDF prints it, so the comparison looks like the page.
            prod_html = _link_html(row.get("prod") or "", pl)
            stage_html = _link_html(row.get("stage") or "", sl)
            if prod_html:
                row["prod_html"] = prod_html
            if stage_html:
                row["stage_html"] = stage_html


# A contents / FAQ-index entry: ends in a bare page number, is a bare question,
# or packs "Heading 24 Heading 36" runs together. A long one-sided run of these
# is a navigation list the section extractor swept in (Production's "Product
# support" page lists every FAQ with its page); comparing it line by line is
# noise, so the run is dropped and summarised.
_INDEX_ENTRY_RE = re.compile(r"\b\d{1,3}\s*$|^\s*[A-Z].*\?\s*$|\b\d{1,3}\s+[A-Z][a-z]|\)\s+\d{1,3}\b")


def _strip_index_runs(rows: list[dict]) -> tuple[list[dict], int]:
    keep: list[dict] = []
    i, dropped = 0, 0
    while i < len(rows):
        r = rows[i]
        if r["op"] in ("prod", "stage") and not r.get("callout"):
            j = i
            while j < len(rows) and rows[j]["op"] == r["op"] and not rows[j].get("callout"):
                j += 1
            run = rows[i:j]
            if len(run) >= 4:
                idx_like = sum(1 for x in run if _INDEX_ENTRY_RE.search(x.get(r["op"]) or ""))
                if idx_like / len(run) >= 0.6:
                    dropped += len(run)
                    i = j
                    continue
        keep.append(r)
        i += 1
    return keep, dropped


def _is_bare_callout_label(text: str) -> bool:
    """The whole "sentence" is just a callout word - "NOTE", "警告", "提示:".
    One document extracts the NOTE/TIP/WARNING label as its own block, the
    other glues it to the note body, so it lands as a one-sided row on every
    callout. It is formatting, not content (the callout label/icon check
    covers real label differences), so a bare-label row is dropped here.
    """
    core = re.sub(r"[\s:：.。、,，]+$", "", (text or "").strip())
    return bool(core) and len(core) <= 12 and _content._match_callout_label(core + ":") is not None


def _drop_bare_callout_label_rows(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if not (
            r["op"] in ("prod", "stage")
            and _is_bare_callout_label(r.get("prod") or r.get("stage") or "")
        )
    ]


def _one_sided_rows(own: list[str], parent_other: list[str], own_side: str) -> list[dict]:
    """Rows for a section bookmarked in only one document. Every sentence that
    is genuinely under this heading is shown; each is marked "=" when the same
    sentence also exists in the other document (under the parent heading) or
    "prod only"/"stage only" when it does not. Unlike `_align`, this never
    pairs one of these sentences against an unrelated parent sentence - the
    parent body is only consulted as a presence check.
    """
    other_keys = {_key(s) for s in parent_other}
    other_by_key = {_key(s): s for s in parent_other}
    rows: list[dict] = []
    for s in own:
        k = _key(s)
        if k in other_keys:
            counterpart = other_by_key[k]
            prod, stage = (s, counterpart) if own_side == "prod" else (counterpart, s)
            rows.append({"op": "equal", "prod": prod, "stage": stage})
        elif own_side == "prod":
            rows.append({"op": "prod", "prod": s, "stage": ""})
        else:
            rows.append({"op": "stage", "prod": "", "stage": s})
    return rows


def _counts(rows: list[dict]) -> dict:
    c = {"equal": 0, "change": 0, "prod": 0, "stage": 0}
    for r in rows:
        c[r["op"]] = c.get(r["op"], 0) + 1
    c["diff"] = c["change"] + c["prod"] + c["stage"]
    return c


# A hard cap on how many pages' worth of HTML a single section embeds - a
# section is normally 1-3 pages; without a cap, a mis-detected/huge section
# (e.g. no next heading found) could pull in the rest of the document and
# balloon sections.html to an unusable size.
_MAX_RAW_HTML_PAGES = 8
_Y_MARGIN = 3.0  # points of slack at a section's top/bottom clip edge


def _dominant_font_size(dict_blocks: list) -> float:
    import collections

    sizes: list[float] = []
    for b in dict_blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    sizes.append(round(span.get("size", 0), 1))
    return collections.Counter(sizes).most_common(1)[0][0] if sizes else 10.0


def _heading_tag(max_size: float, body_size: float) -> str:
    if body_size <= 0:
        return "p"
    ratio = max_size / body_size
    if ratio >= 1.8:
        return "h1"
    if ratio >= 1.4:
        return "h2"
    if ratio >= 1.15:
        return "h3"
    return "p"


def _page_links(page: "fitz.Page") -> list[dict]:
    """Every real hyperlink hotspot on the page - external URL links AND
    internal same-document page-jump links (PyMuPDF's `get_links()` only
    sets `uri` for the former; an internal GOTO link only carries a `page`
    field, no `uri` at all). The row-level link-diff feature (`_linked_records`)
    already treats both as real links - the HTML reconstruction was only
    keeping `uri` links, silently rendering an internal cross-reference as
    plain, unlinked text even though the PDF genuinely hyperlinks it.
    """
    out = []
    for l in page.get_links():
        fr = l.get("from")
        if fr is None:
            continue
        if l.get("uri"):
            out.append({"from": fr, "uri": l["uri"]})
        elif isinstance(l.get("page"), int) and l["page"] >= 0:
            out.append({"from": fr, "uri": f"#page-{l['page'] + 1}"})
    return out


def _link_uri_at(bbox, links: list) -> str | None:
    rect = fitz.Rect(bbox)
    for link in links:
        if link.get("uri") and rect.intersects(link["from"]):
            return link["uri"]
    return None


def _render_text_block(block: dict, body_size: float, links: list) -> str:
    import html as _h

    line_htmls: list[str] = []
    max_size = 0.0
    for line in block.get("lines", []):
        parts: list[str] = []
        for span in line.get("spans", []):
            text = span.get("text", "")
            if not text:
                continue
            max_size = max(max_size, span.get("size", body_size))
            frag = _h.escape(text)
            uri = _link_uri_at(span.get("bbox", block.get("bbox")), links)
            if uri:
                frag = f'<a href="{_h.escape(uri, quote=True)}">{frag}</a>'
            if span.get("flags", 0) & (1 << 4):
                frag = f"<b>{frag}</b>"
            parts.append(frag)
        if parts:
            line_htmls.append("".join(parts))
    text_html = " ".join(line_htmls).strip()
    if not text_html:
        return ""
    return f"<{_heading_tag(max_size, body_size)}>{text_html}</{_heading_tag(max_size, body_size)}>"


_MAX_IMAGE_BYTES = 90_000  # keep line-art diagrams inline; a bigger photo is linked out by note
_MIN_IMAGE_DISPLAY_PT = 48.0  # smaller ON THE PAGE than this is an icon/glyph, not a figure


def _render_image_block(block: dict) -> str:
    import base64

    data = block.get("image")
    if not data:
        return ""
    bb = block.get("bbox", (0, 0, 0, 0))
    disp_w, disp_h = bb[2] - bb[0], bb[3] - bb[1]
    if disp_w < _MIN_IMAGE_DISPLAY_PT or disp_h < _MIN_IMAGE_DISPLAY_PT:
        return ""  # a menu glyph / bullet icon, not a real figure - and there are hundreds
    mime = f"image/{block.get('ext', 'png')}"
    if len(data) > _MAX_IMAGE_BYTES:
        shrunk = _shrink_image(data)
        if not shrunk or len(shrunk) > _MAX_IMAGE_BYTES:
            return '<p style="color:#9aa3ad;font-style:italic">[image in the PDF here — too large to embed]</p>'
        data, mime = shrunk, "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img src="data:{mime};base64,{b64}" alt="figure from PDF">'


def _shrink_image(data: bytes) -> bytes | None:
    try:
        import io

        from PIL import Image

        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((900, 900))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=72)
        return buf.getvalue()
    except Exception:
        return None


def _render_table_html(rows: list) -> str:
    import html as _h

    if not rows:
        return ""
    out = ["<table>"]
    for row in rows:
        out.append("<tr>")
        for cell in row or []:
            out.append(f"<td>{_h.escape((cell or '').strip()).replace(chr(10), '<br>')}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


_HTML_STYLE = (
    "<style>body{font-family:-apple-system,Arial,sans-serif;color:#1a1a1a;margin:0;"
    "padding:12px;line-height:1.5;font-size:13px}"
    "h1,h2,h3,h4{font-weight:600;margin:.9em 0 .35em;line-height:1.3}"
    "h1{font-size:1.4em}h2{font-size:1.22em}h3{font-size:1.08em}h4{font-size:1em}"
    "p{margin:.45em 0}a{color:#2d6cdf}img{max-width:100%;height:auto;margin:.5em 0;border:1px solid #e6e8ec}"
    "table{border-collapse:collapse;margin:.7em 0;font-size:.95em}"
    "td,th{border:1px solid #c8ccd2;padding:3px 7px;vertical-align:top}"
    ".pg{border-bottom:1px dashed #d5d8dc;padding-bottom:10px;margin-bottom:12px}"
    ".pg:last-child{border:0}.pg::before{content:'PDF page ' attr(data-page);display:block;"
    "font-size:10px;color:#9aa3ad;margin-bottom:4px}</style>"
)


def _section_html(doc: fitz.Document, span_blocks: list[dict], pdf_path: str | None) -> str:
    """Exactly this section's content, page by page, rebuilt from PyMuPDF's
    clipped `dict` extraction: real heading tags (from font size), paragraphs,
    bold runs, hyperlinks, embedded images, and any detected table kept as a
    real <table> at its place in reading order.

    Clipped per page to the y-band the section's OWN blocks occupy - all of
    them, tables and figure captions included, not just the prose sentences -
    so a table or an image inside the section still falls inside the clip and
    is rendered. (`get_text("xhtml", clip=...)` silently ignores the clip in
    this PyMuPDF build; `dict`/`blocks` honour it, so the HTML is rebuilt from
    the dict rather than taken from xhtml.)
    """
    per_page: dict[int, list[tuple[float, float]]] = {}
    for r in span_blocks:
        p, bb = r.get("page"), r.get("bbox")
        if isinstance(p, int) and bb:
            per_page.setdefault(p, []).append((bb[1], bb[3]))
    if not per_page:
        return ""

    pages = sorted(per_page)
    pages = [p for p in pages if pages[0] <= p <= pages[0] + _MAX_RAW_HTML_PAGES - 1]
    body: list[str] = []
    for p in pages:
        if not (0 <= p < doc.page_count):
            continue
        page = doc[p]
        pr = page.rect
        y0 = max(pr.y0, min(y for y, _ in per_page[p]) - _Y_MARGIN)
        y1 = min(pr.y1, max(y for _, y in per_page[p]) + _Y_MARGIN)
        clip = fitz.Rect(pr.x0, y0, pr.x1, y1)
        if clip.is_empty or clip.height < 4:
            continue
        try:
            data = page.get_text("dict", clip=clip)
        except Exception:
            continue
        links = _page_links(page)
        blocks = data.get("blocks", [])
        body_size = _dominant_font_size(blocks)

        tables = []
        if pdf_path:
            try:
                from pdfval.extractor import get_tables

                tables = [
                    t for t in get_tables(pdf_path, p, page.parent)
                    if t["bbox"][3] > clip.y0 and t["bbox"][1] < clip.y1
                ]
            except Exception:
                tables = []
        table_bboxes = [t["bbox"] for t in tables]

        pieces: list[tuple[float, float, str]] = []
        for b in blocks:
            bb = b.get("bbox", (0, 0, 0, 0))
            if table_bboxes and _content._in_any_table(bb, table_bboxes):
                continue
            if b.get("type") == 1:
                frag = _render_image_block(b)
            else:
                frag = _render_text_block(b, body_size, links)
            if frag:
                pieces.append((round(bb[1]), bb[0], frag))
        for t in tables:
            frag = _render_table_html(t.get("rows") or [])
            if frag:
                pieces.append((round(t["bbox"][1]), t["bbox"][0], frag))

        pieces.sort(key=lambda e: (e[0], e[1]))
        if pieces:
            body.append(
                f'<div class="pg" data-page="{p + 1}">' + "".join(x for _, _, x in pieces) + "</div>"
            )

    if not body:
        return ""
    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8">' + _HTML_STYLE
        + "</head><body>" + "".join(body) + "</body></html>"
    )


# --- exact page snapshots -------------------------------------------------
# Rebuilding a section's HTML from the text layer always loses something -
# multi-column flow, list indents, where an image sits relative to its
# caption, the real fonts. For the "is Staging's layout faithful to
# Production" question the only answer that can't be wrong is the pixels
# themselves, so each section also carries a rendered image of exactly its
# own region, clipped out of the page, Production beside Staging.
_SNAP_DPI = 150
_SNAP_MARGIN = 6.0  # points of slack above/below the section's block span
_SNAP_MAX_PAGES = _MAX_RAW_HTML_PAGES
_SNAP_MAX_BYTES = 240_000  # re-encode to JPEG past this so the run dir stays sane


def _pix_to_jpeg(pix) -> bytes | None:
    try:
        import io

        from PIL import Image

        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if max(im.size) > 1500:
            im.thumbnail((1500, 1500))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def _draw_boxes(pix, clip: "fitz.Rect", scale: float, boxes: list[dict]) -> bytes | None:
    """Overlay highlight rectangles on a rendered page band, in the same visual
    language the validation report uses (`pdfval.report.screenshots`): a bold
    red box on the sentence that differs or was dropped, a thin orange box on
    the place in THIS document where the other one's content belongs, and on
    both the difference's number - so a highlight on the left always has a
    visible partner on the right and the two columns can be read together.
    """
    try:
        import io

        from PIL import Image, ImageDraw
    except Exception:
        return None
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    for b in boxes:
        bx0, by0, bx1, by1 = b["bbox"]
        rect = (
            (bx0 - clip.x0) * scale - 3,
            (by0 - clip.y0) * scale - 2,
            (bx1 - clip.x0) * scale + 3,
            (by1 - clip.y0) * scale + 2,
        )
        _shots.draw_highlight(d, im, rect, b.get("kind") or _shots.KIND_DIFF, b.get("label"))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _png_to_jpeg(data: bytes) -> bytes | None:
    try:
        import io

        from PIL import Image

        im = Image.open(io.BytesIO(data)).convert("RGB")
        if max(im.size) > 1500:
            im.thumbnail((1500, 1500))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def _section_snapshots(
    doc: fitz.Document, span_blocks: list[dict], out_dir: str, prefix: str,
    boxes: list[dict] | None = None,
) -> list[dict]:
    """Render exactly this section's page region(s) to image files in
    `out_dir`. One image per page the section touches, each clipped to the
    y-band the section's own blocks occupy (full page width), with any
    `boxes` ({"page", "bbox", "kind"}) drawn on so a difference is obvious at
    a glance. Returns `[{"page": n, "src": "sections/<file>", "w", "h"}, ...]`.
    """
    per_page: dict[int, list[tuple[float, float]]] = {}
    for r in span_blocks:
        p, bb = r.get("page"), r.get("bbox")
        if isinstance(p, int) and bb:
            per_page.setdefault(p, []).append((bb[1], bb[3]))
    if not per_page:
        return []

    have = sorted(per_page)
    lo = have[0]
    hi = min(have[-1], lo + _SNAP_MAX_PAGES - 1)
    # Fill in a page that has no compared blocks of its own but sits between
    # two that do, and is adjacent to one of them - the section runs straight
    # through it (e.g. Production's "Product support" spilling across a whole
    # "Q&A index" page that text-comparison filtered out). Render it in full so
    # the snapshot shows the real, continuous page range. A far-flung page
    # (a scattered one-sided counterpart match) is left out.
    pages = [p for p in range(lo, hi + 1)
             if p in per_page or (p - 1) in per_page or (p + 1) in per_page]
    mat = fitz.Matrix(_SNAP_DPI / 72, _SNAP_DPI / 72)
    shots: list[dict] = []
    for p in pages:
        if not (0 <= p < doc.page_count):
            continue
        page = doc[p]
        pr = page.rect
        ys = per_page.get(p)
        if ys:
            y0 = max(pr.y0, min(y for y, _ in ys) - _SNAP_MARGIN)
            y1 = min(pr.y1, max(y for _, y in ys) + _SNAP_MARGIN)
        else:
            y0, y1 = pr.y0, pr.y1  # gap page - show it whole
        clip = fitz.Rect(pr.x0, y0, pr.x1, y1)
        if clip.is_empty or clip.height < 6:
            continue
        try:
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        except Exception:
            continue
        page_boxes = [
            b for b in (boxes or [])
            if b.get("page") == p and b.get("bbox")
            and b["bbox"][3] > clip.y0 - 4 and b["bbox"][1] < clip.y1 + 4
        ]
        data = _draw_boxes(pix, clip, _SNAP_DPI / 72, page_boxes) if page_boxes else None
        ext = "png"
        if data is None:
            data = pix.tobytes("png")
        if len(data) > _SNAP_MAX_BYTES:
            j = _png_to_jpeg(data) if page_boxes else _pix_to_jpeg(pix)
            if j:
                data, ext = j, "jpg"
        fname = f"{prefix}_p{p + 1}.{ext}"
        try:
            with open(f"{out_dir}/{fname}", "wb") as f:
                f.write(data)
        except OSError:
            continue
        shots.append({"page": p + 1, "src": f"sections/{fname}", "w": pix.width, "h": pix.height})
    return shots


_IMAGE_NOTE_TEMPLATES = {
    "Image label missing": "🖼 Image label missing — Production labels a figure “{labels}”; that text is not on the matching Staging figure.",
    "Diagram callouts stripped from figure": "🖼 Diagram callouts stripped — Production’s figure carries numbered callouts that Staging’s copy of it does not.",
    "Diagram callout number missing": "🖼 Diagram callout number missing — a leader-line number on Production’s figure is absent from Staging’s.",
    "Image missing": "🖼 Image missing — a figure in this Production section has no counterpart anywhere in the Staging section.",
    "Image content differs": "🖼 Image content differs — the figure in this spot was replaced or its artwork changed in Staging.",
    "Image alignment changed": "🖼 Image alignment changed — a figure sits left/centre/right differently in Staging.",
    "Image size changed": "🖼 Image size changed — a figure is rendered at a materially different size in Staging.",
    "Image width changed": "🖼 Image width changed — a figure is rendered materially wider/narrower in Staging.",
    "Broken image": "🖼 Broken image — a figure in Staging failed to render.",
    "Image highlight box missing": "🖼 Highlight box missing — a callout box drawn on Production’s figure is absent from Staging’s.",
}


def _image_notes_by_heading(findings: list[dict] | None) -> dict[str, list[dict]]:
    """Group the report's confirmed image findings by section heading, as short
    notes for the section browser (the browser compares prose; a figure-level
    difference otherwise never surfaces there)."""
    out: dict[str, list[dict]] = {}
    for f in findings or []:
        msg = f.get("message")
        d = f.get("details") or {}
        if d.get("confidence") == "review":
            continue
        tmpl = _IMAGE_NOTE_TEMPLATES.get(msg)
        heading = d.get("heading")
        if not tmpl or not heading:
            continue
        labels = ", ".join(d.get("missing_labels") or []) or "…"
        out.setdefault(heading, []).append({
            "text": tmpl.format(labels=labels),
            "prod_screenshot": d.get("prod_screenshot"),
            "stage_screenshot": d.get("stage_screenshot"),
        })
    return out


# One short line per kind of table difference, shown under the section in the
# browser (which otherwise only diffs prose) and counted in the "▦" nav tag.
_TABLE_NOTE_TEMPLATES = {
    "Table columns differ":
        "▦ Column count differs — {expected_columns} columns in Production, {actual_columns} in Staging (a column was added, dropped, or split/merged).",
    "Table column layout differs":
        "▦ Column layout differs — a column was widened/narrowed in Staging (largest edge shift {largest_column_shift}), so the table reads differently.",
    "Table cell layout differs":
        "▦ Cells merged / split differently — {merge_detail}",
    "Table split across pages":
        "▦ Table breaks across pages differently — {reason}.",
    "Table header row not repeated on continuation page":
        "▦ Header row not repeated where this table continues onto the next page in Staging.",
    "Table row missing":
        "▦ A row present in Production’s table is missing from Staging’s.",
    "Table heading missing":
        "▦ A header cell present in Production is missing from Staging’s table.",
    "Table cell missing":
        "▦ A cell’s content present in Production is missing from Staging’s table.",
    "Table breaking the margins":
        "▦ Table runs outside the page text area in Staging.",
}
_TABLE_MESSAGE_ORDER = list(_TABLE_NOTE_TEMPLATES)


class _SafeFmt(dict):
    def __missing__(self, key: str) -> str:  # noqa: D401
        return "…"


def _table_merge_detail(d: dict) -> str:
    lost = d.get("merge_missing_in_stage")
    gained = d.get("merge_added_in_stage")
    parts = []
    if lost:
        parts.append(f"Production merges cells across {', '.join(lost)} that Staging keeps as separate columns")
    if gained:
        parts.append(f"Staging merges cells across {', '.join(gained)} that Production keeps as separate columns")
    if parts:
        return "row {}: {}.".format(d.get("row", "?"), "; ".join(parts))
    return "row {}: Production spans {}; Staging spans {}.".format(
        d.get("row", "?"), d.get("expected_layout", "?"), d.get("actual_layout", "?")
    )


def _table_notes_by_heading(findings: list[dict] | None) -> dict[str, list[dict]]:
    """Group every table finding by section heading, one collapsed note per
    distinct difference (repeats of the same kind under one heading carry a
    ×N count). Unlike image notes, low-confidence ("review") findings are kept
    - a column-layout / page-break difference is exactly what the user wants
    surfaced here - but flagged so the UI can show them as "likely"."""
    grouped: dict[str, dict[str, dict]] = {}
    for f in findings or []:
        msg = f.get("message")
        d = f.get("details") or {}
        tmpl = _TABLE_NOTE_TEMPLATES.get(msg)
        heading = d.get("heading")
        if not tmpl or not heading:
            continue
        try:
            text = tmpl.format_map(_SafeFmt({**d, "merge_detail": _table_merge_detail(d)}))
        except Exception:
            text = tmpl.split(" — ")[0]
        bucket = grouped.setdefault(heading, {})
        note = bucket.get(text)
        if note:
            note["count"] += 1
        else:
            bucket[text] = {
                "text": text,
                "count": 1,
                "review": d.get("confidence") == "review",
                "prod_screenshot": d.get("prod_screenshot"),
                "stage_screenshot": d.get("stage_screenshot"),
                "_order": _TABLE_MESSAGE_ORDER.index(msg) if msg in _TABLE_MESSAGE_ORDER else 99,
            }
    out: dict[str, list[dict]] = {}
    for heading, bucket in grouped.items():
        notes = sorted(bucket.values(), key=lambda n: n["_order"])
        for n in notes:
            if n["count"] > 1:
                n["text"] = f"{n['text']}  (×{n['count']})"
            n.pop("_order", None)
        out[heading] = notes
    return out


_LIST_MARKER_MSG = "List marker changed"


def _list_notes_by_heading(findings: list[dict] | None) -> dict[str, list[dict]]:
    """Group the list-marker findings by section heading.

    Shown in RED, unlike the amber formatting notes: a procedure whose steps
    are numbered 1,2,3 in Production and lettered a,b,c in Staging is not a
    styling nicety - every cross-reference to "step 3" in the surrounding text
    now points at nothing, and the two documents' instructions no longer read
    the same. The section's own text diff can't show it, because the marker is
    stripped from (or drawn outside) the sentence it belongs to.
    """
    grouped: dict[str, dict[str, dict]] = {}
    for f in findings or []:
        if f.get("message") != _LIST_MARKER_MSG:
            continue
        d = f.get("details") or {}
        heading = d.get("heading")
        if not heading:
            continue
        texts = d.get("text") or []
        quote = _one_line(texts[0]) if texts else ""
        text = (
            (d.get("changed") or "the list marker style changed")
            + (f': first item “{quote}”' if quote else "")
            + "."
        )
        bucket = grouped.setdefault(heading, {})
        note = bucket.get(text)
        if note:
            note["count"] += 1
        else:
            bucket[text] = {
                "text": text,
                "count": 1,
                "items": d.get("items") or 1,
                "prod_markers": d.get("expected_marker") or "",
                "stage_markers": d.get("actual_marker") or "",
                "prod_screenshot": d.get("prod_screenshot"),
                "stage_screenshot": d.get("stage_screenshot"),
            }
    return {heading: list(bucket.values()) for heading, bucket in grouped.items()}


_FMT_EMPHASIS_MSG = "Bold or italic emphasis removed"
_FMT_ENCODING_MSGS = ("Text encoding regression in Staging", "Possible text encoding issue")


def _one_line(text: str, n: int = 60) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


# The label shown on the finding, identical in the left nav chip and on the
# section note - "Bold", "Italic" or "Encoded".
_FMT_LABELS = {"bold": "Bold", "italic": "Italic", "encoded": "Encoded"}
_FMT_KIND_ORDER = {"bold": 0, "italic": 1, "encoded": 2}


def _format_notes_by_heading(findings: list[dict] | None) -> dict[str, list[dict]]:
    """Group the bold/italic-emphasis and text-encoding findings by section
    heading. Each note carries `kind` ("bold" / "italic" / "encoded") and the
    matching `label`, so the section note and the left-nav chip use the same
    name for the same issue."""
    grouped: dict[str, dict[str, dict]] = {}
    for f in findings or []:
        msg = f.get("message")
        d = f.get("details") or {}
        heading = d.get("heading")
        if not heading:
            continue
        texts = d.get("text") or []
        if isinstance(texts, str):
            texts = [texts]
        quote = _one_line(texts[0]) if texts else ""
        if msg == _FMT_EMPHASIS_MSG:
            emp = (d.get("emphasis") or "bold").lower()
            kind = "italic" if "italic" in emp and "bold" not in emp else "bold"
            pct = d.get("production_bold_pct") if kind == "bold" else d.get("production_italic_pct")
            text = (
                f"dropped in Staging"
                + (f" — Production sets {pct} of the line {kind}" if pct else "")
                + (f': “{quote}”' if quote else "") + "."
            )
        elif msg in _FMT_ENCODING_MSGS:
            kind = "encoded"
            chars = ", ".join(d.get("characters") or []) or "an un-decodable character"
            text = (
                f"character in Staging — {chars} where Production has readable text"
                + (f': “{quote}”' if quote else "") + "."
            )
        else:
            continue
        bucket = grouped.setdefault(heading, {})
        note = bucket.get(text)
        if note:
            note["count"] += 1
        else:
            bucket[text] = {
                "text": text,
                "count": 1,
                "kind": kind,
                "label": _FMT_LABELS[kind],
                "prod_screenshot": d.get("prod_screenshot"),
                "stage_screenshot": d.get("stage_screenshot"),
            }
    out: dict[str, list[dict]] = {}
    for heading, bucket in grouped.items():
        notes = sorted(bucket.values(), key=lambda n: (_FMT_KIND_ORDER[n["kind"]], n["text"]))
        for n in notes:
            if n["count"] > 1:
                n["text"] = f"{n['text']}  (×{n['count']})"
        out[heading] = notes
    return out


def _format_kind_chips(notes: list[dict]) -> list[dict]:
    """One {label, kind, count} per distinct issue type in a section, for the
    left-nav chips - so a section with 6 bold + 2 encoded rows shows
    "Bold 6" and "Encoded 2", the same labels the section notes carry."""
    counts: dict[str, int] = {}
    for n in notes:
        counts[n["kind"]] = counts.get(n["kind"], 0) + n.get("count", 1)
    return [
        {"kind": k, "label": _FMT_LABELS[k], "count": counts[k]}
        for k in sorted(counts, key=lambda k: _FMT_KIND_ORDER[k])
    ]


def build_section_comparison(
    expected: fitz.Document,
    actual: fitz.Document,
    expected_path: str | None = None,
    actual_path: str | None = None,
    output_dir: str | None = None,
    callout_icon_findings: list[dict] | None = None,
    image_findings: list[dict] | None = None,
    table_findings: list[dict] | None = None,
    format_findings: list[dict] | None = None,
    list_findings: list[dict] | None = None,
) -> list[dict]:
    """One entry per TOC heading, in document order, each with the aligned
    Production/Staging content for that section.

    Sections are bounded ONLY by headings that exist on BOTH sides - the same
    rule Content Validation uses - so a sub-heading that only one document
    bookmarks doesn't cut its section short and misalign the two columns.
    Headings unique to one document are still listed (so the reader sees they
    exist) but their body is shown under their parent matched section.
    """
    if not expected.page_count or not actual.page_count:
        return []
    exp_tables = _content._TableBBoxCache(expected_path)
    act_tables = _content._TableBBoxCache(actual_path)
    exp_images = _content._ImageBBoxCache(expected)
    act_images = _content._ImageBBoxCache(actual)

    # The same section list every check in the engine anchors on: real
    # bookmarks where there are any, printed headings detected by style where
    # there aren't, and each side given any heading the other has that is
    # genuinely printed here too - so both columns compare the SAME section
    # instead of one of them going empty. See `pdfval.validators.headings`.
    exp_entries, act_entries = resolve_entries(expected, actual)
    if not exp_entries or not act_entries:
        # One side offers nothing to navigate by even now: compare the two
        # documents whole rather than handing back an empty browser - a run
        # always has to validate something.
        exp_entries = [TocEntry(level=1, title=_WHOLE_DOC_TITLE, page=0, y=0.0)]
        act_entries = [TocEntry(level=1, title=_WHOLE_DOC_TITLE, page=0, y=0.0)]
    exp_ends = _section_ends(exp_entries)
    act_ends = _section_ends(act_entries)

    # Include the excluded headings (a printed contents / Q&A index) so the
    # browser still LISTS them in the nav with their match status - it just
    # won't diff their listing text line by line (that is the TOC Comparison
    # report's job). `_is_toc_listing_entry` below spots which ones to stub.
    matches = match_toc_entries(exp_entries, act_entries, include_excluded=True)
    # Each TOC entry gets exactly its own body - bounded by the NEXT heading of
    # any kind, so a heading unique to one document maps to just its own
    # content (that sub-section then shows as its own "stage only" / "prod
    # only" entry right below in the list) and nothing is double-counted.
    exp_sections = extract_section_blocks(expected, exp_entries)
    act_sections = extract_section_blocks(actual, act_entries)
    # BUT a genuinely matched pair's own body must be bounded ONLY by the next
    # MUTUALLY matched heading, or a sub-heading bookmarked in just one
    # document (e.g. Staging auto-bookmarking every numbered step - "1.
    # Prepare the monitor and area.", "2. Remove the back cover." - under a
    # step-by-step procedure Production keeps as one plain section) cuts that
    # side's body off right where its own extra bookmark begins, making a
    # heading that's still fully present in both documents look like Staging
    # deleted everything under it. Content Validation already learned this
    # the hard way - reuse its exact fix here instead of re-deriving it.
    matched = [m for m in matches if m.expected_index is not None and m.actual_index is not None]
    exp_matched_entries = [exp_entries[m.expected_index] for m in matched]
    act_matched_entries = [act_entries[m.actual_index] for m in matched]
    exp_sections_wide = extract_section_blocks(expected, exp_matched_entries)
    act_sections_wide = extract_section_blocks(actual, act_matched_entries)
    exp_wide_pos = {m.expected_index: i for i, m in enumerate(matched)}
    act_wide_pos = {m.actual_index: i for i, m in enumerate(matched)}
    # The nearest mutually-matched heading before and after each entry in the
    # list. A section with nothing of its own on one side borrows that
    # neighbour's region for its snapshot, so the column is never left blank -
    # looking FORWARD as well as back matters for the headings that come before
    # the document's first matched one (a cover page, a title block).
    prev_matched: list[int | None] = [None] * len(matches)
    next_matched: list[int | None] = [None] * len(matches)
    seen: int | None = None
    for i, m in enumerate(matches):
        prev_matched[i] = seen
        if m.expected_index is not None and m.actual_index is not None:
            seen = i
    seen = None
    for i in range(len(matches) - 1, -1, -1):
        next_matched[i] = seen
        if matches[i].expected_index is not None and matches[i].actual_index is not None:
            seen = i

    def counterpart_region(side: str, at: int) -> tuple[list[dict], str | None]:
        """The nearest matched section's own region on `side` - what to show in
        a column that has no content of its own for this heading, with the
        heading it belongs to so the note can say where the reader is looking.
        """
        for j in (prev_matched[at], next_matched[at]):
            if j is None:
                continue
            m = matches[j]
            if side == "prod":
                span = _capped_span(exp_sections_wide[exp_wide_pos[m.expected_index]])
                where = exp_entries[m.expected_index].title
            else:
                span = _capped_span(act_sections_wide[act_wide_pos[m.actual_index]])
                where = act_entries[m.actual_index].title
            if span:
                return span, where
        return [], None

    exp_heading_titles = {normalize_title(e.title) for e in exp_entries if e.title.strip()}
    act_heading_titles = {normalize_title(e.title) for e in act_entries if e.title.strip()}
    # A heading bookmarked in only ONE document (e.g. Staging's own
    # "Typographics" bookmark, with no Production counterpart) still exists
    # as plain, unbookmarked running text at that same spot in the OTHER
    # document - dropping it via only that document's OWN heading set misses
    # this, so it survives into the wide-matched parent section as a bare,
    # uncounterparted "prod only"/"stage only" row, even though its real
    # content is already shown properly under its own separate section entry
    # further down. Use the UNION of both documents' heading titles so either
    # side recognizes and skips it.
    all_heading_titles = exp_heading_titles | act_heading_titles
    icon_dropped = [
        _tok_set(t)
        for d in (callout_icon_findings or [])
        for t in (d.get("text") or [])
        if len(_tok_set(t)) >= 3
    ]
    image_notes_by_heading = _image_notes_by_heading(image_findings)
    table_notes_by_heading = _table_notes_by_heading(table_findings)
    format_notes_by_heading = _format_notes_by_heading(format_findings)
    list_notes_by_heading = _list_notes_by_heading(list_findings)
    snap_dir = None
    if output_dir:
        snap_dir = os.path.join(output_dir, "sections")
        os.makedirs(snap_dir, exist_ok=True)
    exp_words = _ocr.PageWords(expected)
    act_words = _ocr.PageWords(actual)
    # Whole-document text, for the "Match content" button fallback below -
    # every heading's body in each document, not just the current section.
    exp_all_blob = _blocks_blob([b for blocks in exp_sections for b in blocks])
    act_all_blob = _blocks_blob([b for blocks in act_sections for b in blocks])
    # Every sentence in each document, for `_fill_relocated_counterparts`: a
    # row left one-sided after its section is matched gets its empty column
    # filled from the best match anywhere in the other document.
    exp_all_sent = [
        r["text"] for blocks in exp_sections
        for r in _section_records(_prose_only(list(blocks), exp_tables, exp_images), all_heading_titles)
    ]
    act_all_sent = [
        r["text"] for blocks in act_sections
        for r in _section_records(_prose_only(list(blocks), act_tables, act_images), all_heading_titles)
    ]

    out: list[dict] = []
    # For a heading bookmarked in only ONE document (e.g. Staging auto-
    # bookmarks "Power delivery of USB-C™ ports on your monitor", or every
    # numbered assembly step, where Production keeps one plain section), its
    # own body IS real content in that PDF and must be shown - but the OTHER
    # column can't just be blank, because that same content usually exists in
    # the other document too, under the parent heading. So a one-sided
    # section's content is diffed against the OTHER side of the most recent
    # MATCHED parent: sentences that are there too show as "=", genuinely new
    # ones as "stage only"/"prod only". The reader sees exactly what's under
    # the heading and where its counterpart lives.
    parent_exp_recs: list[dict] = []
    parent_act_recs: list[dict] = []
    parent_exp_blocks: list[dict] = []
    parent_act_blocks: list[dict] = []
    parent_heading = ""
    for n, m in enumerate(matches):
        ei, ai = m.expected_index, m.actual_index
        exp_e = exp_entries[ei] if ei is not None else None
        act_e = act_entries[ai] if ai is not None else None
        head = (exp_e or act_e).title
        level = (exp_e or act_e).level

        if ei is not None and ai is not None:
            status = "matched"
        elif ei is not None:
            status = "missing"  # Production only
        else:
            status = "extra"  # Staging only

        # Front matter - a cover page, a title block, a printed contents page
        # that only one document has - is not a section anyone is validating:
        # there is nothing to compare it against, so it would render as a pair
        # of empty columns the reader has to scroll past. Marked out of scope
        # by `pdfval.validators.headings`, and dropped here.
        if (exp_e is None or act_e is None) and ((exp_e or act_e).excluded):
            continue

        # A printed contents / Q&A index heading BOTH documents have: keep it in
        # the nav with its match status, but do NOT diff its listing text line
        # by line - that is the TOC Comparison report's job. Emit a stub.
        if (exp_e and exp_e.excluded) or (act_e and act_e.excluded):
            out.append({
                "id": f"sec{n}",
                "heading": head,
                "level": level,
                "status": status,
                "expected_page": (exp_e.page + 1) if exp_e else None,
                "actual_page": (act_e.page + 1) if act_e else None,
                "rows": [],
                "counts": {"equal": 0, "change": 0, "prod": 0, "stage": 0, "diff": 0},
                "exp_shots": [],
                "act_shots": [],
                # Not diffed line by line, but both sides still get their page
                # snapshot - a stub with two empty columns tells the reader
                # nothing about a contents page that may well have changed.
                "_exp_span": _heading_band(expected, exp_e, exp_ends[ei] if ei is not None else None),
                "_act_span": _heading_band(actual, act_e, act_ends[ai] if ai is not None else None),
                "toc_listing": True,
                "counterpart_note": (
                    "This is a printed table-of-contents / index page. It is compared in the "
                    "TOC Comparison report, not line by line here."
                    + ("" if status == "matched"
                       else "  This heading is "
                       + ("in Production only." if status == "missing" else "in Staging only."))
                ),
            })
            continue

        if status == "matched":
            exp_blocks = exp_sections_wide[exp_wide_pos[ei]]
            act_blocks = act_sections_wide[act_wide_pos[ai]]
            exp_recs = _section_records(_prose_only(exp_blocks, exp_tables, exp_images), all_heading_titles)
            act_recs = _section_records(_prose_only(act_blocks, act_tables, act_images), all_heading_titles)
            rows = _align([r["text"] for r in exp_recs], [r["text"] for r in act_recs])
            _reconcile_cross_present(rows, _blocks_blob(exp_blocks), _blocks_blob(act_blocks))
            exp_pages = sorted({b["page"] for b in exp_blocks if isinstance(b.get("page"), int)})
            act_pages = sorted({b["page"] for b in act_blocks if isinstance(b.get("page"), int)})
            _reconcile_ocr_present(rows, exp_words, exp_pages, act_words, act_pages)
            _mark_whole_doc_matches(rows, exp_all_blob, act_all_blob)
            _annotate_callouts(rows, exp_recs, act_recs, icon_dropped)
            _annotate_links(rows, exp_recs, act_recs, expected, actual)
            _row_locations(rows, exp_recs, act_recs)
            parent_exp_recs, parent_act_recs, parent_heading = exp_recs, act_recs, head
            parent_exp_blocks, parent_act_blocks = exp_blocks, act_blocks
            counterpart_note = None
            exp_span, act_span = exp_blocks, act_blocks
        elif status == "missing":
            exp_blocks = exp_sections[ei]
            exp_recs = _section_records(_prose_only(exp_blocks, exp_tables, exp_images), all_heading_titles)
            # Show every sentence that's actually under this Production-only
            # sub-heading, and line each one up against its counterpart in the
            # Staging parent (so a match reads "=" and a genuinely dropped
            # sentence reads "prod only"). Parent sentences that belong to
            # OTHER sub-sections are dropped - only rows that carry a sentence
            # from THIS section are kept.
            rows = _one_sided_rows([r["text"] for r in exp_recs],
                                   [r["text"] for r in parent_act_recs], "prod")
            _reconcile_cross_present(rows, _blocks_blob(exp_blocks), _blocks_blob(parent_act_blocks))
            act_pages = sorted({b["page"] for b in parent_act_blocks if isinstance(b.get("page"), int)})
            _reconcile_ocr_present(rows, exp_words, [], act_words, act_pages)
            _mark_whole_doc_matches(rows, exp_all_blob, act_all_blob)
            _annotate_callouts(rows, exp_recs, parent_act_recs, icon_dropped)
            _annotate_links(rows, exp_recs, parent_act_recs, expected, actual)
            _row_locations(rows, exp_recs, parent_act_recs)
            counterpart_note = (
                f"Not bookmarked as its own heading in Staging — this content sits under “{parent_heading}” there."
                if parent_heading else "No matching heading in Staging."
            )
            # Staging snapshot = the patch of the parent section where this
            # sub-section's sentences actually live.
            own_keys = {_key(r["text"]) for r in exp_recs}
            exp_span = exp_blocks
            act_span = [r for r in parent_act_recs if _key(r["text"]) in own_keys]
        else:  # extra / Staging-only
            act_blocks = act_sections[ai]
            act_recs = _section_records(_prose_only(act_blocks, act_tables, act_images), all_heading_titles)
            rows = _one_sided_rows([r["text"] for r in act_recs],
                                   [r["text"] for r in parent_exp_recs], "stage")
            _reconcile_cross_present(rows, _blocks_blob(parent_exp_blocks), _blocks_blob(act_blocks))
            exp_pages = sorted({b["page"] for b in parent_exp_blocks if isinstance(b.get("page"), int)})
            _reconcile_ocr_present(rows, exp_words, exp_pages, act_words, [])
            _mark_whole_doc_matches(rows, exp_all_blob, act_all_blob)
            _annotate_callouts(rows, parent_exp_recs, act_recs, icon_dropped)
            _annotate_links(rows, parent_exp_recs, act_recs, expected, actual)
            _row_locations(rows, parent_exp_recs, act_recs)
            counterpart_note = (
                f"Not bookmarked as its own heading in Production — this content sits under “{parent_heading}” there."
                if parent_heading else "No matching heading in Production."
            )
            own_keys = {_key(r["text"]) for r in act_recs}
            exp_span = [r for r in parent_exp_recs if _key(r["text"]) in own_keys]
            act_span = act_blocks

        rows, n_index = _strip_index_runs(rows)
        rows = _drop_bare_callout_label_rows(rows)

        # Neither column may be left empty. A section can have no text blocks
        # of its own on one side - a container heading, a page that is all
        # figure or all table, or a one-sided heading none of whose sentences
        # turned up under the other document's parent - and the snapshot pane
        # then read "Not in Staging" about content that is in fact printed
        # right there. Fall back to the heading's own page band first (the real
        # answer whenever the heading exists on that side), then to the nearest
        # matched section's region, labelled so the reader knows which it is.
        own_exp, own_act = bool(exp_span), bool(act_span)
        exp_span = exp_span or _heading_band(expected, exp_e, exp_ends[ei] if ei is not None else None)
        act_span = act_span or _heading_band(actual, act_e, act_ends[ai] if ai is not None else None)
        exp_note = act_note = None
        if not exp_span:
            exp_span, where = counterpart_region("prod", n)
            if where:
                exp_note = f"nearest matched section — “{where}”"
        if not act_span:
            act_span, where = counterpart_region("stage", n)
            if where:
                act_note = f"nearest matched section — “{where}”"

        row_dict = {
            "id": f"sec{n}",
            "heading": head,
            "level": level,
            "status": status,
            "expected_page": (exp_e.page + 1) if exp_e else None,
            "actual_page": (act_e.page + 1) if act_e else None,
            "rows": rows,
            "counts": _counts(rows),
            "exp_shots": [],
            "act_shots": [],
            # Rendered after `_fill_relocated_counterparts` below, once every
            # row's final op is known.
            "_exp_span": exp_span,
            "_act_span": act_span,
            "_own_text": own_exp or own_act,
        }
        if exp_note:
            row_dict["exp_shots_note"] = exp_note
        if act_note:
            row_dict["act_shots_note"] = act_note
        # "No text content extracted for this section", in a section only one
        # document even has, tells the reader nothing - drop it rather than
        # make them scroll past it.
        if status != "matched" and not rows:
            continue
        # A section whose body is entirely a table or a figure has no prose to
        # diff, but it is NOT empty - its content is real and is compared by
        # Table/Image Validation. Saying "no text content extracted" about a
        # page full of a RoHS substance table reads as a hole in the report.
        if not rows and (own_exp or own_act):
            row_dict["table_only"] = True
        if counterpart_note:
            row_dict["counterpart_note"] = counterpart_note
        content_match_count = sum(1 for r in rows if r.get("content_match_text"))
        if content_match_count:
            row_dict["content_match_count"] = content_match_count
        if image_notes_by_heading.get(head):
            row_dict["image_notes"] = image_notes_by_heading[head]
        if table_notes_by_heading.get(head):
            row_dict["table_notes"] = table_notes_by_heading[head]
        if format_notes_by_heading.get(head):
            row_dict["format_notes"] = format_notes_by_heading[head]
            row_dict["format_kinds"] = _format_kind_chips(format_notes_by_heading[head])
        if list_notes_by_heading.get(head):
            row_dict["list_notes"] = list_notes_by_heading[head]
        if n_index:
            row_dict["index_note"] = (
                f"{n_index} contents / FAQ-index navigation entries in this section "
                "(each linking to a page) are not shown here or compared line by line — "
                "see the page snapshot below for the full list."
            )
        out.append(row_dict)

    # Fill the opposite column of any row still one-sided from the other
    # document's full text, so a cell is never left blank for content that
    # merely moved to a different heading.
    _fill_relocated_counterparts(out, exp_all_sent, act_all_sent)

    # Snapshots last. `_fill_relocated_counterparts` above can turn a row that
    # looked one-sided into a plain match, and a highlight box drawn before
    # that ran would box content the report no longer calls a difference.
    for s in out:
        exp_span = s.pop("_exp_span", None)
        act_span = s.pop("_act_span", None)
        if not snap_dir:
            continue
        # Pixel-exact snapshot of exactly this section's region in each PDF -
        # the faithful "what does it actually look like" view, Production
        # beside Staging, clipped per page to the section's own block span,
        # with every difference boxed and NUMBERED identically on both sides so
        # a highlight on the left can be found on the right at a glance.
        prod_boxes, stage_boxes = _diff_highlight_boxes(s["rows"])
        if exp_span:
            s["exp_shots"] = _section_snapshots(
                expected, exp_span, snap_dir, f"{s['id']}_prod", prod_boxes
            )
        if act_span:
            s["act_shots"] = _section_snapshots(
                actual, act_span, snap_dir, f"{s['id']}_stage", stage_boxes
            )

    # Flag a heading that only groups the deeper headings right below it (its
    # own body is empty because its first child starts at the same spot) so
    # the reader isn't told "no text extracted" for what is really just a
    # container heading.
    for i, s in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        own_text = s.pop("_own_text", True)
        if s["rows"] or own_text or not nxt:
            continue
        same_page = (
            (s["expected_page"] and s["expected_page"] == nxt["expected_page"])
            or (s["actual_page"] and s["actual_page"] == nxt["actual_page"])
        )
        if nxt["level"] > s["level"] or same_page:
            s["is_container"] = True
    return out
