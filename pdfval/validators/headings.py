"""The section list both documents are compared by.

Every check in this engine is heading-anchored: Content, Image and Table
Validation each walk `matched_heading_ranges` so that Production page 4 is
compared against whatever Staging page actually carries the same section, not
against Staging page 4. That only works if both documents HAVE a usable list of
headings, and if the two lists line up - and a PDF's embedded outline cannot be
relied on for either:

* A Production PDF exported straight from the layout tool often has no
  bookmarks at all, while its Staging rebuild has one or two. With no outline
  the checks fall back to comparing page N against page N, which is silently
  wrong the moment the two documents paginate differently (8 pages vs 25 on the
  pair this was built against).
* Even with outlines, the two sides rarely bookmark the same set. A heading
  only one side lists makes its section match nothing, so a section that is
  plainly present in both PDFs gets reported as missing.

So this module resolves the heading list the whole engine works from:

1. `get_toc_entries` - the real embedded outline, always preferred.
2. `synthesize_heading_entries` - failing that, the headings as they are
   PRINTED, detected by style (set larger than the body text, or set bold at
   body size).
3. `cross_seed_entries` - then each side is given an entry for any heading only
   the other side has, wherever that heading text really is printed there.

`validate_toc` and the TOC comparison report deliberately do NOT use this: their
job is to check the embedded outline itself, so they must keep reading it raw.
"""
from __future__ import annotations

import collections
import re

import fitz

from pdfval.validators.toc import (
    TocEntry,
    get_toc_entries,
    is_excluded_heading,
    looks_like_toc_listing,
    match_toc_entries,
    normalize_title,
)

_HEADING_MIN_RATIO = 1.3  # font size vs. body text, to count as a heading
_HEADING_MAX_CHARS = 90  # a heading is a title, not a wrapped paragraph
_HEADING_MIN_CHARS = 3  # "(Pb)" is a bold table column header, not a section
# A heading is not always set LARGER than the body text - plenty of manuals keep
# the section title at body size and only set it bold/medium. Detecting on size
# alone found 34 headings in one real Production PDF and 4 in its Staging
# rebuild, which left almost every section one-sided, so weight counts as
# prominence too.
_HEAVY_FONT_RE = re.compile(r"bold|black|heavy|semib|demi|medium", re.I)

_SEED_MAX_CHARS = 120
_LISTING_PAGE_TITLES = 5  # this many of the other side's titles on one page = a printed contents page


# --- detecting headings by how they are printed ---------------------------


def body_style(doc: fitz.Document, sample_pages: int = 20) -> tuple[float, bool]:
    """`(size, is_heavy)` of the document's body text - the style the most
    CHARACTERS are set in, not the most spans, so a page of short bold table
    cells can't outvote the prose around them."""
    weight: collections.Counter = collections.Counter()
    for page in doc[: min(sample_pages, doc.page_count)]:
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        style = (
                            round(span.get("size", 0), 1),
                            bool(_HEAVY_FONT_RE.search(span.get("font", ""))),
                        )
                        weight[style] += len(text)
    return weight.most_common(1)[0][0] if weight else (10.0, False)


def _line_is_prominent(line: dict, body_size: float, body_heavy: bool) -> bool:
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans or body_size <= 0:
        return False
    max_size = max(round(s.get("size", 0), 1) for s in spans)
    if max_size / body_size >= _HEADING_MIN_RATIO:
        return True
    heavy = all(_HEAVY_FONT_RE.search(s.get("font", "")) for s in spans)
    return heavy and not body_heavy and max_size >= body_size * 0.95


def _overlaps(bbox: tuple, regions: list[tuple], threshold: float) -> bool:
    ax0, ay0, ax1, ay1 = bbox
    area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    for bx0, by0, bx1, by1 in regions:
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 > ix0 and iy1 > iy0 and ((ix1 - ix0) * (iy1 - iy0)) / area >= threshold:
            return True
    return False


def _table_regions(doc: fitz.Document, page_index: int) -> list[tuple]:
    path = getattr(doc, "name", None)
    if not path:
        return []
    try:
        from pdfval.extractor import get_all_detected_regions

        return get_all_detected_regions(path, page_index)
    except Exception:
        return []


def _figure_regions(doc: fitz.Document, page_index: int) -> list[tuple]:
    try:
        from pdfval.extractor import get_figures

        return [f.bbox for f in get_figures(doc, page_index)]
    except Exception:
        return []


def synthesize_heading_entries(doc: fitz.Document) -> list[TocEntry]:
    """The document's headings as they are PRINTED, for a PDF whose embedded
    outline is missing or a token stub.

    Deliberately conservative: a running header/footer that repeats
    near-identically on most pages is dropped (it's not a section heading), a
    bold table column header or a label set on a diagram is dropped (that region
    is Table/Image Validation's job, and its cells otherwise read as dozens of
    one-word "sections"), and only short, non-sentence-punctuated lines qualify
    - so a document that genuinely has no distinct heading styling yields an
    empty list rather than false headings.
    """
    body_size, body_heavy = body_style(doc)
    if body_size <= 0:
        return []
    raw: list[tuple[int, float, str]] = []
    seen_counts: dict[str, int] = {}
    for page_index in range(doc.page_count):
        table_boxes = _table_regions(doc, page_index)
        figure_boxes = _figure_regions(doc, page_index)
        for b in doc[page_index].get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            for line in b.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = " ".join(" ".join(s["text"] for s in spans).split())
                if not (_HEADING_MIN_CHARS <= len(text) <= _HEADING_MAX_CHARS):
                    continue
                if looks_like_toc_listing(text) or text.rstrip().endswith((".", ",", ";", ":")):
                    continue
                if not _line_is_prominent(line, body_size, body_heavy):
                    continue
                bbox = tuple(line["bbox"])
                if table_boxes and _overlaps(bbox, table_boxes, 0.5):
                    continue
                if figure_boxes and _overlaps(bbox, figure_boxes, 0.8):
                    continue
                raw.append((page_index, bbox[1], text))
                seen_counts[text] = seen_counts.get(text, 0) + 1
    repeat_cap = max(3, doc.page_count // 3)
    return [
        TocEntry(level=1, title=text, page=page_index, y=y, excluded=is_excluded_heading(text))
        for page_index, y, text in raw
        if seen_counts[text] <= repeat_cap
    ]


# --- keeping both sides' section lists in step ----------------------------


def heading_line_index(doc: fitz.Document) -> dict[str, list[tuple[int, float, str, bool]]]:
    """`normalized title -> [(page, y, text as printed, styled_as_a_heading)]`
    for every short text line - and short block, for a title this document wraps
    over two lines - so a heading known only to the other side can be located
    here.

    The style flag is the tie-breaker: the same words appear both in a printed
    contents list and as the section's own title, and the one actually set as a
    heading is where the section starts.
    """
    body_size, body_heavy = body_style(doc)
    index: dict[str, list[tuple[int, float, str, bool]]] = {}

    def add(text: str, page: int, y: float, styled: bool) -> None:
        text = " ".join(text.split())
        if not text or len(text) > _SEED_MAX_CHARS or looks_like_toc_listing(text):
            return
        key = normalize_title(text)
        if key:
            index.setdefault(key, []).append((page, y, text, styled))

    for page_index in range(doc.page_count):
        for b in doc[page_index].get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            lines = [
                ln for ln in b.get("lines", [])
                if any(s.get("text", "").strip() for s in ln.get("spans", []))
            ]
            for line in lines:
                add(
                    " ".join(s["text"] for s in line.get("spans", [])),
                    page_index, line["bbox"][1],
                    _line_is_prominent(line, body_size, body_heavy),
                )
            if 2 <= len(lines) <= 3:
                whole = " ".join(
                    " ".join(s["text"] for s in ln.get("spans", [])) for ln in lines
                )
                add(
                    whole, page_index, b["bbox"][1],
                    all(_line_is_prominent(ln, body_size, body_heavy) for ln in lines),
                )
    for key in index:
        index[key].sort()
    return index


def _seed_side(
    src_entries: list[TocEntry], dst_entries: list[TocEntry],
    index: dict[str, list[tuple[int, float, str, bool]]],
) -> list[TocEntry]:
    """Give `dst_entries` its own entry for every `src_entries` heading whose
    text is really printed in the destination document (`index`, from
    `heading_line_index`) but was never bookmarked or styled as a heading there.

    The spot is picked in reading order between the surrounding already-matched
    headings, so a seeded heading can't land out of sequence; a page carrying
    five or more of the other side's titles is that document's printed contents
    page and is skipped, so the seed lands on the section itself rather than on
    its line in the table of contents.
    """
    if not index:
        return dst_entries
    matches = match_toc_entries(src_entries, dst_entries, include_excluded=True)
    if not any(m.expected_index is not None and m.actual_index is None for m in matches):
        return dst_entries

    page_titles: dict[int, set[str]] = {}
    for e in src_entries:
        key = normalize_title(e.title)
        for page, _y, _t, _styled in index.get(key, []):
            page_titles.setdefault(page, set()).add(key)
    listing_pages = {p for p, titles in page_titles.items() if len(titles) >= _LISTING_PAGE_TITLES}

    pos: list[tuple[float, float] | None] = [
        (float(dst_entries[m.actual_index].page), dst_entries[m.actual_index].y)
        if m.actual_index is not None else None
        for m in matches
    ]
    taken = {(e.page, round(e.y)) for e in dst_entries}
    added: list[TocEntry] = []
    for i, m in enumerate(matches):
        if m.actual_index is not None or m.expected_index is None:
            continue
        entry = src_entries[m.expected_index]
        candidates = [
            c for c in index.get(normalize_title(entry.title), [])
            if (c[0], round(c[1])) not in taken
        ]
        if not candidates:
            continue
        lo = next((p for p in reversed(pos[:i]) if p is not None), (-1.0, -1.0))
        hi = next((p for p in pos[i + 1:] if p is not None), (float("inf"), float("inf")))
        on_page = [c for c in candidates if c[0] not in listing_pages] or candidates
        in_order = [c for c in on_page if lo < (float(c[0]), c[1]) < hi] or on_page
        # Where the same words appear more than once in range, the occurrence
        # actually SET as a heading is the section's real start.
        styled = [c for c in in_order if c[3]]
        page, y, text, _ = (styled or in_order)[0]
        taken.add((page, round(y)))
        pos[i] = (float(page), y)
        added.append(TocEntry(
            level=entry.level, title=text, page=page, y=y, excluded=is_excluded_heading(text)
        ))
    if not added:
        return dst_entries
    return sorted(dst_entries + added, key=lambda e: (e.page, e.y))


def cross_seed_entries(
    expected: fitz.Document, exp_entries: list[TocEntry],
    actual: fitz.Document, act_entries: list[TocEntry],
) -> tuple[list[TocEntry], list[TocEntry]]:
    """Make both documents offer the SAME section list wherever the heading text
    exists on both sides - seeding each side from the other, then once more so a
    heading found only by the first pass can pull its counterpart across too.
    Each document's text is indexed once and reused across both passes; it does
    not change as entries are added."""
    exp_index = heading_line_index(expected)
    act_index = heading_line_index(actual)
    for _ in range(2):
        before = (len(exp_entries), len(act_entries))
        act_entries = _seed_side(exp_entries, act_entries, act_index)
        exp_entries = _seed_side(act_entries, exp_entries, exp_index)
        if (len(exp_entries), len(act_entries)) == before:
            break
    return exp_entries, act_entries


# --- the resolved list every check works from -----------------------------
#
# Resolving walks both documents' text several times, and five checks plus the
# section browser each ask for it once per run - so the answer is memoised for
# the run. Keyed by a cheap fingerprint rather than id(), because the web server
# opens and closes many documents in one process and CPython reuses ids.
# `reset_heading_cache()` is called at the end of every run, next to the other
# per-run cache resets.
_RESOLVED: dict[tuple, tuple[list[TocEntry], list[TocEntry]]] = {}
_MIN_REAL_ENTRIES = 2  # fewer bookmarks than this is a stub, not an outline


def reset_heading_cache() -> None:
    _RESOLVED.clear()


def _fingerprint(doc: fitz.Document) -> tuple:
    try:
        outline = len(doc.get_toc())
    except Exception:
        outline = -1
    return (getattr(doc, "name", None), doc.page_count, outline)


def resolve_entries(
    expected: fitz.Document, actual: fitz.Document
) -> tuple[list[TocEntry], list[TocEntry]]:
    """`(exp_entries, act_entries)` - the heading list every check anchors on.

    Real bookmarks win whenever there are enough of them to navigate by; a
    document without them gets its printed headings detected instead; and either
    side is then given any heading the other has that is genuinely printed here
    too, so the two lists describe the same sections.
    """
    key = (_fingerprint(expected), _fingerprint(actual))
    hit = _RESOLVED.get(key)
    if hit is not None:
        # Copies: callers filter and re-order the lists they are handed.
        return list(hit[0]), list(hit[1])

    exp_entries = get_toc_entries(expected)
    act_entries = get_toc_entries(actual)
    if len(exp_entries) < _MIN_REAL_ENTRIES:
        exp_entries = synthesize_heading_entries(expected) or exp_entries
    if len(act_entries) < _MIN_REAL_ENTRIES:
        act_entries = synthesize_heading_entries(actual) or act_entries
    exp_entries, act_entries = cross_seed_entries(expected, exp_entries, actual, act_entries)
    _mark_front_matter(exp_entries, act_entries)

    _RESOLVED[key] = (exp_entries, act_entries)
    return list(exp_entries), list(act_entries)


# --- locating a section, for the other document's column ------------------

_BAND_MAX_HEIGHT = 320.0  # points - a counterpart crop is a look, not a page dump


def section_band(
    doc: fitz.Document, entries: list[TocEntry], title: str
) -> tuple[int | None, tuple[float, float, float, float] | None]:
    """`(page, bbox)` of where `title`'s section starts in `doc` - the heading
    line down to the next heading on that page, capped so the crop stays
    readable.

    This is what fills the empty column of a finding only one document has
    evidence for. It is only meaningful because `resolve_entries` guarantees
    both documents carry the same headings: without that, a "counterpart
    region" would be a guess.
    """
    key = normalize_title(title or "")
    if not key or not entries:
        return None, None
    match = next((e for e in entries if normalize_title(e.title) == key), None)
    if match is None or not (0 <= match.page < doc.page_count):
        return None, None
    rect = doc[match.page].rect
    later = [e.y for e in entries if e.page == match.page and e.y > match.y + 1]
    end = min(later) if later else rect.y1
    y0 = max(rect.y0, match.y - 4.0)
    y1 = min(rect.y1, end, y0 + _BAND_MAX_HEIGHT)
    if y1 - y0 < 8:
        y1 = min(rect.y1, y0 + _BAND_MAX_HEIGHT)
    return match.page, (rect.x0, y0, rect.x1, y1)


# --- front matter is not content to validate ------------------------------
#
# A cover page ("Monitor LCD-RS-Online-V14", "user manual") and a printed
# contents page are not sections anyone wants compared: they exist in only one
# of the two documents as often as not, they carry no prose to diff, and the
# embedded outline is already checked by `validate_toc` and the TOC comparison
# report. Left in, each one becomes a section with both columns empty - exactly
# the "nothing here" rows a reader has to scroll past to reach the real
# findings.
#
# They are MARKED (excluded=True), never removed: an entry is also a section
# BOUNDARY, and dropping the cover's entries would let the first real section
# swallow the cover and contents pages whole.


def _mark_front_matter(exp_entries: list[TocEntry], act_entries: list[TocEntry]) -> None:
    """Mark as out-of-scope every heading that sits before the document's first
    heading the two sides share, and every printed contents page."""
    matches = match_toc_entries(exp_entries, act_entries, include_excluded=True)
    first_exp = next(
        (m.expected_index for m in matches
         if m.expected_index is not None and m.actual_index is not None),
        None,
    )
    first_act = next(
        (m.actual_index for m in matches
         if m.expected_index is not None and m.actual_index is not None),
        None,
    )
    for entries, first in ((exp_entries, first_exp), (act_entries, first_act)):
        if first is None or not (0 <= first < len(entries)):
            continue
        start = (entries[first].page, entries[first].y)
        for e in entries:
            if (e.page, e.y) < start:
                e.excluded = True
