"""5. Alignment validation: how text is laid out, as opposed to what it says.

Findings:

* List marker changed          - 1,2,3 vs a,b,c vs bullets
* List marker size changed     - same kind of marker (e.g. bullet vs bullet), different size
* List alignment/indent changed - a list item's indent moved
* Text alignment changed       - left/center/right/justified changed
* Paragraph merged with heading - a numbered item's description glued onto the
                                  same line as its bold label

Everything here compares text that is identical on both sides, so a finding is
always "the same words, presented differently" and never a disguised content
difference - those belong to Content Validation.

The three block-level checks pair whole blocks whose text matches exactly. The
marker check cannot: a list whose markers changed has, by definition, different
text on the two sides, and half the time the marker isn't in the item's text at
all - it is drawn as its own text object beside the item. So it pairs list
ITEMS on the item's own wording, with the marker (wherever it is drawn) read
separately and compared. See `_list_items`.
"""
from __future__ import annotations

import itertools
import re

import fitz

from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import (
    extract_section_blocks,
    match_toc_entries,
    normalize_block_text,
)

MAX_SCREENSHOTS = 150
MAX_ISSUES_PER_KIND = 60  # layout findings are per-block; cap so they can't swamp the report

INDENT_TOLERANCE = 6.0  # points a list item's indent may move before it's reported
ALIGNMENT_TOLERANCE = 12.0  # points of asymmetry allowed before alignment reclassifies
JUSTIFY_TOLERANCE = 3.0  # points of ragged right edge allowed before a block counts as justified
MIN_JUSTIFY_LINES = 3  # a block needs this many lines before "justified" is meaningful

_BOLD_FLAG = 1 << 4

# The leading marker of a list item, captured so its STYLE (number vs letter vs
# bullet) can be compared even though the marker text itself is not content.
#
# Multi-letter roman numerals (ii., iii., iv., ... xx.) are spelled out
# explicitly (not a general roman-numeral regex) to keep the pattern bounded
# and avoid ambiguity with ordinary short words - a lone "i."/"I." is still
# handled by the `[a-zA-Z][.)]` branch (equally plausible as a lettered-list
# marker), so it's deliberately not repeated here. Mirrors content.py's
# identical marker regex.
_ROMAN_NUMERAL_MARKERS = r"ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx"
# A middle dot is in this set because a bullet set in one of the base-14 fonts
# has no U+2022 to map to and extracts as U+00B7 - the page shows a bullet, the
# text layer says middle dot, and the list would otherwise go unrecognised.
_BULLET_GLYPHS = "\u2022\u25e6\u25aa\u25b8\u2023\u2043\u00b7\u2219"
_LIST_MARKER_RE = re.compile(
    r"^\s*(\d{1,2}[.)]|[a-zA-Z][.)]|(?:" + _ROMAN_NUMERAL_MARKERS + r")[.)]|[" + _BULLET_GLYPHS + r"])\s+",
    re.IGNORECASE,
)

# A numbered item's bold label, e.g. "1. Brightness" or "3) Input Source" - the
# part that is supposed to stand alone on its own line.
_NUMBERED_LABEL_RE = re.compile(r"^\s*\d{1,2}[.)]\s+\S")

# A line that is NOTHING BUT a list marker. Publishing tools routinely draw an
# ordered list's marker as its own text object to the LEFT of the item it
# labels - Staging does exactly this ("a." at x=280, "Disable Set time
# automatically." at x=295, two separate blocks) - so the marker never appears
# inside the item's own text. A check that only looked at text STARTING with a
# marker therefore saw an unnumbered paragraph and had nothing to compare,
# which is why a document whose every procedure was renumbered 1,2,3 -> a,b,c
# produced no marker finding at all.
#
# A digit/letter marker must carry its "." or ")" here: a bare "41" sitting on
# its own is a page number, not a list item.
_MARKER_ONLY_RE = re.compile(
    r"^(\d{1,3}[.)]|(?:" + _ROMAN_NUMERAL_MARKERS + r")[.)]|[a-zA-Z][.)]|[" + _BULLET_GLYPHS + r"])$",
    re.IGNORECASE,
)

# How far to the right of a detached marker its item text may start (points).
# Wide enough for a deep hanging indent, narrow enough that the marker can't
# adopt a neighbouring column's text.
MARKER_TEXT_MAX_GAP = 48.0
# Vertical gap between two items that still reads as ONE list, so a renumbered
# procedure is reported as a single finding covering the whole list.
LIST_RUN_MAX_GAP = 60.0
# How far one line's box may reach INTO the next one's before the two stop
# counting as consecutive items (descenders and ascenders overlap routinely).
LINE_OVERLAP_SLACK = 8.0
# An item's text must be at least this long before it is safe to pair two
# documents' items by it - "OK." or "Yes" occurs everywhere and would pair the
# wrong two lines.
MIN_ITEM_KEY_CHARS = 6


def classify_marker(marker: str) -> str:
    if re.match(r"^\d{1,3}[.)]$", marker):
        return "number"
    if re.match(rf"^(?:{_ROMAN_NUMERAL_MARKERS})[.)]$", marker, re.IGNORECASE):
        return "roman"
    if re.match(r"^[a-zA-Z][.)]$", marker):
        return "letter"
    return "bullet"


def validate_alignment(
    expected: fitz.Document, actual: fitz.Document, output_dir: str | None = None
) -> CheckResult:
    result = CheckResult(name="Alignment Validation")
    counter = itertools.count(1)
    counts: dict[str, int] = {}

    exp_entries, act_entries = resolve_entries(expected, actual)
    # One cache per document, not one per section: every check here reads a
    # page's lines, and a page is read by as many sections as start or end on
    # it. Built once, each page is extracted at most once for the whole run.
    exp_lines = _LineCache(expected)
    act_lines = _LineCache(actual)

    if exp_entries and act_entries:
        matches = match_toc_entries(exp_entries, act_entries)
        matched = [m for m in matches if m.expected_index is not None and m.actual_index is not None]
        exp_sections = extract_section_blocks(expected, [exp_entries[m.expected_index] for m in matched])
        act_sections = extract_section_blocks(actual, [act_entries[m.actual_index] for m in matched])
        for idx, m in enumerate(matched):
            _compare_section(
                result, expected, actual, exp_sections[idx], act_sections[idx],
                exp_lines, act_lines, exp_entries[m.expected_index].title, output_dir, counter, counts,
            )
    else:
        _compare_section(
            result, expected, actual, _all_blocks(expected), _all_blocks(actual),
            exp_lines, act_lines, None, output_dir, counter, counts,
        )
    return result


def _all_blocks(doc: fitz.Document) -> list[dict]:
    blocks: list[dict] = []
    for page in range(doc.page_count):
        for b in doc[page].get_text("blocks"):
            if b[4].strip():
                blocks.append(
                    {"text": normalize_block_text(b[4]), "page": page, "bbox": (b[0], b[1], b[2], b[3])}
                )
    return blocks


class _LineCache:
    """Per-page line geometry and span styling, memoized. Block-level text
    extraction says nothing about how lines sit inside a block, which is
    exactly what alignment and indent findings need.
    """

    def __init__(self, doc: fitz.Document):
        self._doc = doc
        self._pages: dict[int, list[dict]] = {}
        self._content: dict[int, tuple[float, float]] = {}

    def lines(self, page: int) -> list[dict]:
        if page not in self._pages:
            out: list[dict] = []
            try:
                data = self._doc[page].get_text("dict")
            except Exception:
                data = {"blocks": []}
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    out.append(
                        {
                            "bbox": tuple(line.get("bbox", (0, 0, 0, 0))),
                            "text": "".join(s["text"] for s in spans),
                            "spans": [
                                {
                                    "text": s["text"],
                                    "bold": bool(s.get("flags", 0) & _BOLD_FLAG),
                                    "bbox": tuple(s.get("bbox", (0, 0, 0, 0))),
                                    "size": float(s.get("size", 0.0)),
                                }
                                for s in spans
                            ],
                        }
                    )
            self._pages[page] = out
        return self._pages[page]

    def lines_in(self, page: int, bbox: tuple) -> list[dict]:
        out = []
        for line in self.lines(page):
            cy = (line["bbox"][1] + line["bbox"][3]) / 2
            if bbox[1] - 1 <= cy <= bbox[3] + 1:
                out.append(line)
        return sorted(out, key=lambda ln: ln["bbox"][1])

    def body_left(self, page: int) -> float:
        """The page's body-text left margin, taken as the MOST COMMON line left
        edge rather than the leftmost one.

        Indent has to be measured against something both documents agree on.
        The leftmost edge on a page is set by whatever happens to stick out
        furthest there - a running header, a figure, a table - so it moves for
        reasons that have nothing to do with the list being measured. The modal
        edge is the body column itself, which is stable.
        """
        if page not in self._content:
            rect = self._doc[page].rect
            counts: dict[float, int] = {}
            for ln in self.lines(page):
                key = round(ln["bbox"][0])
                counts[key] = counts.get(key, 0) + 1
            self._content[page] = (
                max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0] if counts else rect.x0
            )
        return self._content[page]


def _alignment_of(cache: _LineCache, page: int, bbox: tuple) -> str:
    """Classify a block's alignment from the raggedness of ITS OWN line edges.

    Measuring each block against a page-wide "text column" was tried first and
    is not stable: the column's extent is set by whatever sits furthest out on
    that particular page - a running header, a wide figure - so the same
    paragraph classified as "right" in one document and "center" in the other
    purely because the pages around it differed. A block's own lines are
    self-contained evidence: an aligned edge is one its lines share, and a
    ragged edge is one they don't.
    """
    lines = cache.lines_in(page, bbox)
    # One line cannot be ragged on any edge, so it carries no alignment
    # evidence at all. Reporting on it would be guesswork.
    if len(lines) < 2:
        return "unknown"

    lefts = [ln["bbox"][0] for ln in lines]
    rights = [ln["bbox"][2] for ln in lines]
    # A block's last line is short by definition and would make any alignment
    # look ragged on the right, so it is excluded from the right-edge test.
    body_rights = rights[:-1] if len(lines) >= MIN_JUSTIFY_LINES else rights

    left_flush = max(lefts) - min(lefts) <= JUSTIFY_TOLERANCE
    right_flush = max(body_rights) - min(body_rights) <= JUSTIFY_TOLERANCE

    if left_flush and right_flush:
        return "justified" if len(lines) >= MIN_JUSTIFY_LINES else "left"
    if left_flush:
        return "left"
    if right_flush:
        return "right"

    # Neither edge is shared: centred if the lines' midpoints line up, and
    # otherwise not classifiable - which is the honest answer for an irregular
    # block, and better than inventing a class for it.
    centers = [(ln["bbox"][0] + ln["bbox"][2]) / 2 for ln in lines]
    if max(centers) - min(centers) <= ALIGNMENT_TOLERANCE:
        return "center"
    return "unknown"


def _indent_of(cache: _LineCache, page: int, bbox: tuple) -> float | None:
    """A block's indent: how far its first line starts from the page's text
    column, so the figure is comparable between two documents whose margins
    differ slightly.
    """
    lines = cache.lines_in(page, bbox)
    if not lines:
        return None
    return round(lines[0]["bbox"][0] - cache.body_left(page), 1)


def _pair_blocks(exp_blocks: list[dict], act_blocks: list[dict]) -> list[tuple[dict, dict]]:
    """Pair blocks that carry EXACTLY the same text, and only where that text
    occurs once on each side. Anything ambiguous is skipped rather than guessed
    at - a mispaired block would report a layout change that never happened.
    """
    def unique_index(blocks: list[dict]) -> dict[str, dict]:
        seen: dict[str, dict | None] = {}
        for b in blocks:
            key = b["text"].strip()
            if not key:
                continue
            seen[key] = None if key in seen else b
        return {k: v for k, v in seen.items() if v is not None}

    exp_index = unique_index(exp_blocks)
    act_index = unique_index(act_blocks)
    return [(exp_index[k], act_index[k]) for k in exp_index.keys() & act_index.keys()]


def _compare_section(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_blocks: list[dict],
    act_blocks: list[dict],
    exp_lines: _LineCache,
    act_lines: _LineCache,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    pairs = _pair_blocks(exp_blocks, act_blocks)
    pairs.sort(key=lambda p: (p[0]["page"], p[0]["bbox"][1]))

    # Marker style is compared across the WHOLE section, not per paired block:
    # the marker often isn't in the item's block text at all (see
    # `_MARKER_ONLY_RE`), and the blocks that carry a renumbered list can never
    # pair here anyway - `_pair_blocks` pairs on identical text, and a list
    # whose markers changed has, by definition, different text on the two
    # sides.
    _check_list_markers(
        result, expected, actual, exp_blocks, act_blocks, exp_lines, act_lines,
        heading, output_dir, counter, counts,
    )
    _check_marker_size(
        result, expected, actual, exp_blocks, act_blocks, exp_lines, act_lines,
        heading, output_dir, counter, counts,
    )

    for exp_b, act_b in pairs:
        _check_indent(
            result, expected, actual, exp_b, act_b, exp_lines, act_lines, heading, output_dir, counter, counts
        )
        _check_text_alignment(
            result, expected, actual, exp_b, act_b, exp_lines, act_lines, heading, output_dir, counter, counts
        )

    _check_merged_heading(
        result, expected, actual, pairs, exp_lines, act_lines, heading, output_dir, counter, counts
    )


def _budget(counts: dict[str, int], kind: str) -> bool:
    if counts.get(kind, 0) >= MAX_ISSUES_PER_KIND:
        return False
    counts[kind] = counts.get(kind, 0) + 1
    return True


def _details(exp_b: dict, act_b: dict, heading: str | None) -> dict:
    details: dict = {}
    if heading:
        details["heading"] = heading
    details["expected_page"] = exp_b["page"] + 1
    details["actual_page"] = act_b["page"] + 1
    return details


def _item_key(text: str) -> str:
    """The key two documents' list items are paired on: the item's own text,
    without its marker - the marker is the thing being compared, so it can't
    be part of what identifies the item."""
    return " ".join(text.split()).casefold()


def _make_item(page: int, marker: str, text: str, bbox: tuple, marker_size: float = 0.0) -> dict | None:
    marker = marker.strip()
    key = _item_key(text)
    if len(key) < MIN_ITEM_KEY_CHARS or not any(ch.isalnum() for ch in key):
        return None
    return {
        "page": page,
        "marker": marker,
        "kind": classify_marker(marker),
        "text": " ".join(text.split()),
        "key": key,
        "bbox": tuple(bbox),
        "marker_size": marker_size,
    }


def _text_right_of(marker_line: dict, bodies: list[dict], claimed: set[int]) -> tuple[str, tuple] | None:
    """The text a DETACHED marker labels: the nearest run of spans beginning to
    the right of it on the same row.

    Span-level rather than line-level because a table row's other cells extract
    into the same line as the step text beside them - Staging's "Manual
    Disable Set time automatically." is one line whose first span is the
    neighbouring cell, so taking the whole line would compare the wrong text.
    """
    mx1 = marker_line["bbox"][2]
    my0, my1 = marker_line["bbox"][1], marker_line["bbox"][3]
    best: tuple[float, dict, list[dict]] | None = None
    for line in bodies:
        if id(line) in claimed:
            continue
        by0, by1 = line["bbox"][1], line["bbox"][3]
        overlap = min(my1, by1) - max(my0, by0)
        if overlap <= 0.4 * min(my1 - my0, by1 - by0):
            continue  # not on the marker's own row
        spans = [sp for sp in line["spans"] if sp["bbox"][0] >= mx1 - 1]
        if not spans:
            continue
        gap = spans[0]["bbox"][0] - mx1
        if gap > MARKER_TEXT_MAX_GAP:
            continue
        if best is None or gap < best[0]:
            best = (gap, line, spans)
    if best is None:
        return None
    _, line, spans = best
    claimed.add(id(line))
    bbox = (
        min(marker_line["bbox"][0], spans[0]["bbox"][0]),
        min(my0, line["bbox"][1]),
        max(sp["bbox"][2] for sp in spans),
        max(my1, line["bbox"][3]),
    )
    return "".join(sp["text"] for sp in spans), bbox


def _page_list_items(lines: list[dict], page: int) -> list[dict]:
    markers: list[tuple[dict, str]] = []
    bodies: list[dict] = []
    for line in lines:
        text = " ".join(line["text"].split())
        m = _MARKER_ONLY_RE.match(text)
        if m:
            markers.append((line, m.group(1)))
        else:
            bodies.append(line)

    items: list[dict] = []
    claimed: set[int] = set()
    # Detached markers first: each claims one specific line, and claiming it
    # also stops that line being re-read below as an item of its own.
    for marker_line, marker in markers:
        found = _text_right_of(marker_line, bodies, claimed)
        if not found:
            continue
        marker_size = marker_line["spans"][0]["size"] if marker_line["spans"] else 0.0
        item = _make_item(page, marker, found[0], found[1], marker_size)
        if item:
            items.append(item)
    for line in bodies:
        if id(line) in claimed:
            continue
        text = " ".join(line["text"].split())
        m = _LIST_MARKER_RE.match(text)
        if not m:
            continue
        marker_size = line["spans"][0]["size"] if line["spans"] else 0.0
        item = _make_item(page, m.group(1), text[m.end():], tuple(line["bbox"]), marker_size)
        if item:
            items.append(item)
    return items


def _list_items(cache: _LineCache, blocks: list[dict]) -> list[dict]:
    """Every list item in this section, as {page, marker, kind, text, key, bbox}.

    Built from LINES, with a detached marker joined to the text it labels,
    because the two documents disagree about where a marker even lives:
    Production sets "1.\t Disable Set time automatically." as one line, while
    Staging draws "a." as a separate text object beside it. Both have to reduce
    to the same item before the marker style can be compared at all.
    """
    bands: dict[int, list[float]] = {}
    for b in blocks:
        page, bbox = b.get("page"), b.get("bbox")
        if page is None or not bbox:
            continue
        band = bands.get(page)
        if band is None:
            bands[page] = [bbox[1], bbox[3]]
        else:
            band[0] = min(band[0], bbox[1])
            band[1] = max(band[1], bbox[3])

    items: list[dict] = []
    for page, (top, bottom) in sorted(bands.items()):
        lines = [
            ln for ln in cache.lines(page)
            if top - 1 <= (ln["bbox"][1] + ln["bbox"][3]) / 2 <= bottom + 1
        ]
        items.extend(_page_list_items(lines, page))
    items.sort(key=lambda it: (it["page"], round(it["bbox"][1], 1), it["bbox"][0]))
    return items


def _pair_list_items(exp_items: list[dict], act_items: list[dict]) -> list[tuple[dict, dict]]:
    """Pair items whose text is identical, and only where that text occurs once
    on each side - the same "never guess" rule `_pair_blocks` applies. A
    mispaired item would report a renumbering that never happened.
    """
    def unique_index(items: list[dict]) -> dict[str, dict]:
        seen: dict[str, dict | None] = {}
        for it in items:
            seen[it["key"]] = None if it["key"] in seen else it
        return {k: v for k, v in seen.items() if v is not None}

    exp_index = unique_index(exp_items)
    act_index = unique_index(act_items)
    pairs = [(exp_index[k], act_index[k]) for k in exp_index.keys() & act_index.keys()]
    pairs.sort(key=lambda p: (p[0]["page"], round(p[0]["bbox"][1], 1)))
    return pairs


def _restarts_numbering(marker: str, kind: str) -> bool:
    """True when this marker is the FIRST value of its kind ("1.", "a.", "i.")
    - i.e. a new list starts here rather than the previous one continuing. A
    table of procedures stacks several short lists one under the other with no
    more whitespace between them than between their own items, so the restart,
    not the gap, is what tells them apart.
    """
    value = marker.rstrip(".)").strip().casefold()
    if kind == "number":
        return value == "1"
    if kind == "letter":
        return value == "a"
    if kind == "roman":
        return value == "i"
    return False  # a bullet has no sequence to restart


def _marker_runs(pairs: list[tuple[dict, dict]]) -> list[list[tuple[dict, dict]]]:
    """Group changed items back into the list they came from, so a five-step
    procedure renumbered 1-5 -> a-e is ONE finding showing the whole list
    rather than five the reader has to reassemble.
    """
    runs: list[list[tuple[dict, dict]]] = []
    for exp_it, act_it in pairs:
        run = runs[-1] if runs else None
        # Consecutive lines' bboxes overlap slightly (one line's descenders
        # reach below the next line's ascenders), so the gap test has to allow
        # a small negative value or every item starts its own "run".
        gap = exp_it["bbox"][1] - run[-1][0]["bbox"][3] if run else None
        if (
            run
            and run[-1][0]["page"] == exp_it["page"]
            and run[-1][0]["kind"] == exp_it["kind"]
            and run[-1][1]["kind"] == act_it["kind"]
            and -LINE_OVERLAP_SLACK <= gap <= LIST_RUN_MAX_GAP
            and not _restarts_numbering(exp_it["marker"], exp_it["kind"])
        ):
            run.append((exp_it, act_it))
        else:
            runs.append([(exp_it, act_it)])
    return runs


# How each marker kind is spoken about in the finding.
_MARKER_KIND_VERB = {
    "number": "numbers",
    "letter": "letters",
    "roman": "numbers with roman numerals",
    "bullet": "bullets",
}
_MARKER_KIND_ADJ = {"number": "numbered", "letter": "lettered", "roman": "roman", "bullet": "bulleted"}


def _marker_summary(items: list[dict]) -> str:
    shown = [it["marker"] for it in items[:6]]
    listed = ", ".join(shown) + (" …" if len(items) > len(shown) else "")
    return f"{listed} ({_MARKER_KIND_ADJ[items[0]['kind']]})"


def _marker_range(items: list[dict]) -> str:
    """"1.–5." for a run, just "3." for a single item."""
    first, last = items[0]["marker"], items[-1]["marker"]
    return first if first == last else f"{first}\u2013{last}"


def _run_anchor(items: list[dict]) -> dict:
    """The page and region a run of list items covers, for the screenshot."""
    page = items[0]["page"]
    boxes = [it["bbox"] for it in items if it["page"] == page]
    return {
        "page": page,
        "bbox": (
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ),
    }


def _check_list_markers(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_blocks: list[dict],
    act_blocks: list[dict],
    exp_lines: _LineCache,
    act_lines: _LineCache,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    """The same list item marked differently on each side - Production numbers
    a procedure 1., 2., 3. where Staging letters it a., b., c. or drops it to a
    bullet.

    Reported as a mismatch rather than advisory layout drift: renumbering a
    procedure changes what the reader is told to do ("repeat step 3" no longer
    resolves), and unlike indent it cannot happen by accident between two
    exports of the same source.
    """
    pairs = [
        (exp_it, act_it)
        for exp_it, act_it in _pair_list_items(
            _list_items(exp_lines, exp_blocks), _list_items(act_lines, act_blocks)
        )
        if exp_it["kind"] != act_it["kind"]
    ]
    for run in _marker_runs(pairs):
        if not _budget(counts, "marker"):
            return
        exp_items = [e for e, _ in run]
        act_items = [a for _, a in run]
        exp_anchor = _run_anchor(exp_items)
        act_anchor = _run_anchor(act_items)
        details = _details(exp_anchor, act_anchor, heading)
        details["expected_marker"] = _marker_summary(exp_items)
        details["actual_marker"] = _marker_summary(act_items)
        details["items"] = len(run)
        count = len(run)
        details["changed"] = (
            f"Production {_MARKER_KIND_VERB[exp_items[0]['kind']]} "
            f"{'this' if count == 1 else 'these'} {count} "
            f"{'item' if count == 1 else 'items'} ({_marker_range(exp_items)}); "
            f"Staging {_MARKER_KIND_VERB[act_items[0]['kind']]} "
            f"{'it' if count == 1 else 'them'} ({_marker_range(act_items)})"
        )
        details["text"] = [it["text"] for it in exp_items[:5]]
        _attach(details, expected, actual, output_dir, counter, exp_anchor, act_anchor)
        result.issues.append(
            Issue(
                severity="error",
                page=exp_anchor["page"],
                message="List marker changed",
                details=details,
            )
        )


# Relative marker font-size change big enough to flag - a marker drawn at
# ~1.15x/0.85x its counterpart's size no longer reads as "the same bullet",
# even though the item's own text (and body-text font-size noise generally)
# is deliberately not compared, see content.py's `_check_text_style`.
_MARKER_SIZE_RATIO = 0.15


def _marker_size_changed(exp_it: dict, act_it: dict) -> bool:
    exp_size, act_size = exp_it.get("marker_size"), act_it.get("marker_size")
    if not exp_size or not act_size:
        return False
    return abs(act_size - exp_size) / exp_size >= _MARKER_SIZE_RATIO


def _size_runs(pairs: list[tuple[dict, dict]]) -> list[list[tuple[dict, dict]]]:
    """Group changed items back into the list they came from, same rule as
    `_marker_runs` but for a size change rather than a kind change."""
    runs: list[list[tuple[dict, dict]]] = []
    for exp_it, act_it in pairs:
        run = runs[-1] if runs else None
        gap = exp_it["bbox"][1] - run[-1][0]["bbox"][3] if run else None
        if (
            run
            and run[-1][0]["page"] == exp_it["page"]
            and run[-1][0]["kind"] == exp_it["kind"]
            and -LINE_OVERLAP_SLACK <= gap <= LIST_RUN_MAX_GAP
            and not _restarts_numbering(exp_it["marker"], exp_it["kind"])
        ):
            run.append((exp_it, act_it))
        else:
            runs.append([(exp_it, act_it)])
    return runs


def _check_marker_size(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_blocks: list[dict],
    act_blocks: list[dict],
    exp_lines: _LineCache,
    act_lines: _LineCache,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    """The same list item's marker (bullet, number, letter) prints at a
    different size in Staging than in Production - e.g. a bullet drawn much
    larger or smaller than its Production counterpart, most often from a
    mismatched list style rather than a body-text-wide font change (that shows
    up, if at all, as a font-size difference on the matched item text, which
    is deliberately not reported - see content.py). Only pairs whose marker
    KIND still matches are considered here; a kind change is
    `_check_list_markers`'s finding, not this one.
    """
    pairs = [
        (exp_it, act_it)
        for exp_it, act_it in _pair_list_items(
            _list_items(exp_lines, exp_blocks), _list_items(act_lines, act_blocks)
        )
        if exp_it["kind"] == act_it["kind"] and _marker_size_changed(exp_it, act_it)
    ]
    for run in _size_runs(pairs):
        if not _budget(counts, "marker_size"):
            return
        exp_items = [e for e, _ in run]
        act_items = [a for _, a in run]
        exp_anchor = _run_anchor(exp_items)
        act_anchor = _run_anchor(act_items)
        details = _details(exp_anchor, act_anchor, heading)
        details["expected_marker"] = _marker_summary(exp_items)
        details["actual_marker"] = _marker_summary(act_items)
        details["items"] = len(run)
        details["changed"] = (
            f"Production's {_MARKER_KIND_ADJ[exp_items[0]['kind']]} marker prints at "
            f"{exp_items[0]['marker_size']:.1f}pt; Staging's prints at {act_items[0]['marker_size']:.1f}pt"
        )
        details["text"] = [it["text"] for it in exp_items[:5]]
        _attach(details, expected, actual, output_dir, counter, exp_anchor, act_anchor)
        result.issues.append(
            Issue(
                severity="warning",
                page=exp_anchor["page"],
                message="List marker size changed",
                details=details,
            )
        )


def _check_indent(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_b: dict,
    act_b: dict,
    exp_lines: _LineCache,
    act_lines: _LineCache,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    """A list item whose indent moved - it changed nesting level, or lost its
    hanging indent. Only checked on blocks that actually are list items, since
    an ordinary paragraph's first-line offset carries no such meaning.
    """
    if not _LIST_MARKER_RE.match(exp_b["text"]):
        return
    exp_indent = _indent_of(exp_lines, exp_b["page"], exp_b["bbox"])
    act_indent = _indent_of(act_lines, act_b["page"], act_b["bbox"])
    if exp_indent is None or act_indent is None:
        return
    if abs(exp_indent - act_indent) <= INDENT_TOLERANCE or not _budget(counts, "indent"):
        return
    details = _details(exp_b, act_b, heading)
    details["expected_indent_pt"] = exp_indent
    details["actual_indent_pt"] = act_indent
    details["shift_pt"] = round(act_indent - exp_indent, 1)
    details["text"] = [exp_b["text"]]
    _attach(details, expected, actual, output_dir, counter, exp_b, act_b)
    result.issues.append(
        Issue(
            severity="warning", page=exp_b["page"], message="List alignment/indent changed", details=details
        )
    )


def _check_text_alignment(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_b: dict,
    act_b: dict,
    exp_lines: _LineCache,
    act_lines: _LineCache,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    exp_align = _alignment_of(exp_lines, exp_b["page"], exp_b["bbox"])
    act_align = _alignment_of(act_lines, act_b["page"], act_b["bbox"])
    if exp_align == act_align or "unknown" in (exp_align, act_align):
        return
    if not _budget(counts, "align"):
        return
    details = _details(exp_b, act_b, heading)
    details["expected_alignment"] = exp_align
    details["actual_alignment"] = act_align
    details["text"] = [exp_b["text"]]
    _attach(details, expected, actual, output_dir, counter, exp_b, act_b)
    result.issues.append(
        Issue(severity="warning", page=exp_b["page"], message="Text alignment changed", details=details)
    )


def _check_merged_heading(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    pairs: list[tuple[dict, dict]],
    exp_lines: _LineCache,
    act_lines: _LineCache,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    """A numbered item's bold label stands alone on its own line in Production
    but in Staging the description has been pulled up onto that same line, so
    "1. Brightness" / "Adjusts the ..." becomes "1. Brightness Adjusts the ...".
    Detected structurally: the Staging line starts bold and continues non-bold,
    where the matching Production line was bold all the way across.
    """
    for exp_b, act_b in pairs:
        exp_labels = {
            _line_key(ln): ln
            for ln in exp_lines.lines_in(exp_b["page"], exp_b["bbox"])
            if _is_standalone_bold_label(ln)
        }
        if not exp_labels:
            continue
        for act_line in act_lines.lines_in(act_b["page"], act_b["bbox"]):
            label = _merged_label(act_line)
            if label is None or _line_key({"text": label}) not in exp_labels:
                continue
            if not _budget(counts, "merged"):
                return
            details = _details(exp_b, act_b, heading)
            details["label"] = label.strip()
            details["expected"] = [exp_labels[_line_key({"text": label})]["text"].strip()]
            details["actual"] = [act_line["text"].strip()]
            _attach(details, expected, actual, output_dir, counter, exp_b, act_b)
            result.issues.append(
                Issue(
                    severity="warning",
                    page=exp_b["page"],
                    message="Paragraph merged with heading",
                    details=details,
                )
            )


def _line_key(line: dict) -> str:
    return " ".join(line["text"].split()).lower()


def _is_standalone_bold_label(line: dict) -> bool:
    """A line that is a numbered label and nothing else, entirely in bold."""
    return bool(_NUMBERED_LABEL_RE.match(line["text"])) and all(s["bold"] for s in line["spans"])


def _merged_label(line: dict) -> str | None:
    """If this line starts with a bold numbered label and then continues in
    non-bold text, return just the bold label part.
    """
    if not _NUMBERED_LABEL_RE.match(line["text"]) or not line["spans"][0]["bold"]:
        return None
    bold_prefix = []
    for span in line["spans"]:
        if not span["bold"]:
            break
        bold_prefix.append(span["text"])
    if len(bold_prefix) == len(line["spans"]):
        return None  # the whole line is bold - nothing was merged in
    label = "".join(bold_prefix).strip()
    return label if label else None


def _attach(
    details: dict,
    expected: fitz.Document,
    actual: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
    exp_b: dict,
    act_b: dict,
) -> None:
    if not output_dir:
        return
    seq = next(counter)
    if seq > MAX_SCREENSHOTS:
        return
    label = str(seq)  # the same number on both crops - one block, two layouts
    details["shot_label"] = label
    prod = screenshots.capture_region(
        expected, exp_b["page"], output_dir, f"align_{seq}_prod", exp_b["bbox"],
        screenshots.KIND_DIFF, label,
    )
    stage = screenshots.capture_region(
        actual, act_b["page"], output_dir, f"align_{seq}_stage", act_b["bbox"],
        screenshots.KIND_DIFF, label,
    )
    if prod:
        details["prod_screenshot"] = prod
    if stage:
        details["stage_screenshot"] = stage
