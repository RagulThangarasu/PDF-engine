"""5. Alignment validation: how text is laid out, as opposed to what it says.

Findings:

* List marker changed          - 1,2,3 vs a,b,c vs bullets
* List alignment/indent changed - a list item's indent moved
* Text alignment changed       - left/center/right/justified changed
* Paragraph merged with heading - a numbered item's description glued onto the
                                  same line as its bold label

Everything here compares blocks whose TEXT is identical on both sides, so a
finding is always "the same words, laid out differently" and never a disguised
content difference - those belong to Content Validation.
"""
from __future__ import annotations

import itertools
import re

import fitz

from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.toc import (
    extract_section_blocks,
    get_toc_entries,
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
_LIST_MARKER_RE = re.compile(
    r"^\s*(\d{1,2}[.)]|[a-zA-Z][.)]|(?:" + _ROMAN_NUMERAL_MARKERS + r")[.)]|[•◦▪▸‣⁃])\s+",
    re.IGNORECASE,
)

# A numbered item's bold label, e.g. "1. Brightness" or "3) Input Source" - the
# part that is supposed to stand alone on its own line.
_NUMBERED_LABEL_RE = re.compile(r"^\s*\d{1,2}[.)]\s+\S")


def classify_marker(marker: str) -> str:
    if re.match(r"^\d{1,2}[.)]$", marker):
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

    exp_entries = get_toc_entries(expected)
    act_entries = get_toc_entries(actual)

    if exp_entries and act_entries:
        matches = match_toc_entries(exp_entries, act_entries)
        matched = [m for m in matches if m.expected_index is not None and m.actual_index is not None]
        exp_sections = extract_section_blocks(expected, [exp_entries[m.expected_index] for m in matched])
        act_sections = extract_section_blocks(actual, [act_entries[m.actual_index] for m in matched])
        for idx, m in enumerate(matched):
            _compare_section(
                result, expected, actual, exp_sections[idx], act_sections[idx],
                exp_entries[m.expected_index].title, output_dir, counter, counts,
            )
    else:
        _compare_section(
            result, expected, actual, _all_blocks(expected), _all_blocks(actual),
            None, output_dir, counter, counts,
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
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    exp_lines = _LineCache(expected)
    act_lines = _LineCache(actual)
    pairs = _pair_blocks(exp_blocks, act_blocks)
    pairs.sort(key=lambda p: (p[0]["page"], p[0]["bbox"][1]))

    for exp_b, act_b in pairs:
        _check_list_marker(result, expected, actual, exp_b, act_b, heading, output_dir, counter, counts)
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


def _check_list_marker(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_b: dict,
    act_b: dict,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    counts: dict[str, int],
) -> None:
    """The same list item styled with a different marker kind - Production
    numbers it 1,2,3 while Staging uses a,b,c or a bullet.
    """
    exp_m = _LIST_MARKER_RE.match(exp_b["text"])
    act_m = _LIST_MARKER_RE.match(act_b["text"])
    if not exp_m or not act_m:
        return
    exp_kind = classify_marker(exp_m.group(1))
    act_kind = classify_marker(act_m.group(1))
    if exp_kind == act_kind or not _budget(counts, "marker"):
        return
    details = _details(exp_b, act_b, heading)
    details["expected_marker"] = f"{exp_m.group(1)} ({exp_kind})"
    details["actual_marker"] = f"{act_m.group(1)} ({act_kind})"
    details["text"] = [exp_b["text"]]
    _attach(details, expected, actual, output_dir, counter, exp_b, act_b)
    result.issues.append(
        Issue(severity="warning", page=exp_b["page"], message="List marker changed", details=details)
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
    prod = screenshots.capture_region(
        expected, exp_b["page"], output_dir, f"align_{seq}_prod", exp_b["bbox"]
    )
    stage = screenshots.capture_region(
        actual, act_b["page"], output_dir, f"align_{seq}_stage", act_b["bbox"]
    )
    if prod:
        details["prod_screenshot"] = prod
    if stage:
        details["stage_screenshot"] = stage
