"""4. Table validation.

Findings, all one-directional (a loss or a defect in STAGE, never an addition):

* Table header row not repeated on continuation page
                             - a STAGE table runs onto another page and that
                               page does not reprint the header row. Mandatory,
                               and therefore checked against STAGE ALONE,
                               document-wide: it is a property of the document
                               being validated, not a difference between the
                               two, so it holds whether or not the table pairs
                               with a Production one and whether or not
                               Production drops the header too.
* Table split across pages   - a page break splits a table that wasn't split in
                               Production
* Table columns differ       - the two tables don't have the same column count
* Table column layout differs - same columns, different column widths/positions
* Table cell layout differs  - a row's cells merged or split
* Table breaking the margins - the table extends past the page margins
* Table heading missing      - a header cell present in Production is gone
* Table row missing          - a whole row present in Production is gone
* Table cell missing         - part of a cell's content is gone

Rows are NOT matched by position. A row is found by its ANCHOR cell - the value
in the most distinctive other column - and then only the target cell's own
tokens are compared, as an order-insensitive multiset. Positional row matching
reported every row below an inserted one as changed, and comparing cells as
ordered strings reported a reordered "1920 x 1080 / 60 Hz" against
"60 Hz / 1920 x 1080" as a difference when nothing was lost.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter

import fitz

from pdfval.extractor import get_tables
from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import (
    get_toc_entries,
    heading_at,
    in_section_bounds,
    matched_heading_ranges,
    normalize_block_text,
)

MAX_CELL_ISSUES_PER_TABLE = 30
MAX_SCREENSHOTS = 300  # cap total screenshot pairs so a huge document doesn't stall the report
PAGE_MISMATCH_PENALTY = 1000.0  # used only by the positional fallback matcher
HEADER_BAND_RATIO = 0.15  # near-top of page - a continued table's header row should reappear here
FOOTER_BAND_RATIO = 0.2  # near-bottom of page - a table this close to the bottom likely continues

# Two tables are treated as the same table (possibly edited) at or above this
# text similarity. An edited table still shares most of its text with its
# counterpart; two genuinely different tables under one heading do not.
TABLE_MATCH_SIMILARITY = 0.5
# The positional fallback may only pair tables that still share this much
# wording. Below it, the two are different tables and pairing them would report
# every row of both as changed.
FALLBACK_MIN_SIMILARITY = 0.25

# Column edges are compared as a fraction of table width, so a table that is
# merely a little wider in one document doesn't read as relaid-out.
COLUMN_LAYOUT_TOLERANCE = 0.03

# Recognising that one page's table continues onto the next: how close two
# fragments' column rules must fall (points) and how many of them must line up.
COLUMN_ALIGN_TOLERANCE = 4.0
COLUMN_ALIGN_MIN_SHARE = 0.7
# How much of the header row's wording a continuation page must reprint for the
# header to count as repeated.
HEADER_REPEAT_SIMILARITY = 0.7

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def validate_tables(
    expected: fitz.Document, actual: fitz.Document, expected_path: str, actual_path: str, output_dir: str | None = None
) -> CheckResult:
    result = CheckResult(name="Table Validation")
    exp_entries, act_entries = resolve_entries(expected, actual)
    counter = itertools.count(1)
    summary: list[dict] = []
    margins = _document_margins(actual)

    # Run document-wide, not per section: a repeated header is mandatory on
    # every Staging table that spans a page break, and a table sitting under a
    # heading that has no counterpart in Production would never be looked at by
    # the section walk below.
    _check_stage_continuation_headers(
        result, actual, actual_path, act_entries, output_dir, counter
    )

    if exp_entries and act_entries:
        # Heading-anchored: page-index matching (page N vs page N) silently
        # compares the wrong pages once the two documents' page counts diverge.
        for exp_entry, act_entry, exp_bounds, act_bounds in matched_heading_ranges(
            exp_entries, act_entries, expected.page_count, actual.page_count
        ):
            exp_tables = _tables_in_bounds(expected_path, exp_bounds, expected)
            act_tables = _tables_in_bounds(actual_path, act_bounds, actual)
            _check_margins(result, actual, act_tables, exp_entry.title, output_dir, counter, margins)
            # A long table is extracted as a separate fragment per page, and the
            # two documents don't necessarily paginate it the same way, so
            # continuation fragments are stitched back into one logical table
            # before anything is compared.
            exp_tables = _stitch_continued_tables(expected, exp_tables)
            act_tables = _stitch_continued_tables(actual, act_tables)
            act_section_tokens = _section_tokens(
                actual, range(max(0, act_bounds.start_page), min(act_bounds.end_page, actual.page_count - 1) + 1)
            )
            _compare_section_tables(
                result, expected, actual, exp_tables, act_tables, exp_entry.title, output_dir, counter,
                act_section_tokens, summary,
            )
    else:
        _validate_tables_by_page_index(
            result, expected, actual, expected_path, actual_path, output_dir, counter, summary, margins
        )

    result.summary_rows = summary
    result.summary_title = "Table-by-table comparison (Prod vs Stage)"
    return result


def _check_stage_continuation_headers(
    result: CheckResult,
    actual: fitz.Document,
    actual_path: str,
    act_entries: list,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """The mandatory repeated-header rule, applied to every Staging table."""
    all_tables: list[tuple[int, dict]] = []
    for page in range(actual.page_count):
        all_tables.extend((page, t) for t in get_tables(actual_path, page, actual))
    stitched = _stitch_continued_tables(actual, all_tables)
    for anchor_page, t in stitched:
        if not (t.get("continuations") or []):
            continue
        heading = heading_at(act_entries, anchor_page, t["bbox"][1]) if act_entries else None
        _check_continuation_headers(result, actual, [(anchor_page, t)], heading, output_dir, counter)


def _tables_in_bounds(pdf_path: str, bounds, doc: fitz.Document) -> list[tuple[int, dict]]:
    items: list[tuple[int, dict]] = []
    for p in range(max(0, bounds.start_page), min(bounds.end_page, doc.page_count - 1) + 1):
        items.extend(
            (p, t) for t in get_tables(pdf_path, p, doc) if in_section_bounds(p, t["bbox"][1], bounds)
        )
    return items


def _column_positions(t: dict) -> list[float]:
    """The table's column edges in absolute page coordinates."""
    edges: set[float] = set()
    for row in t.get("cells") or []:
        for cell in row or []:
            if cell:
                edges.add(round(cell[0], 1))
                edges.add(round(cell[2], 1))
    return sorted(edges)


def _columns_line_up(prev_t: dict, cur_t: dict) -> bool:
    """Do the two fragments rule their columns in the same places?

    Column COUNT alone is not a safe test for a continuation, and this is
    exactly why: a table's column boundaries are often defined by its header
    row, so a continuation page that FAILS to repeat the header is frequently
    the one pdfplumber reads with a different number of columns. Requiring equal
    counts would make the check blind to the defect it exists to find, so the
    fragments are matched on where their rules actually fall instead.
    """
    prev_edges, cur_edges = _column_positions(prev_t), _column_positions(cur_t)
    if not prev_edges or not cur_edges:
        return False
    shared = sum(
        1 for e in prev_edges if any(abs(e - other) <= COLUMN_ALIGN_TOLERANCE for other in cur_edges)
    )
    return shared / max(len(prev_edges), len(cur_edges)) >= COLUMN_ALIGN_MIN_SHARE


def _is_continuation(doc: fitz.Document, prev_page: int, prev_t: dict, cur_page: int, cur_t: dict) -> bool:
    """A table runs to near the bottom of a page and another table starts near
    the top of the very next page over the same columns - the same logical table
    continuing rather than two distinct tables.
    """
    if cur_page != prev_page + 1 or cur_page >= doc.page_count:
        return False
    prev_rect = doc[prev_page].rect
    cur_rect = doc[cur_page].rect
    if prev_t["bbox"][3] < prev_rect.y1 - prev_rect.height * FOOTER_BAND_RATIO:
        return False  # previous table doesn't reach the bottom - not a continuation
    if cur_t["bbox"][1] > cur_rect.y0 + cur_rect.height * HEADER_BAND_RATIO:
        return False  # this table doesn't start near the top - not a continuation
    prev_rows, cur_rows = prev_t["rows"], cur_t["rows"]
    prev_cols = len(prev_rows[0]) if prev_rows else 0
    cur_cols = len(cur_rows[0]) if cur_rows else 0
    if prev_cols and prev_cols == cur_cols:
        return True
    return _columns_line_up(prev_t, cur_t)


def _headers_match(header: str, candidate: str) -> bool:
    """Is `candidate` a repeat of the table's header row?

    Compared as a case-folded token multiset rather than as an exact string: a
    header re-typeset on the continuation page legitimately re-wraps, and a cell
    that wrapped differently would otherwise read as a missing header. What has
    to survive is the header's WORDS.
    """
    want = Counter(_TOKEN_RE.findall(header.lower()))
    got = Counter(_TOKEN_RE.findall(candidate.lower()))
    if not want:
        return True  # no header to repeat
    shared = sum((want & got).values())
    return shared / sum(want.values()) >= HEADER_REPEAT_SIMILARITY

# Some tables in these manuals carry their header across two rows - a title
# row ("PD2705U / PD2705UE / PD2705UA") followed by a column-label row
# ("USB-C(tm) 65W... / USB-C(tm)...") - and a continuation page can reprint
# either just the first or both. Capped at 2: no real-world case in these
# manuals has been seen with a 3+ row header, and an unbounded scan risks
# eating into genuine data rows that happen to token-overlap a header.
_MAX_HEADER_ROWS = 2


def _repeated_leading_rows(anchor_rows: list, cont_rows: list) -> int:
    """How many of the continuation's leading rows repeat the anchor table's
    own leading rows, position by position (row 0 vs row 0, row 1 vs row 1,
    ...) - not just the first. Stripping only row 0 when a continuation
    reprints BOTH header rows leaves the second one behind as a bogus extra
    "data" row, shifting every real row after it by one.
    """
    count = 0
    for i in range(min(_MAX_HEADER_ROWS, len(anchor_rows), len(cont_rows))):
        if not _headers_match(_row_text(anchor_rows[i]), _row_text(cont_rows[i])):
            break
        count += 1
    return count


def _stitch_continued_tables(doc: fitz.Document, tables: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Merge a table's per-page fragments into one logical table (rows
    concatenated), recording what each fragment did with the header row.

    `fragments` is how many pages the table spans. `continuations` describes
    every page after the first - its page, the row it actually starts with, and
    whether that row repeats the header - which is what the mandatory
    repeated-header check reads.
    """
    groups: list[list[tuple[int, dict]]] = []
    for page, t in tables:
        if groups:
            last_page, last_t = groups[-1][-1]
            if _is_continuation(doc, last_page, last_t, page, t):
                groups[-1].append((page, t))
                continue
        groups.append([(page, t)])

    stitched: list[tuple[int, dict]] = []
    for group in groups:
        anchor_page, anchor_t = group[0]
        header = _row_text(anchor_t["rows"][0]) if anchor_t["rows"] else ""
        if len(group) == 1:
            stitched.append(
                (
                    anchor_page,
                    {**anchor_t, "fragments": 1, "continuations": [], "header": header, "last_page": anchor_page},
                )
            )
            continue
        merged_rows: list = []
        merged_cells: list = []
        continuations: list[dict] = []
        for i, (page, t) in enumerate(group):
            rows = t["rows"] or []
            cells = t.get("cells") or []
            if i == 0:
                merged_rows.extend(rows)
                merged_cells.extend(cells)
                continue
            first_row = _row_text(rows[0] if rows else None)
            header_repeated = _headers_match(header, first_row)
            continuations.append(
                {
                    "page": page,
                    "bbox": t["bbox"],
                    "first_row": first_row,
                    "header_repeated": header_repeated,
                }
            )
            # A correctly repeated header row is the SAME logical row as row 0,
            # not a new one - a table that continues onto another page is
            # SUPPOSED to reprint it there (see `_check_continuation_headers`),
            # so counting it again here would misalign every row after it
            # against Production's un-split table and read as spurious row/
            # cell/merge differences purely because Staging paginated
            # differently. Only dropped when it actually WAS recognized as the
            # header; a fragment that drops the header (already flagged on its
            # own as a defect) or starts straight into data keeps its first row.
            # Some tables carry their header across TWO rows (a title row plus
            # a column-label row) and a continuation can reprint both, not just
            # the first - stripping only row 0 in that case leaves the
            # reprinted column-label row behind as a bogus extra "data" row,
            # shifting every real row after it by one and misaligning any
            # position-based comparison against Production's un-split table.
            start = _repeated_leading_rows(anchor_t["rows"], rows) if header_repeated else 0
            merged_rows.extend(rows[start:])
            merged_cells.extend(cells[start:])
        stitched.append(
            (
                anchor_page,
                {
                    "rows": merged_rows,
                    "cells": merged_cells,
                    "bbox": anchor_t["bbox"],
                    "fragments": len(group),
                    "continuations": continuations,
                    "header": header,
                    "last_page": group[-1][0],
                },
            )
        )
    return stitched


def _missing_header_pages(t: dict) -> list[int]:
    return [c["page"] for c in t.get("continuations") or [] if not c["header_repeated"]]


def _attach_screenshots(
    details: dict,
    expected: fitz.Document,
    actual: fitz.Document,
    output_dir: str | None,
    counter: "itertools.count",
    exp_page: int | None,
    exp_bbox: tuple[float, float, float, float] | None,
    act_page: int | None,
    act_bbox: tuple[float, float, float, float] | None,
) -> None:
    """Render the table's real appearance from each document, zoomed to the
    table, so the finding can be checked by eye. A side with nothing to point
    at gets no screenshot rather than an unrelated page.
    """
    if not output_dir:
        return
    seq = next(counter)
    if seq > MAX_SCREENSHOTS:
        return
    # Both crops carry the finding's number so the reader can tell at a glance
    # that they are the same table on the two sides.
    label = str(seq)
    details["shot_label"] = label
    prod_path = (
        screenshots.capture_region(
            expected, exp_page, output_dir, f"table_{seq}_prod", exp_bbox,
            screenshots.KIND_DIFF, label,
        )
        if exp_page is not None
        else None
    )
    stage_path = (
        screenshots.capture_region(
            actual, act_page, output_dir, f"table_{seq}_stage", act_bbox,
            screenshots.KIND_DIFF if act_bbox is not None else screenshots.KIND_CONTEXT, label,
        )
        if act_page is not None
        else None
    )
    if prod_path:
        details["prod_screenshot"] = prod_path
    if stage_path:
        details["stage_screenshot"] = stage_path


def _table_text(t: dict) -> str:
    return "\n".join(_row_text(row) for row in (t["rows"] or []))


def _match_tables(exp_tables: list[tuple[int, dict]], act_tables: list[tuple[int, dict]]) -> dict[int, int]:
    """Pair tables across the two documents by their CONTENT, best match first.

    Position-based matching (with a per-page penalty large enough that a
    cross-page candidate could never win) reported a table that had simply moved
    to another page - because content above it reflowed - as one missing plus
    one added table. Content matching finds it wherever in the section it ended
    up.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, (_, exp_t) in enumerate(exp_tables):
        exp_tokens = _table_tokens(exp_t)
        exp_cols = _col_count(exp_t)
        for j, (_, act_t) in enumerate(act_tables):
            ratio = _table_similarity(exp_tokens, _table_tokens(act_t))
            if ratio < TABLE_MATCH_SIMILARITY:
                continue
            bonus = 0.05 if exp_cols == _col_count(act_t) else 0.0
            candidates.append((ratio + bonus, i, j))
    candidates.sort(key=lambda c: -c[0])

    pairs: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in candidates:
        if i in pairs or j in used:
            continue
        pairs[i] = j
        used.add(j)

    # Whatever content matching couldn't place falls back to shape-then-position
    # so an entirely rewritten table still pairs with its counterpart - but only
    # if it still shares SOME wording. Without that floor the fallback pairs
    # whatever is left over regardless of content, which is how a table got
    # matched against something entirely unrelated and had its whole contents
    # reported as changed. A table with no plausible counterpart is genuinely
    # gone, and saying so is the correct answer.
    for i, (exp_page, exp_t) in enumerate(exp_tables):
        if i in pairs:
            continue
        idx = _find_closest_table(exp_page, exp_t, act_tables, used)
        if idx is None:
            continue
        if _table_similarity(_table_tokens(exp_t), _table_tokens(act_tables[idx][1])) < FALLBACK_MIN_SIMILARITY:
            continue
        pairs[i] = idx
        used.add(idx)
    return pairs


# How much of a table's wording must still be present in the Staging section
# for the table to count as "there, just not detected as a table".
TEXT_SURVIVAL_THRESHOLD = 0.6


def _text_survives(exp_t: dict, act_section_tokens: Counter) -> bool:
    tokens = _table_tokens(exp_t)
    total = sum(tokens.values())
    if not total:
        return True  # nothing to lose
    shared = sum((tokens & act_section_tokens).values())
    return (shared / total) >= TEXT_SURVIVAL_THRESHOLD


def _section_tokens(doc: fitz.Document, pages) -> Counter:
    """All the words in a run of pages, as a multiset."""
    counts: Counter = Counter()
    for p in pages:
        if 0 <= p < doc.page_count:
            try:
                counts.update(_TOKEN_RE.findall(doc[p].get_text().lower()))
            except Exception:
                pass
    return counts


def _table_tokens(t: dict) -> Counter:
    """Tokens used to decide WHICH tables correspond across the two documents -
    deliberately excluding the header row.

    These manuals repeat one column-label row ("Model | ST4304 | ST5504 |
    ST6504") verbatim across every spec table in a section. Counting it here
    means two genuinely UNRELATED small tables that both carry that header
    look like a strong match purely from the header's word overlap - and the
    smaller the rest of a table's content, the more that shared header
    dominates its score, so the smallest table in a section becomes a false
    magnet for every other table's fallback match. The header's words carry no
    information about which table an OTHER table corresponds to, so they are
    left out of the matching signal; the full text (header included) is still
    used everywhere else - the actual cell/row diff, screenshots, etc.
    """
    rows = t.get("rows") or []
    body = rows[1:] if len(rows) > 1 else rows
    text = "\n".join(_row_text(r) for r in body)
    return Counter(_TOKEN_RE.findall(text.lower()))


def _table_similarity(a: Counter, b: Counter) -> float:
    """How much of the larger table's wording the two tables share.

    Compared as token multisets rather than as character sequences.
    `SequenceMatcher.quick_ratio`, used here originally, only counts characters
    in common and ignores order completely - it is an upper bound on
    similarity, not a measure of it - so an on-screen-display menu listing
    ("Mode ON Color Gamut DCI-P3 Gamma 2.6") scored as a strong match against a
    real settings table that happened to reuse the same vocabulary.
    """
    if not a or not b:
        return 0.0
    shared = sum((a & b).values())
    return shared / max(sum(a.values()), sum(b.values()))


def _col_count(t: dict) -> int:
    return len(t["rows"][0]) if t["rows"] else 0


def _compare_section_tables(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_tables: list[tuple[int, dict]],
    act_tables: list[tuple[int, dict]],
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    act_section_tokens: Counter,
    summary: list[dict],
) -> None:
    pairs = _match_tables(exp_tables, act_tables)

    for i, (exp_page, exp_t) in enumerate(exp_tables):
        idx = pairs.get(i)
        if idx is None:
            # Before calling a table gone, check whether its WORDS are still in
            # the Staging section. The two documents don't rule their tables the
            # same way, and pdfplumber finds tables by their rules - so a table
            # that Production draws with full borders and Staging draws
            # borderless is detected on one side only, and would otherwise be
            # reported as entirely missing while sitting in plain sight. Only a
            # table whose text is genuinely absent is reported.
            if _text_survives(exp_t, act_section_tokens):
                summary.append(_summary_row(heading, exp_page, exp_t, None, None, "Not detected in Stage"))
                continue
            details = _with_heading(
                {"bbox": exp_t["bbox"], "rows": len(exp_t["rows"]), "reason": "the whole table is gone"},
                heading,
            )
            _attach_screenshots(details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], None, None)
            result.issues.append(Issue(severity="error", page=exp_page, message="Table row missing", details=details))
            summary.append(_summary_row(heading, exp_page, exp_t, None, None, "Missing"))
            continue
        act_page, act_t = act_tables[idx]
        before = len(result.issues)
        _check_broken(result, expected, actual, exp_page, act_page, exp_t, act_t, heading, output_dir, counter)
        _compare_tables(result, expected, actual, exp_page, act_page, exp_t, act_t, heading, output_dir, counter)
        status = "OK" if len(result.issues) == before else "Issues"
        summary.append(_summary_row(heading, exp_page, exp_t, act_page, act_t, status))

    # Tables present only in STAGE are not reported as issues: an addition is an
    # editorial choice, and the spec for this engine is losses only. They still
    # earn a summary row, so the reader can see everything that was compared.
    for j, (act_page, act_t) in enumerate(act_tables):
        if j not in set(pairs.values()):
            summary.append(_summary_row(heading, None, None, act_page, act_t, "Stage only"))


def _summary_row(
    heading: str | None,
    exp_page: int | None,
    exp_t: dict | None,
    act_page: int | None,
    act_t: dict | None,
    status: str,
) -> dict:
    """One line of the report's table-by-table breakdown - including the tables
    that came through clean, which an issue list can never show."""
    def shape(t: dict | None) -> str:
        if not t:
            return "-"
        return f"{len(t['rows'] or [])} x {_col_count(t)}"

    def spanned(t: dict | None) -> str:
        if not t:
            return "-"
        n = t.get("fragments", 1)
        return "1 page" if n <= 1 else f"{n} pages"

    header_state = "-"
    if act_t is not None:
        conts = act_t.get("continuations") or []
        if conts:
            header_state = "Yes" if all(c["header_repeated"] for c in conts) else "NO"
    if header_state == "NO":
        # The repeated-header rule is checked document-wide rather than inside
        # the section comparison, so the row's status has to pick it up here or
        # a table with a dropped header would be summarised as clean.
        status = "Issues"
    return {
        "heading": heading or "(no heading)",
        "expected_page": (exp_page + 1) if exp_page is not None else None,
        "actual_page": (act_page + 1) if act_page is not None else None,
        "expected_shape": shape(exp_t),
        "actual_shape": shape(act_t),
        "expected_span": spanned(exp_t),
        "actual_span": spanned(act_t),
        "header_repeated": header_state,
        "status": status,
    }


def _validate_tables_by_page_index(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    expected_path: str,
    actual_path: str,
    output_dir: str | None,
    counter: "itertools.count",
    summary: list[dict],
    margins: tuple[float | None, float | None],
) -> None:
    """Fallback used only when one/both documents have no TOC to anchor by -
    only reliable while the two documents' page counts haven't drifted apart.
    """
    exp_entries = get_toc_entries(expected)
    act_entries = get_toc_entries(actual)
    common_pages = min(expected.page_count, actual.page_count)

    for i in range(common_pages):
        exp_tables = [(i, t) for t in get_tables(expected_path, i, expected)]
        act_tables = [(i, t) for t in get_tables(actual_path, i, actual)]
        heading = None
        if exp_tables:
            heading = heading_at(exp_entries, i, exp_tables[0][1]["bbox"][1])
        if not heading and act_tables:
            heading = heading_at(act_entries, i, act_tables[0][1]["bbox"][1])
        _check_margins(result, actual, act_tables, heading, output_dir, counter, margins)
        exp_tables = _stitch_continued_tables(expected, exp_tables)
        act_tables = _stitch_continued_tables(actual, act_tables)
        _compare_section_tables(
            result, expected, actual, exp_tables, act_tables, heading, output_dir, counter,
            _section_tokens(actual, [i]), summary,
        )


def _with_heading(details: dict, heading: str | None) -> dict:
    if heading:
        details = {"heading": heading, **details}
    return details


def _find_closest_table(exp_page: int, exp_t: dict, act_tables: list[tuple[int, dict]], used: set[int]) -> int | None:
    """Positional fallback. Prefers a candidate with the same column count - a
    heading can hold several genuinely distinct tables, and matching purely by
    nearest position can pair one document's description table against the
    other's unrelated settings matrix.
    """
    idx = _nearest_table_idx(exp_page, exp_t, act_tables, used, require_cols=_col_count(exp_t))
    if idx is not None:
        return idx
    return _nearest_table_idx(exp_page, exp_t, act_tables, used, require_cols=None)


def _nearest_table_idx(
    exp_page: int, exp_t: dict, act_tables: list[tuple[int, dict]], used: set[int], require_cols: int | None
) -> int | None:
    ex0, ey0 = exp_t["bbox"][0], exp_t["bbox"][1]
    best_idx, best_dist = None, None
    for idx, (act_page, act_t) in enumerate(act_tables):
        if idx in used:
            continue
        if require_cols is not None and _col_count(act_t) != require_cols:
            continue
        ax0, ay0 = act_t["bbox"][0], act_t["bbox"][1]
        dist = ((ex0 - ax0) ** 2 + (ey0 - ay0) ** 2) ** 0.5 + abs(exp_page - act_page) * PAGE_MISMATCH_PENALTY
        if best_dist is None or dist < best_dist:
            best_dist, best_idx = dist, idx
    return best_idx


def _check_broken(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_page: int,
    act_page: int,
    exp_t: dict,
    act_t: dict,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """A page break splits the table where Production kept it whole.

    The other way a table stops reading as one table - a continuation page that
    drops the header row - is NOT checked here. It is mandatory on Staging
    whatever Production does, so it is checked against Staging on its own (see
    `_check_continuation_headers`) rather than only on tables that happened to
    find a Production counterpart.
    """
    exp_fragments = exp_t.get("fragments", 1)
    act_fragments = act_t.get("fragments", 1)
    if act_fragments <= exp_fragments:
        return

    details = _with_heading(
        {
            "expected_page": exp_page + 1,
            "actual_page": act_page + 1,
            "reason": (
                f"a page break splits this table across {act_fragments} pages in Staging "
                f"but {exp_fragments} in Production"
            ),
            "expected_pages_spanned": exp_fragments,
            "actual_pages_spanned": act_fragments,
            "continues_onto_pages": ", ".join(
                str(c["page"] + 1) for c in act_t.get("continuations") or []
            ),
        },
        heading,
    )
    if act_t.get("header"):
        details["header_row"] = act_t["header"]
    _attach_screenshots(
        details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], act_page, act_t["bbox"]
    )
    result.issues.append(
        Issue(severity="error", page=act_page, message="Table split across pages", details=details)
    )


def _check_continuation_headers(
    result: CheckResult,
    actual: fitz.Document,
    act_tables: list[tuple[int, dict]],
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """A table that runs onto another page MUST repeat its header row there.

    Checked against Staging alone, on every multi-page table, whether or not it
    pairs with a Production table - the rule is a property of the document being
    validated, not a difference between the two. A continuation page without the
    header leaves the reader with columns of values and nothing saying what they
    are, which is why this is an error rather than a note.

    The Production side is deliberately not consulted: if Production also drops
    the header, that is Production's defect and repeating it in Staging is still
    wrong.
    """
    for anchor_page, t in act_tables:
        for cont in t.get("continuations") or []:
            if cont["header_repeated"]:
                continue
            page = cont["page"]
            details = _with_heading(
                {
                    "actual_page": page + 1,
                    "reason": (
                        f"this table starts on page {anchor_page + 1} and continues onto page "
                        f"{page + 1}, but page {page + 1} does not reprint the header row"
                    ),
                    "header_row": t.get("header") or "(the table's first row could not be read)",
                    "first_row_on_continuation": cont["first_row"] or "(empty)",
                    "table_starts_on_page": anchor_page + 1,
                    "pages_spanned": t.get("fragments", 1),
                },
                heading,
            )
            # Staging only: there is no Production side to this finding, so
            # showing a Production screenshot beside it would only mislead.
            _attach_screenshots(
                details, actual, actual, output_dir, counter, None, None, page, cont["bbox"]
            )
            result.issues.append(
                Issue(
                    severity="error",
                    page=page,
                    message="Table header row not repeated on continuation page",
                    details=details,
                )
            )


def _compare_tables(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_page: int,
    act_page: int,
    exp_t: dict,
    act_t: dict,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    exp_cols, act_cols = _col_count(exp_t), _col_count(act_t)
    if exp_cols != act_cols:
        details = _with_heading({"expected_columns": exp_cols, "actual_columns": act_cols}, heading)
        _attach_screenshots(
            details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], act_page, act_t["bbox"]
        )
        result.issues.append(
            Issue(severity="error", page=exp_page, message="Table columns differ", details=details)
        )
    else:
        _check_column_layout(
            result, expected, actual, exp_page, act_page, exp_t, act_t, heading, output_dir, counter
        )

    _compare_rows(result, expected, actual, exp_page, act_page, exp_t, act_t, heading, output_dir, counter)


def _column_fractions(t: dict) -> list[float]:
    """Column edge positions as a fraction of the table's own width, so two
    tables of different absolute widths remain comparable.
    """
    x0, x1 = t["bbox"][0], t["bbox"][2]
    width = x1 - x0
    if width <= 0:
        return []
    edges = set()
    for row in t.get("cells") or []:
        for cell in row or []:
            if cell:
                edges.add(round((cell[0] - x0) / width, 4))
                edges.add(round((cell[2] - x0) / width, 4))
    return sorted(edges)


def _check_column_layout(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_page: int,
    act_page: int,
    exp_t: dict,
    act_t: dict,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """Same number of columns, but they sit in different places - a column was
    widened or narrowed, changing how the table reads.
    """
    exp_edges, act_edges = _column_fractions(exp_t), _column_fractions(act_t)
    if not exp_edges or not act_edges or len(exp_edges) != len(act_edges):
        return
    worst = max(abs(a - b) for a, b in zip(exp_edges, act_edges))
    if worst <= COLUMN_LAYOUT_TOLERANCE:
        return
    details = _with_heading(
        {
            "expected_page": exp_page + 1,
            "actual_page": act_page + 1,
            "largest_column_shift": f"{worst * 100:.1f}% of table width",
            "expected_column_edges": [f"{e:.0%}" for e in exp_edges],
            "actual_column_edges": [f"{e:.0%}" for e in act_edges],
        },
        heading,
    )
    _attach_screenshots(
        details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], act_page, act_t["bbox"]
    )
    result.issues.append(
        Issue(severity="warning", page=exp_page, message="Table column layout differs", details=details)
    )


def _norm_cell(cell) -> str:
    return normalize_block_text(cell or "").strip()


_CJK_TOKEN_RE = re.compile(r"[぀-㏿㐀-鿿가-힯豈-﫿]|[^\W_]{2,}", re.UNICODE)


def _fine_tokens(text: str) -> list[str]:
    """CJK-character-level tokens plus Latin words >= 2 chars, with printed
    page-number cross-references stripped. `_TOKEN_RE` keeps a whole Chinese
    phrase as ONE token, so when the two documents split or merge a numbered
    row differently ("6. 7. 右扬声器和通风口顶部可调式支脚" vs "右扬声器和通风口"
    + "顶部可调式支脚") every token mismatches and the row reads as lost.
    Comparing per CJK character instead survives that. Page numbers are dropped
    because the two documents paginate differently ("第 36 页" vs "第 34 页")."""
    text = _CELL_PAGE_REF_RE.sub("", _norm_cell(text))
    text = re.sub(r"第\s*\d{1,4}\s*页", "", text)  # CJK "page N" cross-reference
    return [t.lower() for t in _CJK_TOKEN_RE.findall(text)]


# How much of a row's / cell's own characters must be found in the Staging
# counterpart for the content to count as "there, just divided up differently".
_ROW_SURVIVES_RATIO = 0.75


def _row_present_in_table(exp_row: list, act_t: dict) -> bool:
    """True when (almost) every meaningful character/word of this Production
    row also appears somewhere in the Staging table - i.e. the row is there,
    the two documents just ruled it into different cells/rows. Used to hold
    back a "Table row missing" finding that is really an extraction artefact."""
    want = Counter(t for t in _fine_tokens(_row_text(exp_row)) if t not in _CELL_STOPWORDS)
    if not want:
        return True
    have = Counter(_fine_tokens(_table_text(act_t)))
    shared = sum((want & have).values())
    return shared / sum(want.values()) >= _ROW_SURVIVES_RATIO


# A printed cross-reference page number ("see X on page 43") legitimately
# differs between two documents that paginate differently - the user confirmed
# these as false positives - so "on page 43" is stripped from a cell before it
# is compared, the same way Content Validation strips it from prose.
_CELL_PAGE_REF_RE = re.compile(r"\s*\b(?:on|see|refer to)?\s*page\s*\d{1,4}\b", re.IGNORECASE)
# Words too generic for "this word is gone" to mean anything on its own.
_CELL_STOPWORDS = frozenset(
    "a an and or of to for in on at is are be the this that with by from as it "
    "see refer page will can may".split()
)


def _tokens(cell) -> list[str]:
    """A cell's comparison tokens, case-folded, with printed page-number
    cross-references stripped. Compared as a multiset, so the order the values
    are written in doesn't matter - only whether they are all still there.
    """
    text = _CELL_PAGE_REF_RE.sub("", _norm_cell(cell))
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _anchor_column(rows: list) -> int | None:
    """The single column that best identifies a row - the one whose values
    carry the most text (a setting name or a description), which is what makes
    it usable as a row key when rows have been inserted, removed or reordered.
    """
    if not rows:
        return None
    width = max(len(r or []) for r in rows)
    best_col, best_len = None, -1
    for c in range(width):
        total = sum(len(_norm_cell(_cell_at(r, c))) for r in rows)
        if total > best_len:
            best_col, best_len = c, total
    return best_col if best_len > 0 else None


def _cell_at(row, index: int):
    row = row or []
    return row[index] if 0 <= index < len(row) else None


def _row_key(row: list, anchor: int) -> str:
    """A row's identity for exact matching: its FIRST non-empty cell - the
    setting name ("Color Gamut", "Brightness", "Full"), which is what actually
    identifies a row.

    NOT the "most text" column: in a settings table every row's description
    column tends to repeat the same words ("...in Color Mode", "...the default
    setting"), so keying on it collapsed unrelated rows onto the same key and
    paired "Color Gamut" against "Gamma". The `anchor` argument is kept for the
    fuzzy fallback's benefit and only used when every cell is blank.
    """
    for cell in row or []:
        v = _norm_cell(cell).lower()
        if v:
            return v
    return _norm_cell(_cell_at(row, anchor)).lower()


def _compare_rows(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_page: int,
    act_page: int,
    exp_t: dict,
    act_t: dict,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """Find each Production row in Staging by its anchor cell, then compare only
    what that row actually lost.
    """
    exp_rows, act_rows = exp_t["rows"] or [], act_t["rows"] or []
    if not exp_rows or not act_rows:
        return
    width = max(len(r or []) for r in exp_rows)
    anchor = _anchor_column(exp_rows)
    if anchor is None:
        return

    # Match every Production row to a Staging row by its key. Exact key first;
    # then, for whatever is left, the closest remaining row by whole-row text
    # similarity - a row whose key cell was itself lightly edited still pairs
    # up instead of being reported as both missing AND (implicitly) changed.
    act_by_key: dict[str, list[int]] = {}
    for j, row in enumerate(act_rows):
        act_by_key.setdefault(_row_key(row, anchor), []).append(j)
    used: set[int] = set()
    pairs: dict[int, int] = {}
    for i, row in enumerate(exp_rows):
        for j in act_by_key.get(_row_key(row, anchor), []):
            if j not in used:
                pairs[i] = j
                used.add(j)
                break
    for i, row in enumerate(exp_rows):
        if i in pairs:
            continue
        j = _closest_row(row, act_rows, used)
        if j is not None:
            pairs[i] = j
            used.add(j)

    reported = 0
    for i, exp_row in enumerate(exp_rows):
        if reported >= MAX_CELL_ISSUES_PER_TABLE:
            return
        is_header = i == 0
        j = pairs.get(i)
        if j is None:
            if not _row_text(exp_row).strip():
                continue  # an all-blank spacer row is not "missing content"
            # Before calling the row lost: is its text actually still in the
            # Staging table, just ruled into different cells/rows? The two
            # documents routinely split or merge a numbered row differently
            # ("6. 7. …" one side, "6." + "7." the other), and pdfplumber
            # groups a wrapped cell differently between them - none of that is
            # a real content loss.
            if _row_present_in_table(exp_row, act_t):
                continue
            # A "header row missing" finding is only trustworthy when the
            # Production header row IS a clean header - a few short cells. When
            # its extracted text is long and instructional ("请进入 > 设置 >
            # 通用 …说明"), the header cell just got merged with the first data
            # row on the Production side; that is an extraction artefact, not a
            # dropped header. The real repeated-header rule is enforced
            # separately by `_check_continuation_headers`.
            if is_header and len(_row_text(exp_row)) > 24:
                continue
            details = _with_heading(
                {
                    "expected_page": exp_page + 1,
                    "actual_page": act_page + 1,
                    "row": i + 1,
                    "expected": [_row_text(exp_row)],
                    "reason": "this row's content is not present anywhere in the Staging table",
                },
                heading,
            )
            _attach_screenshots(
                details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], act_page, act_t["bbox"]
            )
            result.issues.append(
                Issue(
                    severity="error",
                    page=exp_page,
                    message="Table heading missing" if is_header else "Table row missing",
                    details=details,
                )
            )
            reported += 1
            continue

        act_row = act_rows[j]
        # A pair the fuzzy fallback matched but that shares almost no wording is
        # not really the same row - comparing its cells just manufactures
        # differences. Skip cell-level comparison for such a pair (the row is
        # already flagged, or genuinely absent, by the row-level check).
        if not _rows_are_comparable(exp_row, act_row):
            continue
        act_row_tokens = Counter(_fine_tokens(_row_text(act_row)))
        for c in range(width):
            exp_cell = _cell_at(exp_row, c)
            if not _norm_cell(exp_cell):
                continue
            missing = _missing_tokens(exp_cell, _cell_at(act_row, c))
            # One or two stray tokens after stopword/page-ref filtering is
            # extraction jitter, not a content loss worth reporting.
            if len(missing) < 2 and not any(len(t) >= 6 for t in missing):
                continue
            # The two documents don't always divide a row into the same cells
            # (a wrapped description, a merged "6. 7." number). If this cell's
            # own characters/words are (almost) all somewhere in the matched
            # Staging ROW, nothing was lost - only the cell boundary moved.
            want = Counter(t for t in _fine_tokens(exp_cell) if t not in _CELL_STOPWORDS)
            if want and sum((want & act_row_tokens).values()) / sum(want.values()) >= _ROW_SURVIVES_RATIO:
                continue
            details = _with_heading(
                {
                    "expected_page": exp_page + 1,
                    "actual_page": act_page + 1,
                    "row": i + 1,
                    "column": c + 1,
                    "matched_by": f"row key \"{_row_key(exp_row, anchor)[:60]}\"",
                    "missing_tokens": missing,
                    "expected": [_norm_cell(exp_cell)],
                    "actual": [_norm_cell(_cell_at(act_row, c))] if _norm_cell(_cell_at(act_row, c)) else [],
                },
                heading,
            )
            _attach_screenshots(
                details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], act_page, act_t["bbox"]
            )
            result.issues.append(
                Issue(
                    severity="error",
                    page=exp_page,
                    message="Table heading missing" if is_header else "Table cell missing",
                    details=details,
                )
            )
            reported += 1
            if reported >= MAX_CELL_ISSUES_PER_TABLE:
                return


def _row_tokens(row: list) -> Counter:
    return Counter(t for cell in (row or []) for t in _tokens(cell))


def _row_overlap(a: list, b: list) -> float:
    ta, tb = _row_tokens(a), _row_tokens(b)
    if not ta or not tb:
        return 0.0
    return sum((ta & tb).values()) / max(sum(ta.values()), sum(tb.values()))


def _rows_are_comparable(a: list, b: list) -> bool:
    return _row_overlap(a, b) >= 0.4


def _closest_row(exp_row: list, act_rows: list, used: set[int]) -> int | None:
    """The unused Staging row with the most word overlap with `exp_row` - used
    only after exact-key matching, and only accepted well above a floor so an
    unrelated leftover row isn't force-matched (which just manufactures cell
    differences)."""
    best_j, best = None, 0.0
    for j, act_row in enumerate(act_rows):
        if j in used:
            continue
        score = _row_overlap(exp_row, act_row)
        if score > best:
            best_j, best = j, score
    return best_j if best >= 0.6 else None

    _check_cell_layout(
        result, expected, actual, exp_page, act_page, exp_t, act_t, heading, output_dir, counter
    )


def _missing_tokens(exp_cell, act_cell) -> list[str]:
    """Tokens present in the Production cell that the Staging cell no longer
    has, compared as multisets so order and repetition are handled but
    rearrangement alone is not a finding.
    """
    exp_counts = Counter(_tokens(exp_cell))
    act_counts = Counter(_tokens(act_cell))
    missing = [
        t for t in (exp_counts - act_counts).elements()
        if t not in _CELL_STOPWORDS and (len(t) > 1 or t.isdigit())
    ]
    return sorted(missing)


def _row_spans(cells_row: list | None, width: int) -> list[tuple[int, int]]:
    """1-indexed inclusive column ranges covered by each occupied cell in a
    row - e.g. [(1, 1), (2, 3), (4, 4)] means column 1 alone, columns 2-3
    merged into one cell, then column 4 alone.

    pdfplumber represents a horizontal merge by giving the FIRST spanned
    column the cell's full rect and every later column in the span `None`, so
    a run of (rect, None, None, ...) reads as one cell covering that whole run.
    """
    row = list(cells_row or [])
    row += [None] * max(0, width - len(row))
    spans: list[tuple[int, int]] = []
    i, n = 0, len(row)
    while i < n:
        if row[i] is None:
            i += 1
            continue
        j = i + 1
        while j < n and row[j] is None:
            j += 1
        spans.append((i + 1, j))
        i = j
    return spans


def _format_span(span: tuple[int, int]) -> str:
    start, end = span
    return f"column {start}" if start == end else f"columns {start}-{end}"


def _best_anchor_column(rows: list) -> int | None:
    """The single column most likely to identify a row uniquely - the one
    carrying the most text across the given rows - used to pair up Production
    and Staging DATA rows (the header row is always row 0 on both sides once
    continuation fragments are stitched, so it needs no such pairing).
    """
    if not rows:
        return None
    width = max(len(r or []) for r in rows)
    best_col, best_len = None, -1
    for c in range(width):
        total = sum(len(_norm_cell(_cell_at(r, c))) for r in rows)
        if total > best_len:
            best_col, best_len = c, total
    return best_col if best_len > 0 else None


def _check_cell_layout(
    result: CheckResult,
    expected: fitz.Document,
    actual: fitz.Document,
    exp_page: int,
    act_page: int,
    exp_t: dict,
    act_t: dict,
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """Compares each row's CELL-MERGE STRUCTURE against its Production
    counterpart - exactly which columns are combined into one cell - not just
    how many cells the row has. A row that merges columns 2-3 in Production but
    keeps them separate in Staging (or the reverse) is a structural regression
    invisible from the cell text alone, since the words can read identically
    either way.

    Rows are paired by CONTENT - the same anchor-column technique
    `_compare_rows` uses for the row/cell text comparison - not by raw
    position, so a row inserted or deleted elsewhere in the table doesn't shift
    every later row's comparison out of alignment (this was the original
    version's bug: it zipped `exp_cells`/`act_cells` by index, so a single
    earlier mismatch cascaded into false merge findings on every row after
    it). The header row is always row 0 on both sides once continuation
    fragments are stitched - and a correctly repeated header on a continuation
    page is excluded from that row list during stitching (see
    `_stitch_continued_tables`), so it is never compared as if it were a data
    row here.
    """
    exp_rows, act_rows = exp_t["rows"] or [], act_t["rows"] or []
    exp_cells, act_cells = exp_t.get("cells") or [], act_t.get("cells") or []
    if not exp_rows or not act_rows or not exp_cells or not act_cells:
        return
    exp_width, act_width = _col_count(exp_t), _col_count(act_t)

    pairs: list[tuple[int, int]] = [(0, 0)]  # the header row, always first on both sides
    anchor = _best_anchor_column(exp_rows[1:])
    if anchor is not None:
        act_by_anchor: dict[str, int] = {}
        for idx, row in enumerate(act_rows[1:], start=1):
            key = _norm_cell(_cell_at(row, anchor)).lower()
            if key:
                act_by_anchor.setdefault(key, idx)
        for idx, row in enumerate(exp_rows[1:], start=1):
            key = _norm_cell(_cell_at(row, anchor)).lower()
            if key and key in act_by_anchor:
                pairs.append((idx, act_by_anchor[key]))

    reported = 0
    for exp_idx, act_idx in pairs:
        if reported >= MAX_CELL_ISSUES_PER_TABLE:
            break
        if exp_idx >= len(exp_cells) or act_idx >= len(act_cells):
            continue
        exp_spans = set(_row_spans(exp_cells[exp_idx], exp_width))
        act_spans = set(_row_spans(act_cells[act_idx], act_width))
        if exp_spans == act_spans:
            continue

        # A genuine merge is a span wider than one column; a difference that
        # only involves 1-column spans is just where an EMPTY column's rect
        # went undetected, not a real merge/split, so it's not worth reporting.
        lost = sorted(s for s in exp_spans - act_spans if s[1] > s[0])
        gained = sorted(s for s in act_spans - exp_spans if s[1] > s[0])
        if not lost and not gained:
            continue

        details = _with_heading(
            {
                "expected_page": exp_page + 1,
                "actual_page": act_page + 1,
                "row": exp_idx + 1,
                "expected_layout": ", ".join(_format_span(s) for s in sorted(exp_spans)),
                "actual_layout": ", ".join(_format_span(s) for s in sorted(act_spans)),
            },
            heading,
        )
        if lost:
            details["merge_missing_in_stage"] = [_format_span(s) for s in lost]
        if gained:
            details["merge_added_in_stage"] = [_format_span(s) for s in gained]
        _attach_screenshots(
            details, expected, actual, output_dir, counter, exp_page, exp_t["bbox"], act_page, act_t["bbox"]
        )
        result.issues.append(
            Issue(severity="error", page=exp_page, message="Table cell layout differs", details=details)
        )
        reported += 1


def _row_text(row: list | None) -> str:
    return " | ".join(_norm_cell(cell) for cell in (row or [])).strip(" |")


# A measurement noise/rounding-tolerance threshold, not a visual one: pdfplumber's
# cell rects occasionally overshoot a ruled line by a fraction of a point, and a
# table flush with the margin (by design) shouldn't fire on that alone. Anything
# past this is a real, visible overflow.
MARGIN_OVERFLOW_THRESHOLD = 3.0  # points


MARGIN_WIDE_BLOCK_FRACTION = 0.4  # a block must span at least this much of the page to count
MARGIN_BIN_SIZE = 5.0  # points - merges near-identical edges (kerning/hyphenation jitter) into one bin


def _document_margins(doc: fitz.Document) -> tuple[float | None, float | None]:
    """The document's established left/right content margin: where an ordinary
    full-width paragraph line is allowed to reach, established from the whole
    document.

    Two things this deliberately is NOT:

    - The raw page MediaBox (the physical paper edge). That was the original
      version of this check, and it is why the check essentially never fired: a
      table runs past the printed MARGIN, the actual defect a reader cares
      about, long before it reaches the page's literal physical bounds.
    - The single most common left/right edge among ALL text blocks on the
      page. Tried first, and wrong: a page number, a bullet glyph, or a short
      list-marker recurs on every page and is far more frequent than any one
      exact paragraph-wrap position, so the "most common edge" was a tiny
      element near the page's OTHER margin, not the body column's real edge.

    Instead: only blocks wide enough to plausibly be a full paragraph line (not
    a marker/page-number/icon) are considered, the WIDEST such block on each
    page is taken as that page's observed edge (a paragraph's last line is
    short, so this needs the widest, not just any, line to find where the
    column actually ends), and the edge that recurs most across pages - after
    rounding to absorb ordinary kerning/hyphenation jitter - is the answer.
    """
    left_per_page: list[float] = []
    right_per_page: list[float] = []
    for page in range(doc.page_count):
        try:
            blocks = doc[page].get_text("blocks")
        except Exception:
            continue
        page_width = doc[page].rect.width
        wide = [b for b in blocks if b[4].strip() and (b[2] - b[0]) > page_width * MARGIN_WIDE_BLOCK_FRACTION]
        if wide:
            left_per_page.append(min(b[0] for b in wide))
            right_per_page.append(max(b[2] for b in wide))

    def binned_mode(values: list[float]) -> float | None:
        if not values:
            return None
        bins: dict[int, list[float]] = {}
        for v in values:
            bins.setdefault(round(v / MARGIN_BIN_SIZE), []).append(v)
        best = max(bins.values(), key=len)
        return sum(best) / len(best)

    return binned_mode(left_per_page), binned_mode(right_per_page)


def _check_margins(
    result: CheckResult,
    actual: fitz.Document,
    tables: list[tuple[int, dict]],
    heading: str | None,
    output_dir: str | None,
    counter: "itertools.count",
    margins: tuple[float | None, float | None],
) -> None:
    """A table that runs past the page's established content margin - most
    concretely the RIGHT margin, which is the specific, common defect this
    check exists to catch: a table too wide for its column, spilling into or
    past the page's normal right-hand white space. Checked on STAGE only: it is
    a defect in the document being validated, not a difference between the two.
    """
    left_margin, right_margin = margins
    for page, t in tables:
        if page >= actual.page_count:
            continue
        page_rect = actual[page].rect
        x0, y0, x1, y1 = t["bbox"]
        right_bound = right_margin if right_margin is not None else page_rect.x1
        left_bound = left_margin if left_margin is not None else page_rect.x0

        right_overflow = x1 - right_bound
        left_overflow = left_bound - x0
        top_overflow = page_rect.y0 - y0
        bottom_overflow = y1 - page_rect.y1

        if max(right_overflow, left_overflow, top_overflow, bottom_overflow) <= MARGIN_OVERFLOW_THRESHOLD:
            continue

        parts = []
        details: dict = {}
        if right_overflow > MARGIN_OVERFLOW_THRESHOLD:
            parts.append(f"right margin by {right_overflow:.1f}pt")
            details["right_overflow_pt"] = round(right_overflow, 1)
        if left_overflow > MARGIN_OVERFLOW_THRESHOLD:
            parts.append(f"left margin by {left_overflow:.1f}pt")
        if top_overflow > MARGIN_OVERFLOW_THRESHOLD:
            parts.append(f"top of page by {top_overflow:.1f}pt")
        if bottom_overflow > MARGIN_OVERFLOW_THRESHOLD:
            parts.append(f"bottom of page by {bottom_overflow:.1f}pt")

        # Rendered as a string, not a dict: the report's detail formatter maps
        # a dict to its keys and would silently drop every measurement.
        details = _with_heading(
            {
                "bbox": t["bbox"],
                "page_rect": tuple(page_rect),
                "established_right_margin_pt": round(right_bound, 1),
                "table_right_edge_pt": round(x1, 1),
                "overflow": ", ".join(parts),
                **details,
            },
            heading,
        )
        # Only Staging is involved, so only Staging gets a screenshot.
        _attach_screenshots(details, actual, actual, output_dir, counter, None, None, page, t["bbox"])
        result.issues.append(
            Issue(severity="error", page=page, message="Table breaking the margins", details=details)
        )
