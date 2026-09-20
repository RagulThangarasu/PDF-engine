"""Table comparison read with Camelot, topic by topic.

Camelot's lattice reader rebuilds a table from its drawn rules, independently
of the pdfplumber extraction the rest of the chapter check uses - a second
reading of every ruled table, so a row one extractor loses the other still
sees. Every Production table is matched to the Staging table of the same topic
it shares the most words with, and compared row by row: a row either side
lacks, a row whose wording changed, a different number of columns, or a whole
table with no counterpart. Every finding's comment starts "Table issue:".
"""
from __future__ import annotations

import difflib
import logging
import re
import warnings
from collections import Counter

import fitz

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_TABLE_MATCH_MIN = 0.3   # share of words two tables must have in common to be the same table
_ROW_CHANGED_MIN = 0.5   # two rows this alike are one row reworded, not one lost and one added
_MIN_ROWS = 2            # a "table" of one row is a ruled box, not a table
_CACHE: dict[tuple, list[dict]] = {}


def available() -> bool:
    try:
        import camelot  # noqa: F401
        return True
    except Exception:
        return False


def reset_cache() -> None:
    _CACHE.clear()


# "on page 38" is a printed cross-reference Staging turns into a link with no
# page number - the same reference, not a changed row.
_PAGE_REF_RE = re.compile(r"\s*\b(?:on|see|refer to)?\s*page\s*\d{1,4}\b", re.IGNORECASE)


# A bracketed cross-reference: Production's "(See page 47)" and Staging's
# "(See "Optimizing image quality by Auto Cinema mode")" - the same reference,
# made a link naming its section - compare equal.
_XREF_RE = re.compile(r"\(\s*(?:see\s*)?(?:[\"“”'‘’][^\"“”'‘’]*[\"“”'‘’])?\s*\)", re.IGNORECASE)


_WRAP_HYPHEN_RE = re.compile(r"(\w)[-\u00ad]\s*\n\s*(?=[a-z])")  # "receiv-\ner": one word wrapped in its cell


def _norm(text: str) -> str:
    text = _WRAP_HYPHEN_RE.sub(r"\1", text or "")
    text = _XREF_RE.sub(" ", _PAGE_REF_RE.sub(" ", text))
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _join_wrapped(words: Counter, order: list[str], other: Counter, row: list[str]) -> None:
    """Two words of `order` that the cell prints as one word broken over a
    line at a hyphen ("Insta-" / "Show™", "Mac-" / "Books") become that one
    word when the other side prints it joined."""
    raw = "\n".join(c or "" for c in row).casefold()
    for x, y in zip(order, order[1:]):
        if other.get(x + y) and words.get(x) and words.get(y) \
                and re.search(re.escape(x) + r"-\s*\n\s*" + re.escape(y), raw):
            words[x] -= 1
            words[y] -= 1
            words[x + y] += 1
    for k in [k for k, v in words.items() if v <= 0]:
        del words[k]


def _row_key(row: list[str]) -> str:
    # Cells joined, empties dropped: one side splitting a column in two, or a
    # merged cell Camelot repeats or blanks, is the same row.
    return " ".join(n for n in (_norm(c) for c in row) if n)


def _shown(row: list[str]) -> str:
    text = " | ".join(" ".join(c.split()) for c in row if c and c.strip())
    return text if len(text) <= 160 else text[:159] + "…"


def extract(doc: fitz.Document, pdf_path: str) -> list[dict]:
    """Every ruled table in the document: [{page, bbox (top-left origin),
    rows: [[cell text]], row_boxes: [bbox]}], continuation pieces with the
    same header row stitched into one table."""
    key = (pdf_path, doc.page_count)
    if key in _CACHE:
        return _CACHE[key]
    tables: list[dict] = []
    try:
        import camelot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception as exc:  # no Camelot, unreadable file: the pdfplumber checks still run
        log.debug("camelot unavailable: %s", exc)
        found = []
    for t in found:
        page = int(t.page) - 1
        if not 0 <= page < doc.page_count:
            continue
        height = doc[page].rect.height
        rows = [[str(c or "") for c in r] for r in t.df.values.tolist()]
        row_boxes = []
        for r in t.cells:
            x0, x1 = min(c.x1 for c in r), max(c.x2 for c in r)
            top, bottom = max(c.y2 for c in r), min(c.y1 for c in r)
            row_boxes.append((x0, height - top, x1, height - bottom))
        grid = {
            "text": [list(r) for r in rows],
            "bottom": [[bool(c.bottom) for c in r] for r in t.cells],
            "top": [[bool(c.top) for c in r] for r in t.cells],
            "boxes": [[(c.x1, height - c.y2, c.x2, height - c.y1) for c in r] for r in t.cells],
        }
        keep = [i for i, r in enumerate(rows) if _row_key(r)]
        rows, row_boxes = [rows[i] for i in keep], [row_boxes[i] for i in keep]
        if len(rows) < _MIN_ROWS:
            continue
        x0, y0, x1, y1 = t._bbox
        tables.append({
            "page": page, "bbox": (x0, height - y1, x1, height - y0), "rows": rows,
            "row_pages": [page] * len(rows), "row_boxes": row_boxes, "cols": t.shape[1],
            "grids": [(page, grid)],
        })
    tables.sort(key=lambda t: (t["page"], t["bbox"][1]))
    _CACHE[key] = _stitch(tables, doc)
    return _CACHE[key]


def _stitch(tables: list[dict], doc: fitz.Document | None = None) -> list[dict]:
    """A table running onto the next page is one table - only when it reaches
    the bottom of its page and the next page opens with the table, not with a
    heading or paragraph first: Production's "PC" timing table ending p.77 and
    the "Video" table under its own heading on p.78 are two tables."""
    from pdfval.validators.table import FOOTER_BAND_RATIO, _starts_after_text

    def runs_on(prev: dict, t: dict) -> bool:
        if doc is None:
            return True
        last = prev["row_pages"][-1]
        page_h = doc[last].rect.height
        if max(b[3] for p, b in zip(prev["row_pages"], prev["row_boxes"]) if p == last) \
                < page_h * (1 - FOOTER_BAND_RATIO):
            return False
        return not _starts_after_text(doc, t["page"], t["bbox"][1])

    out: list[dict] = []
    for t in tables:
        prev = out[-1] if out else None
        # Same columns, same left/right edges, on the very next page: the
        # rest of the same table, whether or not it repeats its header there.
        if (prev is not None and t["page"] == prev["row_pages"][-1] + 1 and t["cols"] == prev["cols"]
                and abs(t["bbox"][0] - prev["bbox"][0]) <= 6 and abs(t["bbox"][2] - prev["bbox"][2]) <= 6
                and runs_on(prev, t)):
            skip = 1 if _row_key(t["rows"][0]) == _row_key(prev["rows"][0]) else 0
            prev["rows"] += t["rows"][skip:]
            prev["row_pages"] += t["row_pages"][skip:]
            prev["row_boxes"] += t["row_boxes"][skip:]
            prev["grids"] += t["grids"]
            continue
        out.append({**t, "rows": list(t["rows"]), "row_pages": list(t["row_pages"]),
                    "row_boxes": list(t["row_boxes"])})
    return out


def _words(t: dict) -> Counter:
    return Counter(w for r in t["rows"] for w in _row_key(r).split())


def _similarity(a: dict, b: dict) -> float:
    wa, wb = _words(a), _words(b)
    total = max(sum(wa.values()), sum(wb.values()), 1)
    return sum((wa & wb).values()) / total


def _in_span(t: dict, span: tuple) -> bool:
    start, end = span
    at = (t["page"], t["bbox"][1])
    return (start is None or at >= (start[0], start[1] - 1)) and (end is None or at < tuple(end))


def table_changes(exp_tables: list[dict], act_tables: list[dict], exp_span: tuple, act_span: tuple,
                  exp_at, act_at, element_cls, kind_table: str, covered_of, printed_of=None) -> list[dict]:
    """The findings for one chapter. `exp_at`/`act_at` map (page, y) to a topic;
    `covered_of(topic)` is the set of words other findings of that topic already
    name, so nothing is reported twice; `printed_of(side, topic)` is every word
    that side prints in the topic (a Counter), so a row the other document
    prints outside a ruled table - a header drawn without rules, a table set
    as text - is not called missing.

    Rows are compared per TOPIC, every table of the topic pooled: Staging
    splitting one table into two, or joining two, moves no row."""
    printed_of = printed_of or (lambda side, topic: None)
    exp_here = [t for t in exp_tables if _in_span(t, exp_span)]
    act_here = [t for t in act_tables if _in_span(t, act_span)]
    for t in exp_here:
        t["topic"] = exp_at(t["page"], t["bbox"][1])
    for t in act_here:
        t["topic"] = act_at(t["page"], t["bbox"][1])

    def rows_of(tables: list[dict], topic: str) -> list[tuple]:
        return [(t, k, _row_key(r)) for t in tables if t["topic"] == topic for k, r in enumerate(t["rows"])]

    def place(t: dict, idx: list[int], text: str):
        boxes = [(t["row_pages"][k], t["row_boxes"][k]) for k in idx] or [(t["page"], t["bbox"])]
        return element_cls(kind=kind_table, text=text, key=_norm(text), boxes=boxes[:12], section=t["topic"])

    def fresh(topic: str, text: str) -> bool:
        words = set(_norm(text).split())
        return bool(words) and not words <= covered_of(topic)

    def printed(side: str, topic: str, words: list[str]) -> bool:
        bag = printed_of(side, topic)
        return bag is not None and all(bag[w] >= n for w, n in Counter(words).items())

    # A row printed word for word in the other document's tables anywhere is
    # the same row, not a missing one: the documents only file it under a
    # differently titled section or chapter (a move, reported as such elsewhere).
    chapter_a = Counter(_row_key(r) for t in exp_tables for r in t["rows"])
    chapter_b = Counter(_row_key(r) for t in act_tables for r in t["rows"])
    out: list[dict] = []
    topics = list(dict.fromkeys([t["topic"] for t in exp_here + act_here]))
    for topic in topics:
        ra, rb = rows_of(exp_here, topic), rows_of(act_here, topic)
        ka, kb = [k for _, _, k in ra], [k for _, _, k in rb]
        gone = [i for i, k in enumerate(ka) if k not in kb and not chapter_b[k]]
        extra = [j for j, k in enumerate(kb) if k not in ka and not chapter_a[k]]
        # Rows reworded in place, paired one to one by likeness.
        for i in list(gone):
            best = max(extra, key=lambda j: difflib.SequenceMatcher(a=ka[i], b=kb[j]).ratio(), default=None)
            if best is None or difflib.SequenceMatcher(a=ka[i], b=kb[best]).ratio() < _ROW_CHANGED_MIN:
                continue
            gone.remove(i)
            extra.remove(best)
            wa, wb = Counter(ka[i].split()), Counter(kb[best].split())
            (ta0, xa0, _), (tb0, xb0, _) = ra[i], rb[best]
            _join_wrapped(wa, ka[i].split(), wb, ta0["rows"][xa0])
            _join_wrapped(wb, kb[best].split(), wa, tb0["rows"][xb0])
            lost = [w for w in (wa - wb).elements() if not printed("stage", topic, [w])]
            added = [w for w in (wb - wa).elements() if not printed("prod", topic, [w])]
            if not (lost or added) or not fresh(topic, " ".join(lost + added)):
                continue
            (ta, xa, _), (tb, xb, _) = ra[i], rb[best]
            parts = []
            spaced = (lost, added) if lost and "".join(lost) == "".join(added) else None
            if spaced:
                # A word broken over a line in its cell ("Mac-" / "Books") is
                # the wrap's hyphen, not a space the text has.
                def wrapped(words: list[str], row: list[str]) -> bool:
                    raw = "\n".join(c or "" for c in row).casefold()
                    return len(words) > 1 and bool(re.search(
                        r"-\s*\n\s*".join(re.escape(w) for w in words), raw))
                if wrapped(lost, ta["rows"][xa]) or wrapped(added, tb["rows"][xb]):
                    continue
                # The same characters, spaced differently: "x1080i" / "x 1080i".
                more = "extra" if len(added) > len(lost) else "missing"
                parts.append(f"space {more} in Staging: “{' '.join(lost)}” → “{' '.join(added)}”")
                lost, added = [], []
            if lost:
                parts.append("missing in Staging: " + ", ".join(f"“{w}”" for w in lost[:12]))
            if added:
                parts.append("extra in Staging: " + ", ".join(f"“{w}”" for w in added[:12]))
            out.append({
                "type": "table-cell", "kind": kind_table, "section": topic, "camelot": True,
                "gone": spaced[0] if spaced else lost, "extra": spaced[1] if spaced else added,
                "summary": f"Table issue: row changed — {' · '.join(parts)}.",
                "detail": f"Production: {_shown(ta['rows'][xa])}\nStaging: {_shown(tb['rows'][xb])}",
                "exp": place(ta, [xa], _shown(ta["rows"][xa])), "act": place(tb, [xb], _shown(tb["rows"][xb])),
            })
        for idx, rows, side, other in ((gone, ra, "missing", "stage"), (extra, rb, "added", "prod")):
            for i in idx:
                t, k, key = rows[i]
                if printed(other, topic, key.split()) or not fresh(topic, key):
                    continue
                shown = _shown(t["rows"][k])
                out.append({
                    "type": "table-row-missing" if side == "missing" else "table-row-added",
                    "kind": kind_table, "section": topic, "camelot": True,
                    "summary": (f"Table issue: row missing in Staging — “{shown}”." if side == "missing"
                                else f"Table issue: row extra in Staging — “{shown}”."),
                    "detail": shown,
                    "exp": place(t, [k], shown) if side == "missing" else None,
                    "act": place(t, [k], shown) if side == "added" else None,
                })
    return out


# --- merged cells ---------------------------------------------------------------

def _groups(grid: dict) -> list[list[int]]:
    """Row groups: runs of rows the first column merges into one cell (no rule
    under its cell)."""
    n = len(grid["text"])
    out, cur = [], [0] if n else []
    for r in range(1, n):
        if grid["bottom"][r - 1] and not grid["bottom"][r - 1][0]:
            cur.append(r)
        else:
            out.append(cur)
            cur = [r]
    if cur:
        out.append(cur)
    return [g for g in out if len(g) >= 2]


def _find(doc: fitz.Document, pages: list[int], text: str, near: tuple | None = None):
    """Where `text`'s first words are printed on these pages - the hit nearest
    `near` (page, y) when given."""
    words = " ".join((text or "").split()[:3])
    if not words:
        return None
    hits = []
    for p in pages:
        try:
            hits += [(p, r) for r in doc[p].search_for(words)]
        except Exception:
            continue
    if not hits:
        return None
    if near is None:
        return hits[0]
    same = [h for h in hits if h[0] == near[0] and h[1].y0 >= near[1] - 40]
    return min(same, key=lambda h: h[1].y0 - near[1]) if same else None


def merge_changes(act_tables: list[dict], act_span: tuple, act_at, expected: fitz.Document,
                  exp_pages: list[int], element_cls, kind_table: str) -> list[dict]:
    """A cell Production prints across all the lines of a merged row that
    Staging splits, leaving an empty cell under it. Staging's ruling is read
    from Camelot's grid; Production's from where the text sits - a value set
    across several lines is centred on the whole group, one set in the first
    line sits at its top - since Production often draws no row rules at all."""
    out: list[dict] = []
    for t in act_tables:
        if not _in_span(t, act_span):
            continue
        topic = act_at(t["page"], t["bbox"][1])
        for page, grid in t["grids"]:
            cols = len(grid["text"][0]) if grid["text"] else 0
            for g in _groups(grid):
                label = " ".join(grid["text"][g[0]][0].split())
                if not label:
                    continue
                for c in range(1, cols):
                    first = " ".join(grid["text"][g[0]][c].split())
                    below = [grid["text"][r][c].strip() for r in g[1:]]
                    split = grid["bottom"][g[0]][c]  # a rule under the first line's cell
                    if not first or any(below) or not split:
                        continue
                    # Production: the group's label, the value, and the other
                    # columns' first and last lines.
                    anchor = _find(expected, exp_pages, label)
                    if anchor is None:
                        continue
                    value = _find(expected, exp_pages, first, near=(anchor[0], anchor[1].y0))
                    others = [cc for cc in range(1, cols) if cc != c and grid["text"][g[-1]][cc].strip()]
                    if value is None or not others:
                        continue
                    last = _find(expected, exp_pages, grid["text"][g[-1]][others[0]], near=(anchor[0], anchor[1].y0))
                    top = _find(expected, exp_pages, grid["text"][g[0]][others[0]], near=(anchor[0], anchor[1].y0))
                    if last is None or top is None or last[1].y0 <= top[1].y0:
                        continue
                    line = max(1.0, value[1].height)
                    middle = (top[1].y0 + last[1].y1) / 2
                    if abs((value[1].y0 + value[1].y1) / 2 - middle) > 1.5 * line:
                        continue  # top-aligned in Production too: not a merged cell there
                    stage_boxes = [(page, grid["boxes"][r][c]) for r in g]
                    out.append({
                        "type": "table-merge", "kind": kind_table, "section": topic, "camelot": True,
                        "summary": (f"Table issue: merged cell split in Staging — “{first}” spans all "
                                    f"{len(g)} lines of the “{label}” row in Production (one merged cell); "
                                    f"Staging puts it in the first line only and leaves an empty cell below."),
                        "detail": label,
                        "exp": element_cls(kind=kind_table, text=first, key=_norm(first),
                                           boxes=[(value[0], tuple(value[1]))], section=topic),
                        "act": element_cls(kind=kind_table, text=first, key=_norm(first),
                                           boxes=stage_boxes, section=topic),
                    })
    return out
