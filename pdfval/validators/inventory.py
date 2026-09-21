"""The completeness sweep: every mirrored page pair compared as a whole.

Every other check in this engine answers a NAMED question - is a word missing,
is a row gone, is a figure resized. That is what makes them precise, and it is
also their limit: a difference nobody wrote a rule for is not found by any of
them, however plain it is on the page. This module is the other half. It takes
an inventory of each page - what is printed there, how many pictures, which
shaded panels and in what colour, which tables, which fonts, which links - and
compares the two inventories straight across. Anything present on one side of a
mirrored pair and absent from the other is a gap, whether or not a rule exists
for it.

It is deliberately blunt, so it is deliberately quiet about the things two
faithful re-exports legitimately disagree about: a picture a little larger, an
icon grouped differently by the drawing operators, a page that paginates
elsewhere, a colour a shade off. Those are matched with tolerance rather than
reported. What survives is a gap a reader would see.

Findings from here are shown for review, never failed on: the sweep knows that
something differs, not what it means.
"""
from __future__ import annotations

import fitz

# --- what counts as what -----------------------------------------------------
PICTURE_MIN_SIDE = 60.0      # pt: artwork smaller than this is an icon, counted not matched
SHADED_MIN_SIDE = 24.0       # pt: smaller filled shapes are rules and glyph decoration
NEAR_WHITE = 0.92            # a fill this pale is the page itself, not a shaded panel
TEXT_SAMPLE_CHARS = 700      # of the page's own text, enough to tell topics apart

# --- how much disagreement is still "the same thing" -------------------------
# Two re-exports of one manual resize artwork slightly and shift colours by a
# shade. Matched with no tolerance at all, every figure on every page reads as
# a gap - which is noise, not validation.
SIZE_TOLERANCE = 0.30        # a picture within this much of another IS that picture
COLOUR_TOLERANCE = 0.12      # 0-1 per channel: a panel this close is the same panel
ICON_COUNT_TOLERANCE = 3     # icons group differently per page; only a bigger gap counts
AREA_TOLERANCE = 0.25        # artwork covering this much less of the page is a picture gone


def _hex(colour) -> str:
    try:
        r, g, b = colour[:3]
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return "?"


def page_inventory(doc: fitz.Document, page_index: int) -> dict:
    """Everything measurable on one page, with no model and no rule involved."""
    page = doc[page_index]
    out: dict = {"page": page_index + 1}
    try:
        text = page.get_text() or ""
    except Exception:
        text = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out["lines"] = len(lines)
    out["words"] = len(text.split())
    out["text"] = " / ".join(lines)[:TEXT_SAMPLE_CHARS]   # the model's sample
    out["full_text"] = text                                # what the comparison uses
    # ARTWORK regions, not embedded image streams: one export draws a diagram
    # as vector lines where the other embeds the same diagram as a PNG, and
    # counting streams reads that as "eight pictures appeared".
    try:
        from pdfval.validators.chapter import _page_artwork_boxes

        regions = [(round(r.width), round(r.height), (round(r.x0, 1), round(r.y0, 1),
                                                       round(r.x1, 1), round(r.y1, 1)))
                   for r in _page_artwork_boxes(doc, page_index)]
    except Exception:
        try:
            regions = [(round(i["bbox"][2] - i["bbox"][0]), round(i["bbox"][3] - i["bbox"][1]),
                        tuple(round(float(v), 1) for v in i["bbox"]))
                       for i in page.get_image_info(xrefs=True) if i.get("bbox")]
        except Exception:
            regions = []
    out["images"] = sorted(r[:2] for r in regions if min(r[:2]) >= PICTURE_MIN_SIDE)
    out["image_boxes"] = [r for r in regions if min(r[:2]) >= PICTURE_MIN_SIDE]
    out["icons"] = sum(1 for r in regions if min(r[:2]) < PICTURE_MIN_SIDE)
    out["icon_boxes"] = [r for r in regions if min(r[:2]) < PICTURE_MIN_SIDE]
    panels = []
    try:
        for d in page.get_drawings():
            fill, rect = d.get("fill"), d.get("rect")
            if not fill or rect is None:
                continue
            if rect.width < SHADED_MIN_SIDE or rect.height < SHADED_MIN_SIDE:
                continue
            if min(fill[:3]) >= NEAR_WHITE:
                continue
            panels.append((tuple(round(float(v), 3) for v in fill[:3]),
                           (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))))
    except Exception:
        pass
    # A filled shape INSIDE a picture is part of that picture - the dark plate
    # behind a diagram, the coloured disc of a drawn icon. Production draws its
    # artwork as vectors and Staging bakes the same artwork into images, so
    # counted as page panels those shapes read as "Production has 12 panels
    # Staging lacks" on every page with a diagram on it. The picture itself is
    # compared as a picture.
    art = [r[2] for r in regions]
    panels = [(c, bb) for c, bb in panels if not _inside_any(bb, art)]
    out["panel_boxes"] = panels
    out["panels"] = sorted(c for c, _ in panels)
    try:
        out["tables"] = sorted((t.row_count, t.col_count) for t in page.find_tables().tables)
    except Exception:
        out["tables"] = []
    try:
        out["links"] = sorted((l.get("uri") or "internal")[:60]
                              for l in page.get_links() if l.get("kind") != 0)
    except Exception:
        out["links"] = []
    try:
        out["fonts"] = sorted({(s.get("font") or "").split("+")[-1].split("-")[0]
                               for b in page.get_text("dict").get("blocks", [])
                               for l in b.get("lines", []) for s in l.get("spans", [])
                               if s.get("font")})
    except Exception:
        out["fonts"] = []
    return out


def _inside_any(box, regions, slack: float = 2.0) -> bool:
    """`box` sits within one of `regions` (both (x0, y0, x1, y1))."""
    for r in regions:
        if (box[0] >= r[0] - slack and box[1] >= r[1] - slack
                and box[2] <= r[2] + slack and box[3] <= r[3] + slack):
            return True
    return False


def _unmatched(mine: list, theirs: list, same) -> list:
    """Items of `mine` with no partner in `theirs`, each partner used once."""
    spare = list(theirs)
    out = []
    for item in mine:
        for k, other in enumerate(spare):
            if same(item, other):
                spare.pop(k)
                break
        else:
            out.append(item)
    return out


def _same_size(a, b) -> bool:
    return all(abs(x - y) <= SIZE_TOLERANCE * max(x, y, 1) for x, y in zip(a, b))


def _same_colour(a, b) -> bool:
    return all(abs(x - y) <= COLOUR_TOLERANCE for x, y in zip(a, b))


def compare_pages(exp_inv: dict, act_inv: dict) -> list[str]:
    """Plain sentences for every gap between two mirrored pages, or []."""
    return [g["text"] for g in compare_pages_boxed(exp_inv, act_inv)]


def compare_pages_boxed(exp_inv: dict, act_inv: dict) -> list[dict]:
    """Every gap, each with WHERE it is: {"text", "side", "bbox"}.

    A finding without a place on the page is a sentence a reader has to go
    hunting for. Everything measured here was measured at a position, so the
    position travels with the gap and the viewer can box it - which is the
    whole difference between telling someone a panel is missing and showing
    them where it should have been.
    """
    gaps: list[dict] = []

    def add(text, side, bbox=None):
        gaps.append({"text": text, "side": side, "bbox": list(bbox) if bbox else None})

    # Artwork by area, not region by region - the detector groups a diagram
    # into one region on one page and into its parts on another purely by how
    # the drawing operators are batched, and matching region against region
    # called a picture "missing" on pages whose artwork is identical.
    exp_area = sum(w * h for w, h in exp_inv["images"])
    act_area = sum(w * h for w, h in act_inv["images"])
    if max(exp_area, act_area) and abs(exp_area - act_area) > AREA_TOLERANCE * max(exp_area, act_area):
        lost = exp_area > act_area
        side, boxes = ("prod", exp_inv.get("image_boxes")) if lost else ("stage", act_inv.get("image_boxes"))
        biggest = max(boxes, key=lambda r: r[0] * r[1])[2] if boxes else None
        if lost:
            add(f"{_n('picture', len(exp_inv['images']))} on this Production page cover "
                f"{exp_area:,}pt\u00b2 against {act_area:,}pt\u00b2 in Staging - artwork is missing here",
                side, biggest)
        else:
            add(f"artwork on this Staging page covers {act_area:,}pt\u00b2 against "
                f"{exp_area:,}pt\u00b2 in Production - there is a picture here Production does not have",
                side, biggest)
    icon_gap = exp_inv["icons"] - act_inv["icons"]
    if abs(icon_gap) > ICON_COUNT_TOLERANCE:
        side = "prod" if icon_gap > 0 else "stage"
        boxes = (exp_inv if icon_gap > 0 else act_inv).get("icon_boxes") or []
        add(f"{abs(icon_gap)} more icons on this page in "
            f"{'Production' if icon_gap > 0 else 'Staging'} "
            f"({exp_inv['icons']} in Production, {act_inv['icons']} in Staging)",
            side, _span([r[2] for r in boxes]))
    # Shaded panels, grouped by colour with a count and boxed on the panels
    # themselves: three note panels of one blue is one thing a reader sees.
    for colour, boxes in _grouped_panels(exp_inv, act_inv, lost=True):
        add(f"{_n('shaded panel', len(boxes))} in {_hex(colour)} printed in Production, not in Staging",
            "prod", _span(boxes))
    for colour, boxes in _grouped_panels(act_inv, exp_inv, lost=False):
        add(f"Staging prints {_n('shaded panel', len(boxes))} in {_hex(colour)} that Production does not",
            "stage", _span(boxes))
    # Hyperlinks. Measured since the first version of this sweep and never
    # compared - so a link on one page and not the other was invisible here,
    # which is exactly the kind of "nobody wrote a rule for it" gap the sweep
    # exists to catch. Internal links are compared by count (their targets are
    # page objects, which differ by design between two paginations); a real
    # URL is compared as itself.
    exp_uri = sorted(u for u in exp_inv["links"] if u != "internal")
    act_uri = sorted(u for u in act_inv["links"] if u != "internal")
    for uri in sorted(set(exp_uri) - set(act_uri)):
        add(f"the link to {uri} is on this Production page and not on the Staging one", "prod")
    for uri in sorted(set(act_uri) - set(exp_uri)):
        add(f"Staging links to {uri} here, Production does not", "stage")
    # Internal links are NOT counted against each other. They point at page
    # objects, and one document's contents page carrying 33 of them against
    # the other's none is how the two were built, not a gap - measured, it was
    # 20 findings of pure noise. A real URL is a claim about the world and is
    # compared above; where an internal link goes is the link checker's own.
    # A table's SHAPE is the row-by-row table check's business, in far more
    # detail. The sweep asks only what that check cannot: is there a table on
    # one of these pages and none at all on the other.
    if exp_inv["tables"] and not act_inv["tables"]:
        add(f"{_n('table', len(exp_inv['tables']))} on this Production page, none on the Staging one", "prod")
    elif act_inv["tables"] and not exp_inv["tables"]:
        add(f"{_n('table', len(act_inv['tables']))} on this Staging page, none on the Production one", "stage")
    # Words. The sweep measured how much text is on each page from the start
    # and never compared it, leaving every wording difference to the rules -
    # so anything the rules do not name went unseen here too. Compared as word
    # multisets, which wrapping and reflow cannot change, and only against
    # words that appear NOWHERE on the other side's page or its neighbours:
    # a page boundary that falls in a different place is not a missing word.
    for text, side in ((_words_only_here(exp_inv, act_inv), "prod"),
                       (_words_only_here(act_inv, exp_inv), "stage")):
        if text:
            where = "Production" if side == "prod" else "Staging"
            other = "Staging" if side == "prod" else "Production"
            add(f"{where} prints words on this page that are nowhere on the {other} one: {text}", side)
    # Fonts are compared once for the whole document - see `font_changes`.
    return gaps


_WORD_RE = __import__("re").compile(r"[\w\u00c0-\uffff]+", __import__("re").UNICODE)
WORDS_SHOWN = 12          # of the words only one side prints, before "and N more"
_WORD_MIN_LEN = 3         # shorter tokens are markers and noise, not content


def _page_words(inv: dict) -> "object":
    from collections import Counter

    return Counter(w.casefold() for w in _WORD_RE.findall(inv.get("full_text") or inv.get("text") or "")
                   if len(w) >= _WORD_MIN_LEN)


def _words_only_here(mine: dict, theirs: dict) -> str:
    """Words this page prints more often than the other page and its
    neighbours do - quoted, or "" when there are none."""
    mine_words = _page_words(mine)
    theirs_words = _page_words(theirs)
    for neighbour in theirs.get("neighbours") or ():
        theirs_words.update(neighbour)
    short = [w for w, n in mine_words.items() if n > theirs_words.get(w, 0)]
    if not short:
        return ""
    shown = ", ".join(f"\u201c{w}\u201d" for w in short[:WORDS_SHOWN])
    return shown + (f" and {len(short) - WORDS_SHOWN} more" if len(short) > WORDS_SHOWN else "")


def _span(boxes: list) -> tuple | None:
    """One box round them all, so a group of panels is marked in one place."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _grouped_panels(mine: dict, theirs: dict, lost: bool) -> list[tuple]:
    """[(colour, [bbox, ...])] for panels `mine` has that `theirs` does not."""
    spare = list(theirs.get("panel_boxes") or [])
    unmatched: list[tuple] = []
    for colour, bbox in mine.get("panel_boxes") or []:
        for k, (other, _) in enumerate(spare):
            if _same_colour(colour, other):
                spare.pop(k)
                break
        else:
            unmatched.append((colour, bbox))
    grouped: dict = {}
    for colour, bbox in unmatched:
        for known in grouped:
            if _same_colour(colour, known):
                grouped[known].append(bbox)
                break
        else:
            grouped[colour] = [bbox]
    return sorted(grouped.items())


def _by_colour(colours: list) -> list[tuple]:
    grouped: dict = {}
    for c in colours:
        for known in grouped:
            if _same_colour(c, known):
                grouped[known] += 1
                break
        else:
            grouped[c] = 1
    return sorted(grouped.items())


def _n(thing: str, count: int) -> str:
    return f"a {thing}" if count == 1 else f"{count} {thing}s"


def document_fonts(doc: fitz.Document) -> set:
    """Every typeface the document actually prints with."""
    out: set = set()
    for i in range(doc.page_count):
        try:
            for b in doc[i].get_text("dict").get("blocks", []):
                for l in b.get("lines", []):
                    for sp in l.get("spans", []):
                        name = (sp.get("font") or "").split("+")[-1]
                        if name:
                            out.add(name)
        except Exception:
            continue
    return out


def font_changes(expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """A typeface one document prints with and the other never does.

    Said once for the whole document. This is how a rebuild's font
    substitution shows itself - and a substituted CJK face is exactly what
    draws the wrong glyphs for a menu entry nobody can read.
    """
    exp_fonts, act_fonts = document_fonts(expected), document_fonts(actual)
    out: list[dict] = []
    for lost, mine, theirs in ((True, exp_fonts, act_fonts), (False, act_fonts, exp_fonts)):
        gone = sorted(mine - theirs)
        if not gone:
            continue
        side, other = ("Production", "Staging") if lost else ("Staging", "Production")
        shown = ", ".join(f"“{f}”" for f in gone[:8])
        out.append({
            "type": "font-set",
            "title": f"Typefaces used in {side} but not in {other}",
            "description": (f"{side} prints with {len(gone)} typeface(s) that {other} never uses: "
                            f"{shown}{' and more' if len(gone) > 8 else ''}. A substituted face can "
                            f"change how text is drawn even where the words are identical."),
        })
    return out


def _neighbour_words(doc: fitz.Document, page_index: int) -> list:
    """Word counts for the pages either side of `page_index`."""
    out = []
    for i in (page_index - 1, page_index + 1):
        if 0 <= i < doc.page_count:
            try:
                out.append(_page_words({"full_text": doc[i].get_text() or ""}))
            except Exception:
                continue
    return out


# Recall first on this path: the sweep is the AI review's eyes, and a gap
# dropped here is a difference nobody ever sees. Listed generously, with a
# count of anything past the limit rather than silence.
MAX_GAPS_PER_PAGE = 12


# Which kind of gap each already-reported category speaks for. A sweep gap on
# a page a named rule has already spoken about is the same difference said
# twice, in vaguer words - and saying it twice is what makes a report unusable.
COVERED_BY = {
    "content": ("words", "table"),
    "tables": ("table", "words"),
    "images": ("picture", "icons"),
    "links": ("link",),
    "formatting": ("panel",),
    "lists": ("words",),
    "bold": ("words",),
}


def _kind_of(gap_text: str) -> str:
    if "prints words" in gap_text:
        return "words"
    if "link" in gap_text:
        return "link"
    if "panel" in gap_text:
        return "panel"
    if "artwork" in gap_text or "pt\u00b2" in gap_text:
        return "picture"
    if "icon" in gap_text:
        return "icons"
    return "table"


# Which gap kinds are solid enough to ASSERT in the report on their own.
# Words, links and tables are read from the text and object layers and mean
# what they say. Pictures, icons and panels are read from the drawing layer,
# where Production draws a diagram as vectors and Staging bakes the same
# diagram into an image - the two are the same picture and the detectors
# disagree about how to group it, so on those the sweep can only raise a
# question, not make a claim. They still go to the AI review, which is a place
# for questions; they do not go in the report as findings.
REPORTABLE_KINDS = ("words", "link", "table")


def inventory_changes(expected: fitz.Document, actual: fitz.Document,
                      pairs: list[tuple[int, int]],
                      covered: dict | None = None,
                      kinds: tuple | None = None) -> list[dict]:
    """One finding per mirrored page pair that differs, listing every gap.

    `covered` is `{(production page, kind): True}` for what the named rules
    already report. The sweep is the net under those rules, not a second voice
    repeating them: a page whose pictures are already reported keeps its
    picture gaps to itself and speaks only about what nothing else covers.
    """
    covered = covered or {}
    out: list[dict] = []
    for prod, stage in pairs:
        if not (1 <= prod <= expected.page_count and 1 <= stage <= actual.page_count):
            continue
        try:
            a, b = page_inventory(expected, prod - 1), page_inventory(actual, stage - 1)
            # Each side's page gets its NEIGHBOURS' words as context. The two
            # documents paginate differently, so the tail of a Production page
            # is routinely the head of the next Staging one - compared page
            # against page alone, every page break reads as a paragraph lost
            # and a paragraph gained. A word is only missing when it is on
            # neither the facing page nor the ones either side of it.
            a["neighbours"] = _neighbour_words(actual, stage - 1)
            b["neighbours"] = _neighbour_words(expected, prod - 1)
        except Exception:
            continue
        gaps = [g for g in compare_pages(a, b)
                if not covered.get((prod, _kind_of(g)))
                and (kinds is None or _kind_of(g) in kinds)]
        if not gaps:
            continue
        shown = gaps[:MAX_GAPS_PER_PAGE]
        more = len(gaps) - len(shown)
        out.append({
            "type": "page-sweep",
            "prod_page": prod,
            "stage_page": stage,
            "title": f"Page differences not covered by another check",
            "description": ("Production p.%d against Staging p.%d: " % (prod, stage))
                           + "; ".join(shown)
                           + (f"; and {more} more" if more > 0 else "") + ".",
            "gaps": gaps,
        })
    return out


# --- how the two documents SET their headings --------------------------------
#
# A heading whose typeface and size both changed is not "bold missing" and not
# a wording change, so no named rule reports it - `size` is in the engine's
# never-reported set precisely because body-text size drifts between exports
# and reporting it per element buries everything else. But a manual whose
# headings go from Poppins 16pt to Roboto 13.5pt looks different on every page
# a reader opens, and calling that unreportable because the per-element rule
# would be noisy is the rule serving itself. Compared across the whole document
# and said ONCE, it is one clear finding instead of two hundred.
_HEADING_SAMPLE = 40          # headings to measure - enough to see a systematic change
_HEADING_MAJORITY = 0.6       # this share must agree before it is "how the document is set"
_HEADING_SIZE_DRIFT = 0.75    # pt: smaller than this is export drift, not a decision


def _printed_style(doc: fitz.Document, page_index: int, title: str):
    """(font family, size) the heading is actually printed in, or None."""
    want = " ".join((title or "").split()).casefold()[:40]
    if not want or not (0 <= page_index < doc.page_count):
        return None
    try:
        blocks = doc[page_index].get_text("dict").get("blocks", [])
    except Exception:
        return None
    for b in blocks:
        for l in b.get("lines", []):
            spans = l.get("spans", [])
            text = " ".join("".join(s.get("text", "") for s in spans).split()).casefold()
            if text.startswith(want[:20]) and spans:
                font = (spans[0].get("font") or "").split("+")[-1]
                return font, round(float(spans[0].get("size") or 0), 1)
    return None


def heading_style_changes(expected: fitz.Document, actual: fitz.Document) -> list[dict]:
    """One finding when Staging sets its headings in a different face or size."""
    try:
        from pdfval.validators.headings import resolve_entries
        from pdfval.validators.toc import match_toc_entries

        exp_entries, act_entries = resolve_entries(expected, actual)
    except Exception:
        return []
    pairs = []
    for m in match_toc_entries(exp_entries, act_entries):
        if m.expected_index is None or m.actual_index is None:
            continue
        e, a = exp_entries[m.expected_index], act_entries[m.actual_index]
        ps, qs = _printed_style(expected, e.page, e.title), _printed_style(actual, a.page, a.title)
        if ps and qs:
            pairs.append((ps, qs))
        if len(pairs) >= _HEADING_SAMPLE:
            break
    if len(pairs) < 3:
        return []
    face = [(p[0], q[0]) for p, q in pairs if p[0] != q[0]]
    size = [(p[1], q[1]) for p, q in pairs if abs(p[1] - q[1]) > _HEADING_SIZE_DRIFT]
    changed = [k for k in (face, size) if len(k) >= _HEADING_MAJORITY * len(pairs)]
    if not changed:
        return []
    from collections import Counter

    said = []
    if len(face) >= _HEADING_MAJORITY * len(pairs):
        (pf, af), _ = Counter(face).most_common(1)[0]
        said.append(f"the typeface from “{pf}” to “{af}”")
    if len(size) >= _HEADING_MAJORITY * len(pairs):
        (ps_, as_), _ = Counter(size).most_common(1)[0]
        said.append(f"the size from {ps_}pt to {as_}pt")
    n = max(len(face), len(size))
    return [{
        "type": "heading-style",
        "title": "Headings are set differently in Staging",
        "description": (f"Staging changes {' and '.join(said)} on {n} of the {len(pairs)} headings "
                        f"checked. The wording is unchanged, so no content check sees this - but a "
                        f"heading set smaller or in another face reads as a different level of "
                        f"heading, or as no longer standing out from the text under it."),
    }]


# --- the gap between a list marker and the words it introduces ---------------
#
# "1." then a space then "SOURCE" is a decision about layout that no wording,
# bold or list-marker rule looks at: the marker is present on both sides, the
# words are identical, only the space between them changed. A reader sees it
# immediately - the item hangs away from its number, or sits jammed against it
# - and it is the commonest thing found in a manual rebuilt by hand. Measured
# against the SAME item on the other side, and against the item's own siblings,
# because a single item set differently from the rest of its list is the shape
# this takes in practice.
_MARKER_RE = __import__("re").compile(r"^(\d{1,2}[.)]?|[•●▪◦‣])$")
MARKER_GAP_TOLERANCE = 6.0    # pt: a difference smaller than this is typesetting drift
MARKER_GAP_MAX_REACH = 120.0  # pt: text further right than this is another column
_MARKER_GAP_MAX_ROWS = 40     # markers to measure on one page
_MARKER_LEFT_SHARE = 0.35     # a list marker starts its line, not mid-column


def marker_gaps(doc: fitz.Document, page_index: int) -> dict:
    """`{marker: (gap in pt, bbox of the gap)}` for each list item on the page.

    The gap is measured from the right edge of the marker to the left edge of
    the first text that sits on its line - which is what the eye reads as the
    space after the bullet, whether the marker is drawn inline with the item
    or as its own text object beside it.
    """
    try:
        blocks = doc[page_index].get_text("dict").get("blocks", [])
    except Exception:
        return {}
    lines = [(l["bbox"], "".join(s.get("text", "") for s in l.get("spans", [])), l.get("spans", []))
             for b in blocks for l in b.get("lines", [])]
    try:
        page_width = doc[page_index].rect.width or 595.0
    except Exception:
        page_width = 595.0
    try:
        from pdfval.validators.chapter import _inside_table
    except Exception:
        _inside_table = None
    out: dict = {}
    for bbox, text, spans in lines:
        stripped = text.strip()
        inline = None
        if _MARKER_RE.match(stripped):          # the marker is its own text object
            y = (bbox[1] + bbox[3]) / 2
            right = [lb for lb, t, _ in lines
                     if lb[0] > bbox[2] - 1 and lb[1] <= y <= lb[3] and t.strip()]
            if not right:
                continue
            nearest = min(right, key=lambda lb: lb[0])
            gap, left, top, bottom = nearest[0] - bbox[2], bbox[2], bbox[1], bbox[3]
        elif len(spans) > 1 and _MARKER_RE.match(spans[0].get("text", "").strip()):
            rest = next((s for s in spans[1:] if s.get("text", "").strip()), None)
            if rest is None:
                continue
            inline = True
            gap, left, top, bottom = (rest["bbox"][0] - spans[0]["bbox"][2],
                                      spans[0]["bbox"][2], bbox[1], bbox[3])
        else:
            continue
        if not (0 <= gap <= MARKER_GAP_MAX_REACH):
            continue
        # A bare number in a TABLE cell is not a list marker, and the next
        # cell along is not "the words it introduces" - measured as one, every
        # timing table in an appendix reported its columns as spacing changes.
        # A real marker also starts its line, near the text column's left edge.
        if bbox[0] > _MARKER_LEFT_SHARE * page_width and not _MARKER_RE.match(stripped[:1]):
            continue
        if _inside_table is not None:
            try:
                if _inside_table(doc, page_index, bbox):
                    continue
            except Exception:
                pass
        key = stripped if inline is None else spans[0]["text"].strip()
        out.setdefault(key, (round(gap, 1), (round(left, 1), round(top, 1),
                                             round(left + max(gap, 2), 1), round(bottom, 1))))
        if len(out) >= _MARKER_GAP_MAX_ROWS:
            break
    return out


def marker_spacing_changes(expected: fitz.Document, actual: fitz.Document,
                           pairs: list[tuple[int, int]]) -> list[dict]:
    """A list item whose space after its marker changed between the two."""
    out: list[dict] = []
    for prod, stage in pairs:
        if not (1 <= prod <= expected.page_count and 1 <= stage <= actual.page_count):
            continue
        mine, theirs = marker_gaps(expected, prod - 1), marker_gaps(actual, stage - 1)
        for marker in sorted(set(mine) & set(theirs)):
            (pg, _pb), (sg, sb) = mine[marker], theirs[marker]
            if abs(sg - pg) < MARKER_GAP_TOLERANCE:
                continue
            wider = sg > pg
            out.append({
                "type": "marker-spacing",
                "prod_page": prod, "stage_page": stage,
                "title": f"Space after “{marker}” changed in Staging",
                "description": (f"The item “{marker}” on Staging p.{stage} sits {sg}pt after its "
                                f"marker against {pg}pt on Production p.{prod} - "
                                f"{'a wider' if wider else 'a tighter'} gap than the baseline, on the "
                                f"same item with the same words."),
                "stage_boxes": [{"page": stage, "bbox": list(sb)}],
                "prod_boxes": [{"page": prod, "bbox": list(mine[marker][1])}],
            })
    return out
