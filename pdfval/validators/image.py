"""3. Image validation.

Every figure Production shows in a section is looked for in STAGE'S SAME
SECTION and compared against what is found there, by appearance. Findings:

* Image missing              - a Production figure has no counterpart at all
* Image content differs      - a figure is there, but it is not the same picture
* Image blurred              - the same picture, rendered soft/low-resolution
                               in STAGE where Production is sharp
(A figure rendered at a different SIZE is counted in the summary table but is
 not a finding: it is a layout choice, not a defect.)
* Image outside its section  - the picture is in STAGE but past the section's end
* Broken image               - STAGE artwork that fails to decode or renders blank
* Image label missing        - a figure caption/label absent from STAGE
* Diagram callout number missing - a leader-line callout number absent from STAGE
* Image alignment changed    - left/center/right only, small shifts ignored
* Image highlight box missing - a coloured highlight box drawn on a figure

WHICH PAGE a figure lands on inside its section is deliberately not a finding.
The two documents paginate differently, so the same figure routinely sits a
page later in STAGE; what matters is that it is still in the section, which is
what the matching below asks. The page it was found on is recorded in the
finding's details and in the per-section summary either way, and only a figure
that has left its section entirely is reported (as a warning, not an error).

Matching is by RENDERED APPEARANCE (see `pdfval.imagefp`), not by reading-order
position: position-based pairing goes off by one for every figure after a lost
one, and reported figures as changed that had never moved. Two figures are only
called the same picture on a strong two-signal agreement, and small positional
drift is still never reported - a figure shifts a few points whenever the text
above it reflows.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import fitz

from pdfval import imagefp, ocr
from pdfval.extractor import ImageInfo, get_all_detected_regions, get_figures
from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import heading_at, in_section_bounds, matched_heading_ranges

MAX_SCREENSHOTS = 300  # cap total screenshot pairs so a huge document doesn't stall the report

# Small UI/decorative icons (menu glyphs, checkboxes, etc.) are numerous per
# page and drift independently of any real content change. Real figure content
# in these manuals is comfortably larger than this ON THE PAGE. Measured in
# points of rendered size, NOT source-raster pixels: a 512x512px icon drawn at
# 8pt is still an icon, and a 60x60px logo stretched across half the page is
# still a figure.
MIN_FIGURE_SIDE = 60.0  # points
MIN_FIGURE_AREA = 6000.0  # points^2

# --- how sure the appearance comparison has to be before it says anything ----
#
# Calibrated against a real Production/Staging manual pair. Two renderings of
# the SAME illustration score 0.93-1.00; two genuinely different figures under
# one heading sit around 0.32, and a pair whose on-page shapes disagree by more
# than a few percent never got past 0.86 even when both showed a monitor. So
# 0.90 is comfortably inside the empty gap between "the same picture" and
# "a different picture", not a value tuned to the edge of one.
SAME_IMAGE_SCORE = 0.90
# The sliding second opinion (`imagefp.region_match`) answers a different, more
# forgiving question - "is this picture in there at all" - so it is held to its
# own bar. It is only consulted when the cheap comparison was unsure, and only
# ever promotes a pair to "the same picture"; it never condemns one.
SLIDE_SAME_SCORE = 0.84
SLIDE_MIN_INTEREST = 0.20  # below this the two share nothing; don't pay for the search
# A picture that still matches this well but has been reshaped is the same
# artwork resized, which is a different (and milder) finding than a swap.
RESIZED_IMAGE_SCORE = 0.78
# A like-for-like pair at or above this is the SAME picture that just
# re-rendered a little (two exports of one vector illustration land ~0.85-1.00,
# not 0.93-1.00 as first assumed - a redrawn hand or a slightly different line
# weight is enough to knock a few points off). Below it, with comparable
# extents, the picture has genuinely changed.
NEAR_IDENTICAL_SCORE = 0.85
ASPECT_TOLERANCE = 0.08  # fractional difference in width/height before "reshaped"
# Two crops of the same artwork can only be compared pixel-for-pixel when they
# cover comparable extents. When they don't, a low score says the two DETECTIONS
# disagree, not that the picture changed - so no defect is asserted, and the
# pair is put in front of a human instead.
COMPARABLE_AREA_RATIO = 1.4
# Below this the two pictures have nothing structurally in common - which, for
# a comparable pair, is a genuine swap.
UNRELATED_SCORE = 0.55
# A figure one document detects as one illustration the other sometimes detects
# as two adjacent pieces. Before calling anything lost, a Production figure is
# retried against the union of adjacent STAGE pieces - but only while that union
# stays close to the original's size, so unrelated neighbours can't be glued
# together into a "match".
UNION_MAX_AREA_RATIO = 2.5
SECTION_SPILLOVER_PAGES = 1  # pages either side of a section searched for a stray figure

CAPTION_GAP = 30.0  # points directly below/above a figure its caption may sit in
CAPTION_MAX_WORDS = 40  # longer than this isn't a caption, it's body text - don't treat it as one
# A real figure caption is a short label ("Front view", "Figure 3-1. Cable
# routing"). A block near a figure that runs to full sentences, is a numbered
# step, a bullet, or a NOTE/TIP callout is body text that merely sits close to
# the figure - comparing it as a "caption" is what produced false "Image label
# missing" findings over notes and step instructions that were never lost.
CAPTION_STRICT_MAX_WORDS = 8
_CAPTION_REJECT_LEAD_RE = re.compile(
    r"^\s*(?:\d{1,2}[.)]|[a-zA-Z][.)]|[•◦▪‣⁃·•]|"
    r"note|tip|warning|caution|important)\b",
    re.IGNORECASE,
)

# A highlight box is a coloured RECTANGULAR OUTLINE deliberately drawn on top
# of a figure to call attention to part of it. It has to be told apart from the
# figure's own line art and from coloured icon/fill shapes, which is what the
# stroke-only + rectangle-shape + saturation + per-figure-cap tests below do
# together. (Saturation alone matched every coloured glyph in an icon and
# produced hundreds of false "highlight box missing" findings.)
HIGHLIGHT_MIN_SATURATION = 0.45
HIGHLIGHT_MIN_SIDE = 16.0  # points
HIGHLIGHT_MAX_SIDE_FRACTION = 0.98  # a "box" as big as the whole figure is the figure
HIGHLIGHT_MAX_PER_FIGURE = 4  # more candidate boxes than this => it's artwork, not highlights
HIGHLIGHT_MATCH_DISTANCE = 60.0  # points - how far a counterpart box may sit

BLANK_STDDEV = 1.0  # a rendered figure flatter than this is blank
LOW_CONTRAST_STDDEV = 3.0  # very low contrast image (high correlation = likely corrupted)
MONOCHROME_THRESHOLD = 0.85  # share of pixels matching dominant color = mostly one color
CORRUPTION_MIN_PIXELS = 100  # minimum pixels in a rendered image before checking quality

_BARE_NUMBER_RE = re.compile(r"^\(?(\d{1,2})[.)]?$")
_LABEL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is", "are",
    "with", "by", "from", "as", "it", "this", "that", "be", "you", "your",
}
MIN_LABEL_WORD_LEN = 3


def _is_meaningful_size(img: ImageInfo) -> bool:
    return (
        img.display_width >= MIN_FIGURE_SIDE
        and img.display_height >= MIN_FIGURE_SIDE
        and img.display_area >= MIN_FIGURE_AREA
    )


def _with_heading(details: dict, heading: str | None) -> dict:
    if heading:
        details = {"heading": heading, **details}
    return details


def validate_images(
    expected: fitz.Document, actual: fitz.Document, output_dir: str | None = None
) -> CheckResult:
    result = CheckResult(name="Image Validation")
    exp_entries, act_entries = resolve_entries(expected, actual)
    counter = itertools.count(1)
    ctx = _Context(expected, actual, output_dir, counter)

    if exp_entries and act_entries:
        # Heading-anchored: page-index matching (page N vs page N) silently
        # compares the wrong pages once the two documents' page counts diverge.
        for exp_entry, act_entry, exp_bounds, act_bounds in matched_heading_ranges(
            exp_entries, act_entries, expected.page_count, actual.page_count
        ):
            exp_figs = _figures_in_bounds(expected, exp_bounds)
            act_figs = _figures_in_bounds(actual, act_bounds)
            _compare_section(result, ctx, exp_figs, act_figs, exp_entry.title, exp_bounds, act_bounds)
    else:
        _validate_by_page_index(result, ctx)

    result.summary_rows = ctx.summary
    result.summary_title = "Figures per section (Prod vs Stage)"
    return result


class _Context:
    """Everything the per-section checks need that isn't section-specific."""

    def __init__(
        self,
        expected: fitz.Document,
        actual: fitz.Document,
        output_dir: str | None,
        counter: "itertools.count",
    ):
        self.expected = expected
        self.actual = actual
        self.output_dir = output_dir
        self.counter = counter
        self.exp_words = ocr.PageWords(expected)
        self.act_words = ocr.PageWords(actual)
        self._fingerprints: dict[tuple, imagefp.Fingerprint] = {}
        self._renders: dict[tuple, object] = {}
        # One row per section for the report's figure-by-figure summary table.
        self.summary: list[dict] = []

    def fingerprint(self, doc: fitz.Document, page: int, bbox: tuple) -> imagefp.Fingerprint:
        """Fingerprints are memoised: the same figure is compared against many
        candidates, and rasterising its region once per comparison dominated
        the check's run time."""
        key = (id(doc), page, tuple(round(v, 1) for v in bbox))
        if key not in self._fingerprints:
            self._fingerprints[key] = imagefp.fingerprint(doc, page, bbox)
        return self._fingerprints[key]

    def render(self, doc: fitz.Document, page: int, bbox: tuple):
        """Memoised common-scale render, for the sliding second opinion."""
        key = (id(doc), page, tuple(round(v, 1) for v in bbox))
        if key not in self._renders:
            self._renders[key] = imagefp.render_gray(doc, page, bbox)
        return self._renders[key]



_TABLE_OVERLAP_FRACTION = 0.75  # a "figure" this much inside a detected table IS the table


def _table_regions(doc: fitz.Document, page: int) -> list[tuple]:
    """pdfplumber's detected grid regions for a page, or [] when the document's
    path isn't available (whole-doc diffs from an in-memory document)."""
    path = getattr(doc, "name", "") or ""
    if not path:
        return []
    try:
        return get_all_detected_regions(path, page)
    except Exception:
        return []


def _is_really_a_table(doc: fitz.Document, page: int, img: ImageInfo) -> bool:
    """A vector 'figure' whose box sits almost entirely inside a detected table
    grid is that table's ruled borders picked up as line art - Table Validation
    owns it, Image Validation should not also be comparing it as a picture."""
    if img.kind != "vector":
        return False
    for region in _table_regions(doc, page):
        if _overlap_fraction(img.bbox, region) >= _TABLE_OVERLAP_FRACTION:
            return True
    return False


def _overlap_fraction(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    a_area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ((ix1 - ix0) * (iy1 - iy0)) / a_area


def _figures_in_bounds(doc: fitz.Document, bounds) -> list[tuple[int, ImageInfo]]:
    first_page = max(0, bounds.start_page)
    last_page = min(bounds.end_page, doc.page_count - 1)
    items: list[tuple[int, ImageInfo]] = []
    for p in range(first_page, last_page + 1):
        items.extend(
            (p, img)
            for img in get_figures(doc, p)
            if _is_meaningful_size(img)
            and in_section_bounds(p, img.bbox[1], bounds)
            and not _is_really_a_table(doc, p, img)
        )
    # Reading order, which is what pairs a section's figures across documents.
    items.sort(key=lambda item: (item[0], round(item[1].bbox[1], 1), round(item[1].bbox[0], 1)))
    return items


def _validate_by_page_index(result: CheckResult, ctx: _Context) -> None:
    """Fallback used only when one/both documents have no TOC to anchor by."""
    exp_entries, act_entries = resolve_entries(ctx.expected, ctx.actual)
    common_pages = min(ctx.expected.page_count, ctx.actual.page_count)

    for i in range(common_pages):
        exp_figs = [
            (i, img)
            for img in get_figures(ctx.expected, i)
            if _is_meaningful_size(img) and not _is_really_a_table(ctx.expected, i, img)
        ]
        act_figs = [
            (i, img)
            for img in get_figures(ctx.actual, i)
            if _is_meaningful_size(img) and not _is_really_a_table(ctx.actual, i, img)
        ]
        heading = None
        if exp_figs:
            heading = heading_at(exp_entries, i, exp_figs[0][1].bbox[1])
        if not heading and act_figs:
            heading = heading_at(act_entries, i, act_figs[0][1].bbox[1])
        _compare_section(result, ctx, exp_figs, act_figs, heading, None, None)


@dataclass
class _Pair:
    """One Production figure and what STAGE has in its place.

    `status` is one of:
      same      - the same picture (page may differ; that is not a finding)
      resized   - the same picture, drawn at a materially different shape/size
      differs   - a figure occupies the slot, and it is NOT the same picture
      review    - a figure occupies the slot and could not be confirmed either
                  way, because the two documents detected different extents of
                  it - a human decides this one
      outside   - the picture is in STAGE, but past the end of its section
      missing   - nothing in STAGE corresponds to it
    """

    exp_page: int
    exp_img: ImageInfo
    act_page: int | None
    act_img: ImageInfo | None
    score: float
    status: str
    note: str = ""
    size_changed: bool = False  # set by _check_dimensions: same picture, resized


def _fig_area(img: ImageInfo) -> float:
    return max(1.0, img.display_area)


def _union_info(a: ImageInfo, b: ImageInfo) -> ImageInfo:
    bbox = (
        min(a.bbox[0], b.bbox[0]),
        min(a.bbox[1], b.bbox[1]),
        max(a.bbox[2], b.bbox[2]),
        max(a.bbox[3], b.bbox[3]),
    )
    return ImageInfo(xref=-1, bbox=bbox, width=int(bbox[2] - bbox[0]), height=int(bbox[3] - bbox[1]), kind="group")


def _score(ctx: _Context, exp_item, act_item) -> float:
    exp_fp = ctx.fingerprint(ctx.expected, exp_item[0], exp_item[1].bbox)
    act_fp = ctx.fingerprint(ctx.actual, act_item[0], act_item[1].bbox)
    # A blank/flat region correlates arbitrarily against anything; refusing to
    # score it keeps a blank box from being "matched" to a real illustration.
    # (STAGE artwork that renders blank is reported by `_check_broken_images`.)
    if exp_fp.featureless or act_fp.featureless:
        return 0.0
    return imagefp.similarity(exp_fp, act_fp)


def _aspect_gap(ctx: _Context, exp_item, act_item) -> float:
    a = ctx.fingerprint(ctx.expected, exp_item[0], exp_item[1].bbox).aspect
    b = ctx.fingerprint(ctx.actual, act_item[0], act_item[1].bbox).aspect
    if a <= 0 or b <= 0:
        return 0.0
    return abs(a - b) / max(a, b)


def _slide_score(ctx: _Context, exp_item, act_item) -> float | None:
    """The forgiving second opinion, used only where the cheap one was unsure."""
    a = ctx.render(ctx.expected, exp_item[0], exp_item[1].bbox)
    b = ctx.render(ctx.actual, act_item[0], act_item[1].bbox)
    return imagefp.region_match(a, b)


def _comparable_extents(ctx: _Context, exp_item, act_item) -> bool:
    """True when the two detections cover close enough to the same thing that a
    pixel comparison between them means anything at all."""
    if _aspect_gap(ctx, exp_item, act_item) > ASPECT_TOLERANCE:
        return False
    exp_area = _fig_area(exp_item[1])
    act_area = _fig_area(act_item[1])
    ratio = max(exp_area, act_area) / min(exp_area, act_area)
    return ratio <= COMPARABLE_AREA_RATIO


def _match_figures(
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    act_figs: list[tuple[int, ImageInfo]],
    spill_figs: list[tuple[int, ImageInfo]],
) -> list[_Pair]:
    """Pair each Production figure with the STAGE figure that IS it, then say
    what happened to the ones left over.

    Confident pairs are taken first, best score first and globally - not in
    reading order - so a figure that moved to a different page inside the
    section still finds itself, and one lost figure doesn't shift every later
    pairing by one. Only what remains after that is judged by position.
    """
    pairs: dict[int, _Pair] = {}
    used: set[int] = set()

    candidates = []
    for i, exp_item in enumerate(exp_figs):
        for j, act_item in enumerate(act_figs):
            score = _score(ctx, exp_item, act_item)
            if score >= SAME_IMAGE_SCORE:
                candidates.append((score, i, j))
    for score, i, j in sorted(candidates, key=lambda c: -c[0]):
        if i in pairs or j in used:
            continue
        used.add(j)
        pairs[i] = _Pair(exp_figs[i][0], exp_figs[i][1], act_figs[j][0], act_figs[j][1], score, "same")

    _match_split_figures(ctx, exp_figs, act_figs, pairs, used)
    _match_by_sliding(ctx, exp_figs, act_figs, pairs, used)
    _match_spillover(ctx, exp_figs, spill_figs, pairs)
    _classify_leftovers(ctx, exp_figs, act_figs, pairs, used)
    return [pairs[i] for i in sorted(pairs)]


def _match_by_sliding(
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    act_figs: list[tuple[int, ImageInfo]],
    pairs: dict[int, _Pair],
    used: set[int],
) -> None:
    """Second matching round for what the cheap comparison could not place.

    This is where a figure whose detected extent differs between the two
    documents gets recognised - the picture is the same, one side's crop just
    also caught the leader lines or the sub-figure beside it. Only pairs that
    already share SOME structure are searched, so this stays cheap.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, exp_item in enumerate(exp_figs):
        if i in pairs:
            continue
        for j, act_item in enumerate(act_figs):
            if j in used or _score(ctx, exp_item, act_item) < SLIDE_MIN_INTEREST:
                continue
            score = _slide_score(ctx, exp_item, act_item)
            if score is not None and score >= SLIDE_SAME_SCORE:
                candidates.append((score, i, j))
    for score, i, j in sorted(candidates, key=lambda c: -c[0]):
        if i in pairs or j in used:
            continue
        used.add(j)
        pairs[i] = _Pair(
            exp_figs[i][0], exp_figs[i][1], act_figs[j][0], act_figs[j][1], score, "same",
            note="matched after allowing for the two documents detecting different extents of it",
        )


def _match_split_figures(
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    act_figs: list[tuple[int, ImageInfo]],
    pairs: dict[int, _Pair],
    used: set[int],
) -> None:
    """Retry each still-unmatched Production figure against the union of two
    adjacent STAGE pieces.

    One document's exporter draws an illustration as a single object and the
    other draws it as two touching ones; figure detection then reports one
    figure on one side and two on the other, and comparing either piece alone
    against the whole scores like a different picture. Comparing the pieces
    TOGETHER is comparing what the reader actually sees.
    """
    for i, exp_item in enumerate(exp_figs):
        if i in pairs:
            continue
        exp_area = _fig_area(exp_item[1])
        best = None
        for j in range(len(act_figs) - 1):
            k = j + 1
            if j in used or k in used or act_figs[j][0] != act_figs[k][0]:
                continue
            group = _union_info(act_figs[j][1], act_figs[k][1])
            if _fig_area(group) > exp_area * UNION_MAX_AREA_RATIO:
                continue
            score = _score(ctx, exp_item, (act_figs[j][0], group))
            if score >= SAME_IMAGE_SCORE and (best is None or score > best[0]):
                best = (score, j, k, group)
        if best is None:
            continue
        score, j, k, group = best
        used.update((j, k))
        pairs[i] = _Pair(exp_item[0], exp_item[1], act_figs[j][0], group, score, "same")


def _match_spillover(
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    spill_figs: list[tuple[int, ImageInfo]],
    pairs: dict[int, _Pair],
) -> None:
    """Last chance for an unmatched figure: the pages just outside the section.

    A figure that repaginated onto the next page is normally still inside its
    section and needs nothing special. This catches the case where it slid past
    the section's own end - the picture is not lost, it has left its section,
    and those are different things to tell the reader.
    """
    if not spill_figs:
        return
    used: set[int] = set()
    for i, exp_item in enumerate(exp_figs):
        if i in pairs:
            continue
        best = None
        for j, act_item in enumerate(spill_figs):
            if j in used:
                continue
            score = _score(ctx, exp_item, act_item)
            if score >= SAME_IMAGE_SCORE and (best is None or score > best[0]):
                best = (score, j)
        if best is None:
            continue
        score, j = best
        used.add(j)
        pairs[i] = _Pair(exp_item[0], exp_item[1], spill_figs[j][0], spill_figs[j][1], score, "outside")


def _classify_leftovers(
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    act_figs: list[tuple[int, ImageInfo]],
    pairs: dict[int, _Pair],
    used: set[int],
) -> None:
    """What is left has no confident counterpart. Line the leftovers up in
    reading order - Production's unmatched figures against STAGE's - and the
    ones that fall opposite each other are the same SLOT in the section: the
    section still shows a figure there, it just isn't the same picture. A
    Production figure with no leftover opposite it is simply gone.
    """
    exp_left = [i for i in range(len(exp_figs)) if i not in pairs]
    act_left = [j for j in range(len(act_figs)) if j not in used]

    for pos, i in enumerate(exp_left):
        exp_page, exp_img = exp_figs[i]
        if pos >= len(act_left):
            pairs[i] = _Pair(exp_page, exp_img, None, None, 0.0, "missing")
            continue
        act_page, act_img = act_figs[act_left[pos]]
        exp_item, act_item = (exp_page, exp_img), (act_page, act_img)
        score = _score(ctx, exp_item, act_item)
        slide = _slide_score(ctx, exp_item, act_item)
        if slide is not None:
            score = max(score, slide)

        if score >= RESIZED_IMAGE_SCORE and _aspect_gap(ctx, exp_item, act_item) > ASPECT_TOLERANCE:
            # Same picture, just drawn/placed at a different size - dimensions
            # are not something this report validates, so this is not a
            # finding at all rather than a downgraded one.
            status, note = "same", ""
        elif _comparable_extents(ctx, exp_item, act_item) and score >= NEAR_IDENTICAL_SCORE:
            # Same extent, high score - the same picture, re-rendered slightly.
            # Not a finding.
            status, note = "same", ""
        elif _comparable_extents(ctx, exp_item, act_item) and score < UNRELATED_SCORE:
            # Same extent, and the pictures have almost nothing in common - the
            # figure in this slot has genuinely changed (different content, or a
            # whole layer of annotation added/removed). A defect worth
            # confirming.
            status, note = "differs", (
                "the two figures occupy the same slot and cover the same extent, but the "
                "picture is not the same - content or annotation (callout numbers, leader "
                "lines, a highlighted area) differs substantially between them"
            )
        elif _comparable_extents(ctx, exp_item, act_item):
            # Same extent, middling score - the pictures share structure but
            # differ in ways a similarity number can't safely classify. Held
            # for a human.
            status = "review"
            note = "the two figures are partly alike - the same subject, drawn or annotated differently"
        else:
            # The crops are NOT comparable - the two documents detected
            # different extents of the figure - so a low pixel score says the
            # detections disagree, not that the picture changed. A person
            # decides this one.
            status = "review"
            note = (
                "the two documents detected different extents of this figure, so their "
                "artwork could not be compared like for like"
            )
        pairs[i] = _Pair(exp_page, exp_img, act_page, act_img, score, status, note=note)


_STATUS_MESSAGE = {
    "differs": ("Image content differs", "error"),
    "review": ("Image could not be confirmed identical", "warning"),
    "outside": ("Image outside its section", "warning"),
    "missing": ("Image missing", "error"),
}


def _report_matches(result: CheckResult, ctx: _Context, pairs: list[_Pair], heading: str | None) -> None:
    for pair in pairs:
        if pair.status == "same":
            continue
        message, severity = _STATUS_MESSAGE[pair.status]
        details: dict = {"expected_page": pair.exp_page + 1}
        if pair.act_page is not None:
            details["actual_page"] = pair.act_page + 1
        if pair.status == "missing":
            details["reason"] = "no figure anywhere in the Staging section resembles this one"
        elif pair.status == "outside":
            details["reason"] = (
                f"this figure was found on Staging page {pair.act_page + 1}, which is past the end "
                "of its section - it is not lost, but it no longer sits under this heading"
            )
        elif pair.status == "differs":
            details["reason"] = (
                "the Staging section shows a figure in this figure's place, and the two have "
                "nothing in common - compare the two images below"
            )
        else:
            details["reason"] = pair.note
            # Nothing is being asserted here, so it belongs in the report's
            # held-for-review block rather than among the confirmed findings.
            details["confidence"] = "review"
        if pair.status != "missing":
            details["similarity"] = f"{pair.score:.0%}"
        details["kind"] = pair.exp_img.kind
        details["bbox"] = pair.exp_img.bbox
        details = _with_heading(details, heading)
        # Only show the Staging crop when it is a real figure region. A
        # mis-detection that swallowed a column of text renders as a full page
        # and tells the reader nothing - better to show the Production figure
        # alone and say why.
        act_page = pair.act_page
        act_bbox = pair.act_img.bbox if pair.act_img else None
        if act_bbox is not None and not _is_sane_figure_region(ctx.actual, act_page, act_bbox):
            act_page, act_bbox = None, None
            details["staging_figure"] = (
                "the matching Staging region could not be isolated to a figure, so only the "
                "Production figure is shown"
            )
        _attach(ctx, details, pair.exp_page, pair.exp_img.bbox, act_page, act_bbox)
        result.issues.append(Issue(severity=severity, page=pair.exp_page, message=message, details=details))


def _record_summary(
    ctx: _Context,
    heading: str | None,
    pairs: list[_Pair],
    exp_figs: list[tuple[int, ImageInfo]],
    act_figs: list[tuple[int, ImageInfo]],
) -> None:
    """One row per section, including sections where everything matched - a
    reader has no way to tell "checked and clean" from "not checked" out of an
    issue list alone.
    """
    if not exp_figs and not act_figs:
        return
    counts = {status: sum(1 for p in pairs if p.status == status) for status in
              ("same", "differs", "review", "outside", "missing")}
    counts["resized"] = sum(1 for p in pairs if p.size_changed)
    # A figure that changed page inside its own section is explicitly fine; the
    # summary says how many did so rather than staying silent about it.
    repaginated = sum(
        1 for p in pairs if p.status == "same" and p.act_page is not None and p.act_page != p.exp_page
    )
    ctx.summary.append(
        {
            "heading": heading or "(no heading)",
            "expected_figures": len(exp_figs),
            "actual_figures": len(act_figs),
            "matched": counts["same"],
            "moved_page": repaginated,
            "differs": counts["differs"],
            "needs_review": counts["review"],
            "resized": counts["resized"],
            "outside_section": counts["outside"],
            "missing": counts["missing"],
            "status": _summary_status(counts),
        }
    )


def _summary_status(counts: dict[str, int]) -> str:
    if counts["differs"] or counts["missing"]:
        return "Issues"
    if counts["review"] or counts["resized"] or counts["outside"]:
        return "Review"
    return "OK"


def _compare_section(
    result: CheckResult,
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    act_figs: list[tuple[int, ImageInfo]],
    heading: str | None,
    exp_bounds,
    act_bounds,
) -> None:
    spill_figs = _spillover_figures(ctx, act_bounds, act_figs)
    pairs = _match_figures(ctx, exp_figs, act_figs, spill_figs)
    counterparts = _counterparts(ctx, pairs)
    _report_matches(result, ctx, pairs, heading)
    _check_broken_images(result, ctx, act_figs, heading)
    _check_blurred(result, ctx, pairs, heading)
    _check_dimensions(result, ctx, pairs, heading)
    _record_summary(ctx, heading, pairs, exp_figs, act_figs)
    _check_alignment(result, ctx, pairs, heading)
    _check_highlight_boxes(result, ctx, exp_figs, counterparts, heading)
    _check_labels(result, ctx, pairs, heading)
    _check_callout_numbers(result, ctx, pairs, heading)


def _fig_key(page: int, img: ImageInfo) -> tuple:
    return (page, tuple(round(v, 1) for v in img.bbox))


# Only a figure the matcher is SURE about is offered to the per-figure checks
# as a Staging screenshot. A "review"/"differs" pairing is, by definition, a
# STAGE region the matcher could not line up with the PROD figure - rendering
# it beside the finding shows the reader two things that may not be the same
# picture at all.
_CONFIDENT_STATUSES = ("same", "resized")
# A "figure" bbox this large is a mis-detection that swallowed a page of text,
# not an illustration - zooming a screenshot to it just renders the whole page.
_MAX_FIGURE_PAGE_FRACTION = 0.60
_MAX_FIGURE_SIDE_FRACTION = 0.92


def _is_sane_figure_region(doc: fitz.Document, page: int, bbox: tuple) -> bool:
    try:
        page_rect = doc[page].rect
    except Exception:
        return False
    pw = max(1.0, page_rect.width)
    ph = max(1.0, page_rect.height)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return False
    if (w * h) / (pw * ph) > _MAX_FIGURE_PAGE_FRACTION:
        return False
    return w / pw <= _MAX_FIGURE_SIDE_FRACTION and h / ph <= _MAX_FIGURE_SIDE_FRACTION


def _counterparts(
    ctx: _Context, pairs: list[_Pair]
) -> dict[tuple, tuple[int, ImageInfo] | None]:
    """Which STAGE figure each PROD figure actually turned out to be - but ONLY
    when the matcher is confident about it and the STAGE region is a real
    figure crop (not a mis-detection that swallowed half a page).

    Every per-figure finding (label missing, callout number missing, resized,
    highlight box missing) shows the reader a picture from each document, and
    that picture is only useful if it is the RIGHT one, zoomed to the figure.
    When there is no counterpart that clears both bars, the finding shows the
    Production figure alone rather than an arbitrary Staging page.
    """
    out: dict[tuple, tuple[int, ImageInfo] | None] = {}
    for pair in pairs:
        key = _fig_key(pair.exp_page, pair.exp_img)
        if (
            pair.act_img is not None
            and pair.act_page is not None
            and pair.status in _CONFIDENT_STATUSES
            and _is_sane_figure_region(ctx.actual, pair.act_page, pair.act_img.bbox)
        ):
            out[key] = (pair.act_page, pair.act_img)
        else:
            out[key] = None
    return out


def _spillover_figures(
    ctx: _Context, act_bounds, act_figs: list[tuple[int, ImageInfo]]
) -> list[tuple[int, ImageInfo]]:
    """STAGE figures on the pages just around the section that are NOT already
    part of it - candidates for a figure that slipped out of its section."""
    if act_bounds is None:
        return []
    inside = {(p, tuple(round(v, 1) for v in img.bbox)) for p, img in act_figs}
    out: list[tuple[int, ImageInfo]] = []
    for p in _section_pages(act_bounds, ctx.actual.page_count, SECTION_SPILLOVER_PAGES):
        for img in get_figures(ctx.actual, p):
            if not _is_meaningful_size(img):
                continue
            if (p, tuple(round(v, 1) for v in img.bbox)) in inside:
                continue
            out.append((p, img))
    return out


def _check_broken_images(
    result: CheckResult, ctx: _Context, act_figs: list[tuple[int, ImageInfo]], heading: str | None
) -> None:
    """Artwork that is present in STAGE but unusable: the embedded stream will
    not decode, or the region renders as a single flat colour (a blank box
    where a figure should be).
    """
    for act_page, act_img in act_figs:
        reason = _broken_reason(ctx.actual, act_page, act_img)
        if not reason:
            continue
        details = _with_heading({"reason": reason, "kind": act_img.kind, "bbox": act_img.bbox}, heading)
        _attach(ctx, details, None, None, act_page, act_img.bbox)
        result.issues.append(
            Issue(severity="error", page=act_page, message="Broken image", details=details)
        )


def _broken_reason(doc: fitz.Document, page: int, img: ImageInfo, stream_only: bool = False) -> str | None:
    """Why this picture will not show, or None when it will.

    `stream_only` keeps to the size-independent half of the test: an image
    whose stream is empty or will not decode is broken whether it is a
    full-page diagram or a 10pt icon. The render heuristics below are the
    half that needs room to judge - a small icon legitimately paints as one
    flat colour - so an icon is asked only the first question.
    """
    if img.kind == "raster" and img.xref > 0:
        try:
            if not doc.extract_image(img.xref).get("image"):
                return "embedded image stream is empty"
        except Exception:
            return "embedded image stream could not be decoded"
    if stream_only:
        return None
    try:
        rect = fitz.Rect(*img.bbox) & doc[page].rect
        if rect.is_empty:
            return "figure lies outside the page"
        pix = doc[page].get_pixmap(clip=rect, alpha=False)
        if not pix.width or not pix.height:
            return "figure renders with no pixels"
        # Check various corruption/quality issues
        quality_issue = _check_image_quality(pix)
        if quality_issue:
            return quality_issue
    except Exception:
        return "figure could not be rendered"
    return None


def _stddev(pix: fitz.Pixmap) -> float:
    try:
        import numpy as np

        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        return float(arr.std())
    except Exception:
        # Without numpy, sample the bytes rather than give up and call it blank.
        data = pix.samples[::17] or b"\x00"
        mean = sum(data) / len(data)
        return (sum((b - mean) ** 2 for b in data) / len(data)) ** 0.5


def _check_image_quality(pix: fitz.Pixmap) -> str | None:
    """Blank/corrupted-render check only: a figure that decodes but paints as a
    single flat colour. Deliberately NOT flagging low-contrast or mostly-
    white/black renders - a legitimate icon or logo on a white background is
    exactly that (confirmed: flagged 'Broken image' on unmodified figures
    compared against themselves), so those checks were removed rather than
    tuned.
    """
    if pix.width * pix.height < CORRUPTION_MIN_PIXELS:
        return None
    std = _stddev(pix)
    if std < BLANK_STDDEV:
        return "figure renders blank (a single flat colour)"
    return None


def _page_offset_fraction(doc: fitz.Document, page: int, img: ImageInfo) -> float:
    """Signed offset of the figure's centre from the PAGE centre, as a fraction
    of page width: ~0 = centred, negative = shifted left, positive = right.
    Measured against the page box (not the text column) because the text
    column's modal edges still drift a little between two re-exports, which was
    enough to flip a near-centred figure's class on its own."""
    page_rect = doc[page].rect
    width = max(1.0, page_rect.width)
    fig_center = (img.bbox[0] + img.bbox[2]) / 2
    return (fig_center - (page_rect.x0 + page_rect.x1) / 2) / width


# A figure's rendered size differs by less than this between the two documents
# for reasons no one can act on - a re-export rounds a scale factor, a vector
# cluster's bounds land a point or two apart. Above it, the figure was
# deliberately made bigger or smaller. Both a fractional and an absolute floor
# have to be cleared, so a small icon isn't flagged over a 3pt wobble and a
# full-page diagram isn't flagged over a 4% rounding.
DIMENSION_CHANGE_FRACTION = 0.15
DIMENSION_CHANGE_MIN_PT = 12.0
# A real deliberate resize keeps the figure recognisably the same size. Beyond
# this ratio on either axis the two "sizes" are really two DETECTIONS that
# disagree about where the figure ends (one caught the caption, the leader
# lines, the sub-figure beside it), which is not a resize and is already
# covered by "could not be confirmed identical".
DIMENSION_MAX_RATIO = 2.0


def _dimensions_comparable(exp_fp: "imagefp.Fingerprint", act_fp: "imagefp.Fingerprint") -> bool:
    """True when the two figure sizes are close enough that comparing them
    reports a RESIZE rather than a detection disagreement. Also requires the two
    to be scaled roughly UNIFORMLY - a real resize keeps the proportions, a
    detection that grew on one axis only did not."""
    for ev, av in ((exp_fp.width, act_fp.width), (exp_fp.height, act_fp.height)):
        if ev <= 0 or av <= 0:
            return False
        ratio = av / ev
        if ratio > DIMENSION_MAX_RATIO or ratio < 1 / DIMENSION_MAX_RATIO:
            return False
    w_ratio = act_fp.width / exp_fp.width
    h_ratio = act_fp.height / exp_fp.height
    # The two axes must scale within ~20% of each other (a uniform resize), OR
    # one axis is essentially unchanged (a deliberate stretch of the other).
    if abs(w_ratio - h_ratio) / max(w_ratio, h_ratio) <= 0.2:
        return True
    return abs(w_ratio - 1) < 0.06 or abs(h_ratio - 1) < 0.06


def _check_dimensions(
    result: CheckResult, ctx: _Context, pairs: list[_Pair], heading: str | None
) -> None:
    """Flags the pairs that are the SAME picture rendered at a materially
    different width or height in Staging - scaled up or down.

    This is recorded (`pair.size_changed`, shown in the figure-by-figure
    summary) but NOT raised as an issue: a resize is a layout choice, and
    reporting one per figure drowned the findings that actually break the
    document. The strict qualifying conditions below are what make the count
    trustworthy: both documents must have detected the figure the same way and
    at a comparable extent, or the two boxes differ for detection reasons that
    have nothing to do with a resize.
    """
    for pair in pairs:
        if pair.status != "same" or pair.act_img is None:
            continue
        if pair.note or pair.act_img.kind == "group":
            continue  # matched despite differing detected extents - size not comparable
        # A vector figure's bounding box is the cluster of its strokes and
        # includes the whitespace and leader lines around the artwork; a raster
        # figure's box is tight to the image. When Production draws a figure as
        # vector art and Staging embeds the same figure as a raster (common -
        # different export pipeline), the two boxes differ by ~20% for that
        # reason alone, which is not a resize. Only compare sizes when both
        # sides detected the figure the same way.
        if pair.exp_img.kind != pair.act_img.kind:
            continue
        # Both detections must be believable figure regions, and the two must
        # not disagree wildly about the figure's extent - either of those means
        # a detection artefact, not a resize.
        if not _is_sane_figure_region(ctx.expected, pair.exp_page, pair.exp_img.bbox):
            continue
        if not _is_sane_figure_region(ctx.actual, pair.act_page, pair.act_img.bbox):
            continue
        exp_fp = ctx.fingerprint(ctx.expected, pair.exp_page, pair.exp_img.bbox)
        act_fp = ctx.fingerprint(ctx.actual, pair.act_page, pair.act_img.bbox)
        if not exp_fp.usable or not act_fp.usable:
            continue
        if not _dimensions_comparable(exp_fp, act_fp):
            continue

        changes = []
        for label, bigger, smaller, ev, av in (
            ("width", "wider", "narrower", exp_fp.width, act_fp.width),
            ("height", "taller", "shorter", exp_fp.height, act_fp.height),
        ):
            delta = av - ev
            if abs(delta) < DIMENSION_CHANGE_MIN_PT:
                continue
            if abs(delta) / max(1.0, ev) < DIMENSION_CHANGE_FRACTION:
                continue
            changes.append(
                {
                    "axis": label,
                    "direction": bigger if delta > 0 else smaller,
                    "expected_pt": round(ev, 1),
                    "actual_pt": round(av, 1),
                    "change": f"{delta:+.0f} pt ({delta / max(1.0, ev):+.0%})",
                }
            )
        if not changes:
            continue

        # NOT a finding. The same picture rendered wider or taller is a layout
        # choice, not a defect - and flagged per figure it buried the breaking
        # issues (a dropped label, a stripped callout, a blank render) under
        # rows nobody acts on. The resize is still COUNTED, so the figure-by-
        # figure summary table below still reports it.
        pair.size_changed = True


# A figure is called blurred only when the SAME picture is materially softer in
# Staging. Calibrated against known blurs of one real page: a 2px Gaussian
# measures 0.57 of the sharp original and a 4px one 0.37, while a mild
# half-resolution upscale still measures 0.88 - so 0.65 catches a genuine
# resolution loss and leaves ordinary re-encoding alone.
BLUR_RATIO = 0.65
BLUR_MIN_PROD_SHARPNESS = 0.05  # below this Production is soft too - nothing to compare


def _check_blurred(
    result: CheckResult, ctx: _Context, pairs: list[_Pair], heading: str | None
) -> None:
    """The same picture, rendered soft or low-resolution in Staging.

    Nothing else here catches this: the fingerprint correlates at ~1.0 because
    it IS the same picture, the dimension check sees the same extent, and
    `_check_broken_images` only fires on artwork that is blank or fails to
    decode. A figure that came through the rebuild as an upscaled low-res copy
    is unreadable in print and passes every other check.
    """
    for pair in pairs:
        if pair.status != "same" or pair.act_img is None:
            continue
        exp_sharp = imagefp.sharpness(ctx.expected, pair.exp_page, pair.exp_img.bbox)
        act_sharp = imagefp.sharpness(ctx.actual, pair.act_page, pair.act_img.bbox)
        if exp_sharp is None or act_sharp is None:
            continue
        if exp_sharp < BLUR_MIN_PROD_SHARPNESS:
            continue
        ratio = act_sharp / exp_sharp
        if ratio >= BLUR_RATIO:
            continue
        details = _with_heading(
            {
                "expected_page": pair.exp_page + 1,
                "actual_page": pair.act_page + 1,
                "reason": (
                    "the same figure, rendered soft in Staging - it carries "
                    f"{ratio:.0%} of Production's edge detail"
                ),
                "bbox": pair.exp_img.bbox,
                "similarity": f"{pair.score:.0%}",
            },
            heading,
        )
        _attach(ctx, details, pair.exp_page, pair.exp_img.bbox, pair.act_page, pair.act_img.bbox)
        result.issues.append(
            Issue(severity="error", page=pair.exp_page, message="Image blurred", details=details)
        )


def _check_alignment(
    result: CheckResult, ctx: _Context, pairs: list[_Pair], heading: str | None
) -> None:
    """Only a change of alignment CLASS is reported. A figure that moved a few
    points is not a finding - that happens whenever the text above it reflows -
    but a figure that was centred and is now flush left is a layout regression.

    Checked only on figures confirmed to BE the same picture. Asking where a
    figure moved to is meaningless for one that is missing or has been replaced,
    and the finding that describes what happened to those has already been made.
    """
    for pair in pairs:
        if pair.status not in ("same", "resized") or pair.act_img is None:
            continue
        exp_page, exp_img = pair.exp_page, pair.exp_img
        act_page, act_img = pair.act_page, pair.act_img
        exp_off = _page_offset_fraction(ctx.expected, exp_page, exp_img)
        act_off = _page_offset_fraction(ctx.actual, act_page, act_img)

        def _cls(off: float) -> str:
            if abs(off) <= 0.08:
                return "center"
            return "left" if off < 0 else "right"

        exp_align, act_align = _cls(exp_off), _cls(act_off)
        if exp_align == act_align:
            continue
        # Report only a genuine, sizeable shift: the figure's centre moved by
        # more than 12% of the page width AND at least one side of the change
        # is "centred" (a centred figure shoved to an edge, or vice versa).
        # A left<->right flip with no centred state, or a small nudge, is
        # reference-frame noise, not a layout regression.
        if abs(exp_off - act_off) < 0.12 or "center" not in (exp_align, act_align):
            continue
        details = _with_heading(
            {
                "expected_page": exp_page + 1,
                "actual_page": act_page + 1,
                "expected_alignment": exp_align,
                "actual_alignment": act_align,
            },
            heading,
        )
        _attach(ctx, details, exp_page, exp_img.bbox, act_page, act_img.bbox)
        result.issues.append(
            Issue(severity="warning", page=exp_page, message="Image alignment changed", details=details)
        )


def _is_rectangle_outline(d: dict) -> bool:
    """True when the drawing is a rectangular OUTLINE - a single "re" item, or
    a closed path of four axis-aligned line segments - rather than a fill, a
    curve, or an open stroke. An attention-highlight box is always a rectangle
    outline; a coloured icon shape is not.
    """
    items = d.get("items") or []
    if not items:
        return False
    if d.get("type") not in ("s", "sf", None):  # must be stroked (not a pure fill "f")
        return False
    if len(items) == 1 and items[0][0] == "re":
        return True
    if d.get("closePath") and len(items) == 4 and all(it[0] == "l" for it in items):
        pts = []
        for it in items:
            pts.extend([it[1], it[2]])
        near_axis = all(
            abs(a.x - b.x) < 1.0 or abs(a.y - b.y) < 1.0
            for a, b in zip(pts, pts[1:] + pts[:1])
        )
        return near_axis
    return False


def _highlight_boxes(doc: fitz.Document, page: int, img: ImageInfo) -> list[tuple]:
    """Saturated-colour RECTANGULAR OUTLINES stroked over a figure. Requiring a
    real rectangle outline (not a fill or a curve), a strongly saturated stroke
    colour, a sensible size, and no more than a handful per figure separates a
    deliberate highlight box from the illustration's own coloured line art.
    """
    boxes = []
    try:
        drawings = doc[page].get_drawings()
    except Exception:
        return boxes
    fig_rect = fitz.Rect(*img.bbox)
    fig_w = max(1.0, fig_rect.width)
    fig_h = max(1.0, fig_rect.height)
    for d in drawings:
        rect = d.get("rect")
        if rect is None or rect.width < HIGHLIGHT_MIN_SIDE or rect.height < HIGHLIGHT_MIN_SIDE:
            continue
        if rect.width >= fig_w * HIGHLIGHT_MAX_SIDE_FRACTION and rect.height >= fig_h * HIGHLIGHT_MAX_SIDE_FRACTION:
            continue
        inter = fitz.Rect(rect) & fig_rect
        if not inter.is_valid or inter.is_empty:
            continue
        if not _is_rectangle_outline(d):
            continue
        stroke = d.get("color")
        if not stroke or _saturation(stroke) < HIGHLIGHT_MIN_SATURATION:
            continue
        boxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
    if len(boxes) > HIGHLIGHT_MAX_PER_FIGURE:
        return []
    return boxes


def _saturation(color) -> float:
    try:
        r, g, b = float(color[0]), float(color[1]), float(color[2])
    except (TypeError, ValueError, IndexError):
        return 0.0
    high, low = max(r, g, b), min(r, g, b)
    return 0.0 if high <= 0 else (high - low) / high


def _check_highlight_boxes(
    result: CheckResult,
    ctx: _Context,
    exp_figs: list[tuple[int, ImageInfo]],
    counterparts: dict[tuple, tuple[int, ImageInfo] | None],
    heading: str | None,
) -> None:
    """Each PROD highlight box is looked for on the STAGE figure that PROD
    figure was matched to.

    Previously this picked the STAGE figure closest in size and page position,
    because there was nothing better to go on. That guess is now unnecessary -
    and it was wrong often enough to matter, reporting a highlight as missing
    because it had been looked for on a different figure entirely. A figure with
    no counterpart is skipped: whatever happened to it, "Image missing" or
    "Image content differs" already describes it.
    """
    for exp_page, exp_img in exp_figs:
        exp_boxes = _highlight_boxes(ctx.expected, exp_page, exp_img)
        if not exp_boxes:
            continue
        match = counterparts.get(_fig_key(exp_page, exp_img))
        if not match:
            continue
        act_page, act_img = match
        act_boxes = _highlight_boxes(ctx.actual, act_page, act_img)
        for box in exp_boxes:
            if _has_counterpart(box, exp_img, act_boxes, act_img):
                continue
            details = _with_heading(
                {
                    "expected_page": exp_page + 1,
                    "actual_page": act_page + 1,
                    "bbox": box,
                    "highlight_boxes_expected": len(exp_boxes),
                    "highlight_boxes_actual": len(act_boxes),
                },
                heading,
            )
            _attach(ctx, details, exp_page, box, act_page, act_img.bbox)
            result.issues.append(
                Issue(
                    severity="error", page=exp_page, message="Image highlight box missing", details=details
                )
            )


def _has_counterpart(box: tuple, exp_img: ImageInfo, act_boxes: list[tuple], act_img: ImageInfo) -> bool:
    """Compare highlight positions RELATIVE to their own figure, so a figure
    that sits elsewhere on the page doesn't make its highlight look missing.
    """
    rel = ((box[0] + box[2]) / 2 - exp_img.bbox[0], (box[1] + box[3]) / 2 - exp_img.bbox[1])
    for other in act_boxes:
        other_rel = (
            (other[0] + other[2]) / 2 - act_img.bbox[0],
            (other[1] + other[3]) / 2 - act_img.bbox[1],
        )
        if (
            abs(rel[0] - other_rel[0]) <= HIGHLIGHT_MATCH_DISTANCE
            and abs(rel[1] - other_rel[1]) <= HIGHLIGHT_MATCH_DISTANCE
        ):
            return True
    return False


def _caption_words(doc: fitz.Document, page: int, img: ImageInfo) -> set[str]:
    """A figure's CAPTION: the text-layer words in a short block sitting
    directly below (or directly above) the figure and horizontally overlapping
    it.

    Deliberately NOT the words OCR reads from inside the artwork. The interior
    of a UI screenshot - the selected menu value ("Adobe RGB"), the navigation
    hints ("Back  Move  Edit"), a menu's option list - is not a caption: it
    legitimately differs between any two screenshots of a menu, and OCR
    misreads it constantly. Comparing it produced a steady stream of bogus
    "adobe" / "edit" / garbled-word "label missing" findings against figures
    that were fine. A real dropped caption is in the text layer.
    """
    fx0, fy0, fx1, fy1 = img.bbox
    try:
        blocks = doc[page].get_text("blocks")
    except Exception:
        return set()
    caption: list[str] = []
    for b in blocks:
        bx0, by0, bx1, by1, text = b[0], b[1], b[2], b[3], b[4]
        stripped = text.strip()
        if not stripped:
            continue
        if bx1 < fx0 or bx0 > fx1:  # doesn't overlap the figure horizontally
            continue
        below = 0 <= (by0 - fy1) <= CAPTION_GAP
        above = 0 <= (fy0 - by1) <= CAPTION_GAP
        if not (below or above):
            continue
        one_line = " ".join(stripped.split())
        # Reject body text that merely sits next to the figure: a numbered
        # step / bullet / NOTE callout, a run of full sentences, or anything
        # longer than a real caption.
        if _CAPTION_REJECT_LEAD_RE.match(one_line):
            continue
        if one_line.count(". ") >= 1 or (one_line.endswith(".") and len(one_line.split()) > 4):
            continue
        if len(one_line.split()) > CAPTION_STRICT_MAX_WORDS:
            continue
        caption.append(one_line)
    words = " ".join(caption).split()
    if not words or len(words) > CAPTION_MAX_WORDS:
        return set()
    return _label_keys(words)


def _label_keys(raw_words: list[str]) -> set[str]:
    """Normalised, meaningful label words.

    Bare numbers are excluded - those are diagram callouts, checked separately.
    So is anything mixing letters and digits: OCR reading "0~40°C" or "0~2000m"
    off a safety pictogram yields "040c" and "02000m", which are not words and
    which the other document's OCR will misread differently, so comparing them
    only ever produces noise. Stopwords and short fragments go too.
    """
    keys = set()
    for word in raw_words:
        if _BARE_NUMBER_RE.match(word):
            continue
        key = ocr.normalize_word(word)
        if len(key) < MIN_LABEL_WORD_LEN or key in _LABEL_STOPWORDS:
            continue
        if not key.isalpha():
            continue
        keys.add(key)
    return keys


def _section_pages(bounds, page_count: int, spillover: int = 0) -> range:
    if bounds is None:
        return range(0)
    return range(
        max(0, bounds.start_page - spillover),
        min(bounds.end_page + spillover, page_count - 1) + 1,
    )


_FIGURE_OCR_DPI = 350  # higher than the page-level pass - figure labels are small


def _figure_ocr_label_keys(ctx: _Context, page: int, img: ImageInfo) -> tuple[set[str], bool]:
    """(label words OCR reads on/around the Staging figure, whether OCR ran).

    A label Production keeps as live text is often baked into the Staging
    figure as pixels - invisible to the text layer, and often too small for
    the page-level OCR pass. This renders just the figure's area (plus a
    caption-gap margin) at a high DPI and OCRs that, so a pixelated label is
    still read. `ran=False` (OCR unavailable, or it read nothing at all from a
    clearly non-empty figure) tells the caller to stay cautious rather than
    assert a loss it could not verify.
    """
    if not ocr.available():
        return set(), False
    m = CAPTION_GAP + 30.0
    fx0, fy0, fx1, fy1 = img.bbox
    try:
        page_rect = ctx.actual[page].rect
        clip = fitz.Rect(
            max(page_rect.x0, fx0 - m), max(page_rect.y0, fy0 - m),
            min(page_rect.x1, fx1 + m), min(page_rect.y1, fy1 + m),
        )
        pm = ctx.actual[page].get_pixmap(clip=clip, dpi=_FIGURE_OCR_DPI)
        with fitz.open("png", pm.tobytes("png")) as tmp:
            tp = tmp[0].get_textpage_ocr(
                flags=0, dpi=_FIGURE_OCR_DPI, full=True, tessdata=ocr.tessdata_dir()
            )
            words = [w[4] for w in tmp[0].get_text("words", textpage=tp)]
    except Exception:
        return set(), False
    # OCR ran but returned nothing from a figure that plainly has content -
    # treat as "could not verify", not "confirmed empty".
    ran = bool(words) or (fx1 - fx0) * (fy1 - fy0) < 4000
    return _label_keys(words), ran


def _check_labels(
    result: CheckResult, ctx: _Context, pairs: list[_Pair], heading: str | None
) -> None:
    """A figure's caption present in Production but gone from its Staging
    counterpart.

    Compared figure-to-figure - this Production figure's caption against the
    caption of the ONE Staging figure it was confidently matched to - and only
    for figures that HAVE such a match. A figure with no confident counterpart
    is already covered by "Image missing" / "Image content differs" / the
    review list; a "label missing" finding piled on top of that told the reader
    a caption was lost when the whole figure comparison was actually
    unresolved, and - because there was no counterpart to point at - showed the
    Production figure beside an unrelated Staging region, which is exactly the
    "the images are different in prod and stage" problem.
    """
    for pair in pairs:
        if pair.act_img is None or pair.act_page is None:
            continue
        if pair.status not in _CONFIDENT_STATUSES and pair.status != "review":
            continue
        exp_caption = _caption_words(ctx.expected, pair.exp_page, pair.exp_img)
        if not exp_caption:
            continue
        act_caption = _caption_words(ctx.actual, pair.act_page, pair.act_img)
        missing = sorted(exp_caption - act_caption)
        if not missing:
            continue
        # The label is very often baked into the Staging figure as pixels
        # (Production keeps it live text) - OCR the figure area at high DPI
        # before calling it lost.
        ocr_keys, ocr_ran = _figure_ocr_label_keys(ctx, pair.act_page, pair.act_img)
        missing = [w for w in missing if w not in ocr_keys]
        if not missing:
            continue
        # If OCR could not read the Staging figure at all, we cannot tell a
        # dropped label from a rasterised one the OCR simply missed - and in
        # these documents the labels are nearly always rasterised. Don't
        # assert a loss we can't verify.
        if not ocr_ran:
            continue
        is_review = pair.status == "review"
        details = _with_heading(
            {
                "expected_page": pair.exp_page + 1,
                "actual_page": pair.act_page + 1,
                "missing_labels": missing,
                "missing_label_count": len(missing),
                "bbox": pair.exp_img.bbox,
                **({"confidence": "review"} if is_review else {}),
            },
            heading,
        )
        _attach(ctx, details, pair.exp_page, pair.exp_img.bbox, pair.act_page, pair.act_img.bbox)
        result.issues.append(
            Issue(
                severity="warning" if is_review else "error",
                page=pair.exp_page,
                message="Image label missing",
                details=details,
            )
        )


def _page_reference_numbers(words: "ocr.PageWords", page: int) -> set[str]:
    """Numbers that are printed page cross-references ("see page 44"), not
    diagram callouts.

    A cross-reference number sitting in body text beside a figure looks exactly
    like a leader-line callout to a positional test, and its value legitimately
    differs between two documents that paginate differently - so it produced a
    steady stream of "callout 44 is missing" findings about text that was never
    a callout. Any number immediately preceded by the word "page" is excluded.
    """
    numbers: set[str] = set()
    native = words.native_words(page)
    for i, (_, _, _, _, text) in enumerate(native):
        if text.strip().lower().rstrip(":.,") != "page":
            continue
        for follower in native[i + 1 : i + 3]:
            m = _BARE_NUMBER_RE.match(follower[4].strip())
            if m:
                numbers.add(str(int(m.group(1))))
                break
    return numbers


def _figure_callouts(words: "ocr.PageWords", page: int, img: ImageInfo) -> set[str]:
    """The leader-line numbers printed ON a figure (1, 2, 3 ...).

    Collected from strictly INSIDE the figure's own bounds. A margin around the
    figure was tried and swept in the body text beside it, whose page
    cross-references and step numbers are not callouts at all.
    """
    found = ocr.words_in(words.native_words(page), img.bbox) + ocr.words_in(
        words.artwork_words(page), img.bbox
    )
    excluded = _page_reference_numbers(words, page)
    numbers = set()
    for word in found:
        m = _BARE_NUMBER_RE.match(word.strip())
        if m:
            value = str(int(m.group(1)))
            if value not in excluded:
                numbers.add(value)
    return numbers


def _check_callout_numbers(
    result: CheckResult, ctx: _Context, pairs: list[_Pair], heading: str | None
) -> None:
    """A leader-line callout number printed on a Production diagram that its
    Staging counterpart no longer has.

    Compared figure-to-figure - this diagram's numbers against the numbers on
    the ONE Staging figure it was confidently matched to - and only for figures
    that HAVE such a match. The old section-wide-union approach was meant to
    survive repagination, but in practice it compared a Production diagram's
    callouts against every number OCR could find anywhere in the Staging
    section, and pointed the finding at whichever figure happened to be first
    on each side - which is how it came to show two unrelated pictures side by
    side. A diagram with no confident counterpart is already reported by
    "Image missing" / "Image content differs".

    A counterpart on which OCR finds NO numbers at all is skipped: that is the
    two documents' detections not lining up, not the document deleting every
    number - only a partial loss (the counterpart has some numbers but not
    this one) is reported.
    """
    for pair in pairs:
        if pair.act_img is None or pair.act_page is None:
            continue
        if pair.status not in _CONFIDENT_STATUSES and pair.status != "review":
            continue
        exp_numbers = _figure_callouts(ctx.exp_words, pair.exp_page, pair.exp_img)
        if not exp_numbers:
            continue
        act_numbers = _figure_callouts(ctx.act_words, pair.act_page, pair.act_img)

        # Production's diagram carries a real run of numbered callouts (>=4) and
        # the Staging counterpart has NONE of them drawn on it. When those same
        # numbers turn up as a numbered legend list on the Staging page, this is
        # the diagram being deliberately stripped of its leader-line callouts -
        # a real, confirmable change, not the OCR/detection mismatch the plain
        # "no numbers found" case usually is.
        if not act_numbers and len(exp_numbers) >= 4:
            legend = _nearby_legend_numbers(ctx.act_words, pair.act_page, pair.act_img)
            if len(exp_numbers & legend) >= max(4, int(0.7 * len(exp_numbers))):
                details = _with_heading(
                    {
                        "expected_page": pair.exp_page + 1,
                        "actual_page": pair.act_page + 1,
                        "missing_callouts": sorted(exp_numbers, key=int),
                        "missing_callout_count": len(exp_numbers),
                        "reason": (
                            "the Production diagram has these numbers drawn on it with leader "
                            "lines; the Staging diagram has none - the numbers survive only in the "
                            "legend list beside it, so a reader can no longer tell which part is which"
                        ),
                    },
                    heading,
                )
                _attach(ctx, details, pair.exp_page, pair.exp_img.bbox, pair.act_page, pair.act_img.bbox)
                result.issues.append(
                    Issue(
                        severity="error", page=pair.exp_page,
                        message="Diagram callouts stripped from figure", details=details,
                    )
                )
            continue

        missing = sorted(exp_numbers - act_numbers, key=int)
        if not missing or not act_numbers:
            continue

        is_review = pair.status == "review"
        details = _with_heading(
            {
                "expected_page": pair.exp_page + 1,
                "actual_page": pair.act_page + 1,
                "missing_callouts": missing,
                "missing_callout_count": len(missing),
                "expected_callouts": sorted(exp_numbers, key=int),
                "actual_callouts": sorted(act_numbers, key=int),
                **({"confidence": "review"} if is_review else {}),
            },
            heading,
        )
        _attach(ctx, details, pair.exp_page, pair.exp_img.bbox, pair.act_page, pair.act_img.bbox)
        result.issues.append(
            Issue(
                severity="warning" if is_review else "error",
                page=pair.exp_page,
                message="Diagram callout number missing",
                details=details,
            )
        )


# A numbered legend entry for a hardware diagram names a PART - a noun phrase,
# often ending in one of these. A numbered STEP starts with an imperative verb
# ("Press the 5-way controller") and is not a legend.
_PART_NOUNS = re.compile(
    r"\b(port|ports|socket|jack|slot|button|key|keys|indicator|lock|input|output|"
    r"connector|cable|led|hole|cover|hook|clip|switch|dial|wheel)\b",
    re.IGNORECASE,
)
_IMPERATIVE_START = re.compile(
    r"^(press|turn|select|connect|move|push|pull|hold|rotate|adjust|use|see|go|"
    r"tap|click|remove|attach|install|place|set|choose|check|make|ensure|slide)\b",
    re.IGNORECASE,
)


def _nearby_legend_numbers(words: "ocr.PageWords", page: int, img: ImageInfo) -> set[str]:
    """Numbers that head a hardware-part LEGEND entry ("7. USB 3.2 Gen 1
    ports") on the figure's page - the list a diagram's numbered callouts refer
    to. A numbered step list ("1. Press the 5-way controller") is deliberately
    NOT counted, so a page of numbered instructions beside an OSD screenshot
    doesn't read as a stripped callout legend.
    """
    raw = words.native_words(page)
    lines: dict[float, list[tuple]] = {}
    for w in raw:
        lines.setdefault(round(w[1] / 3) * 3, []).append(w)
    numbers: set[str] = set()
    for _, ws in lines.items():
        ws.sort(key=lambda w: w[0])
        text = " ".join(w[4] for w in ws).strip()
        m = re.match(r"^(\d{1,2})[.)]\s+(.{2,80})$", text)
        if not m:
            continue
        entry = m.group(2).strip()
        if _IMPERATIVE_START.match(entry):
            continue
        if _PART_NOUNS.search(entry) or (len(entry.split()) <= 6 and entry[:1].isupper()):
            numbers.add(str(int(m.group(1))))
    return numbers


def _attach(
    ctx: _Context,
    details: dict,
    exp_page: int | None,
    exp_bbox: tuple | None,
    act_page: int | None,
    act_bbox: tuple | None,
) -> None:
    """Render the figure's actual appearance from each document so the finding
    can be judged by eye.

    A side is shown only when it can be shown as a real close-up: no page to
    point at, or a bbox so large the "close-up" would just be most of a page
    (a figure mis-detection), means that side gets no screenshot rather than a
    misleading one. `capture_region` first tries to pull an over-large bbox
    back to its artwork; this is the check on the result.
    """
    if not ctx.output_dir:
        return
    seq = next(ctx.counter)
    if seq > MAX_SCREENSHOTS:
        return

    label = str(seq)  # the same number on both crops - one figure, two views
    details["shot_label"] = label
    if exp_page is not None and _capturable(ctx.expected, exp_page, exp_bbox):
        prod = screenshots.capture_region(
            ctx.expected, exp_page, ctx.output_dir, f"image_{seq}_prod", exp_bbox,
            screenshots.KIND_IMAGE, label,
        )
        if prod:
            details["prod_screenshot"] = prod
    if act_page is not None and _capturable(ctx.actual, act_page, act_bbox):
        stage = screenshots.capture_region(
            ctx.actual, act_page, ctx.output_dir, f"image_{seq}_stage", act_bbox,
            screenshots.KIND_IMAGE, label,
        )
        if stage:
            details["stage_screenshot"] = stage
    if exp_page is not None and "prod_screenshot" not in details:
        details.setdefault(
            "prod_figure",
            "the Production figure could not be isolated from the page for a clean close-up",
        )


# A close-up whose bbox still covers this much of the page after tightening is
# not a close-up - the figure detection merged the artwork with surrounding
# text, and rendering it just shows the page.
_CAPTURE_MAX_PAGE_FRACTION = 0.5
_CAPTURE_MAX_SIDE_FRACTION = 0.9


def _capturable(doc: fitz.Document, page: int, bbox: tuple | None) -> bool:
    if bbox is None:
        return True  # no bbox => whole-page capture is intentional
    tight = screenshots.tighten_bbox(doc, page, bbox)
    try:
        page_rect = doc[page].rect
    except Exception:
        return False
    pw, ph = max(1.0, page_rect.width), max(1.0, page_rect.height)
    w, h = tight[2] - tight[0], tight[3] - tight[1]
    if w <= 0 or h <= 0:
        return False
    return (
        (w * h) / (pw * ph) <= _CAPTURE_MAX_PAGE_FRACTION
        and w / pw <= _CAPTURE_MAX_SIDE_FRACTION
        and h / ph <= _CAPTURE_MAX_SIDE_FRACTION
    )
