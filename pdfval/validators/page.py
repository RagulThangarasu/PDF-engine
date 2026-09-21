"""1. Page validation: page count, missing pages, extra pages."""
from __future__ import annotations

import fitz

from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import heading_at


def validate_pages(
    expected: fitz.Document, actual: fitz.Document, output_dir: str | None = None
) -> CheckResult:
    result = CheckResult(name="Page Validation")
    exp_n, act_n = expected.page_count, actual.page_count

    if exp_n != act_n:
        result.issues.append(
            Issue(
                severity="error",
                page=None,
                message=f"Page count mismatch: expected {exp_n}, actual {act_n}",
                details={"expected_count": exp_n, "actual_count": act_n},
            )
        )

    # A page that exists on only one side still gets shown: "Extra page 14" on
    # its own says nothing about WHAT was added, and seeing it is the whole
    # point of the finding. The other column is genuinely empty here - there is
    # no page 14 in the other document to show - so it says so by default. If
    # the page falls inside a heading both documents carry, though, tagging the
    # finding with that heading lets `fill_counterpart_screenshots` replace the
    # placeholder with a real screenshot of that same section in the other
    # document - "there is no page 14" is true but unhelpful when the reader's
    # real question is "what does Staging have here instead?".
    try:
        exp_entries, act_entries = resolve_entries(expected, actual)
    except Exception:
        exp_entries, act_entries = [], []

    def _page_shot(doc: fitz.Document, index: int, side: str, entries: list) -> dict:
        details: dict = {}
        if output_dir:
            path = screenshots.capture_page(doc, index, output_dir, f"page_{side}_{index + 1}")
            if path:
                details[f"{side}_screenshot"] = path
                details[f"{side}_screenshot_caption"] = (
                    f"{'Production' if side == 'prod' else 'Staging'} - p.{index + 1}"
                )
        other, other_name = ("stage", "Staging") if side == "prod" else ("prod", "Production")
        details[f"{other}_screenshot_note"] = f"{other_name} has no page {index + 1}"
        heading = heading_at(entries, index, 0.0)
        if heading:
            details["heading"] = heading
        return details

    # A scanned page has no text layer: its words are pixels, and the content
    # comparison can only reach them through OCR. Say so once per page, so a
    # reader knows the text findings there are OCR readings - and, when OCR is
    # not installed, that the page's text was not compared at all.
    from pdfval import ocr
    ocr_ok = ocr.available()
    for doc, side, name, entries in (
        (expected, "prod", "Production", exp_entries), (actual, "stage", "Staging", act_entries),
    ):
        for i in ocr.scanned_pages(doc):
            details = _page_shot(doc, i, side, entries)
            details.pop("prod_screenshot_note" if side == "stage" else "stage_screenshot_note", None)
            details["side"] = name
            result.issues.append(
                Issue(
                    severity="warning" if ocr_ok else "error", page=i,
                    message=(
                        f"Scanned page {i + 1} in {name} — no text layer; content compared using OCR"
                        if ocr_ok else
                        f"Scanned page {i + 1} in {name} — no text layer and OCR is not installed; "
                        f"its text could not be compared"
                    ),
                    details=details,
                )
            )

    if act_n < exp_n:
        for i in range(act_n, exp_n):
            result.issues.append(
                Issue(
                    severity="error", page=i, message=f"Missing page {i + 1}",
                    details=_page_shot(expected, i, "prod", exp_entries),
                )
            )
    elif act_n > exp_n:
        for i in range(exp_n, act_n):
            result.issues.append(
                Issue(
                    severity="error", page=i, message=f"Extra page {i + 1}",
                    details=_page_shot(actual, i, "stage", act_entries),
                )
            )

    return result

