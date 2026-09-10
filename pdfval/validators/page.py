"""1. Page validation: page count, missing pages, extra pages."""
from __future__ import annotations

import fitz

from pdfval.models import CheckResult, Issue


def validate_pages(expected: fitz.Document, actual: fitz.Document) -> CheckResult:
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

    if act_n < exp_n:
        for i in range(act_n, exp_n):
            result.issues.append(
                Issue(severity="error", page=i, message=f"Missing page {i + 1}")
            )
    elif act_n > exp_n:
        for i in range(exp_n, act_n):
            result.issues.append(
                Issue(severity="error", page=i, message=f"Extra page {i + 1}")
            )

    return result
