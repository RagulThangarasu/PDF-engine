"""Shared result data structures used by all validators and the report builder."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Issue:
    """A single discrepancy found by a validator."""

    severity: str  # "error" or "warning"
    page: int | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "page": self.page,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class CheckResult:
    """The outcome of one validation category (e.g. "Page Validation")."""

    name: str
    issues: list[Issue] = field(default_factory=list)
    # Optional per-section breakdown the report renders as a table beneath the
    # issue list - what was compared and how it came out, including the parts
    # that matched. An issue list alone only ever shows what went wrong, which
    # leaves a reader unable to tell a clean section from an unchecked one.
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_title: str = ""

    @property
    def passed(self) -> bool:
        # A "review" issue is a lower-confidence finding the verification pass
        # could not confirm against the full text of both PDFs - it is shown in
        # the report but does not, on its own, fail the check.
        return not any(
            i.severity == "error" and (i.details or {}).get("confidence") != "review"
            for i in self.issues
        )

    @property
    def message_counts(self) -> dict[str, int]:
        """How many issues of each kind (e.g. "Changed text": 35), most
        frequent first, so the report can show a quick breakdown instead of
        just a single total.
        """
        return dict(Counter(i.message for i in self.issues).most_common())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "message_counts": self.message_counts,
            "summary_rows": self.summary_rows,
            "summary_title": self.summary_title,
        }


@dataclass
class ValidationReport:
    """The aggregate result of comparing expected.pdf to actual.pdf."""

    expected_path: str
    actual_path: str
    checks: list[CheckResult] = field(default_factory=list)
    toc_comparison: list[dict[str, Any]] = field(default_factory=list)
    # Data behind chapters.html - each L1 chapter whole, both sides (set by
    # cli.run). Not part of to_dict(): it carries rendered page images, and the
    # JSON report is a record of findings, not of the browser's pictures.
    chapter_comparison: list[dict[str, Any]] = field(default_factory=list)
    # Data behind issues.html - every issue by category, and both documents'
    # page renders (set by cli.run). Not part of to_dict() either.
    issue_report: dict[str, Any] | None = None
    # Data behind pdf.html - both PDFs whole, every finding placed on its
    # page (set by cli.run). Not part of to_dict(), for the same reason.
    pdf_view: dict[str, Any] | None = None
    # Full Prod-vs-Stage bookmark comparison behind toc.html (set by cli.run).
    toc_report: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_path": self.expected_path,
            "actual_path": self.actual_path,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "toc_comparison": self.toc_comparison,
        }
