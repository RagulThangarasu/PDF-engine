"""Command-line entry point: orchestrates all validation categories."""
from __future__ import annotations

import argparse
import os
import sys

import fitz

from pdfval.extractor import reset_table_cache
from pdfval.models import ValidationReport
from pdfval.report.counterparts import fill_counterpart_screenshots
from pdfval.report.html_report import generate_reports
from pdfval.report.sections import build_section_comparison
from pdfval.validators.headings import reset_heading_cache
from pdfval.verify import verify_report
from pdfval.validators import (
    validate_alignment,
    validate_content,
    validate_images,
    validate_links,
    validate_pages,
    validate_tables,
    validate_toc,
)
from pdfval.validators.toc import build_toc_comparison, build_toc_report, reset_paragraph_cache


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfval",
        description="Compare an expected.pdf against an actual.pdf and produce a validation report.",
    )
    parser.add_argument("expected", help="Path to expected.pdf")
    parser.add_argument("actual", help="Path to actual.pdf")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    return parser


def run(
    expected_path: str,
    actual_path: str,
    output_dir: str,
    progress_cb: "callable | None" = None,
) -> ValidationReport:
    os.makedirs(output_dir, exist_ok=True)

    def _p(pct: float, label: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, label)
            except Exception:
                pass

    _p(2, "Opening PDFs")
    expected = fitz.open(expected_path)
    actual = fitz.open(actual_path)

    report = ValidationReport(expected_path=expected_path, actual_path=actual_path)

    _p(8, "Checking pages")
    report.add(validate_pages(expected, actual, output_dir))
    _p(14, "Checking table of contents")
    report.add(validate_toc(expected, actual))
    report.toc_comparison = build_toc_comparison(expected, actual)
    report.toc_report = build_toc_report(expected, actual)
    _p(22, "Comparing text content")
    content_check, encoding_check = validate_content(expected, actual, expected_path, actual_path, output_dir)
    report.add(content_check)
    report.add(encoding_check)
    _p(48, "Comparing images")
    report.add(validate_images(expected, actual, output_dir))
    _p(62, "Comparing tables")
    report.add(validate_tables(expected, actual, expected_path, actual_path, output_dir))
    _p(70, "Checking alignment")
    report.add(validate_alignment(expected, actual, output_dir))
    _p(74, "Checking hyperlinks")
    report.add(validate_links(expected, actual, output_dir))

    # Second pass: re-check every finding against the full text of both PDFs,
    # drop the ones that are demonstrably false (content that only moved), and
    # tag the rest confirmed / review.
    _p(80, "Verifying findings")
    verify_report(report, expected, actual, expected_path, actual_path)

    # Every finding shows BOTH documents. One that only one side has evidence
    # for gets the other side's view of the same section, so a column is never
    # left blank where the reader most needs the comparison.
    _p(83, "Filling in counterpart screenshots")
    fill_counterpart_screenshots(report, expected, actual, output_dir)

    # Data for sections.html - the standalone TOC-navigated side-by-side
    # content browser. Built here while both documents are still open.
    callout_icon_findings = [
        issue.details or {}
        for check in report.checks
        for issue in check.issues
        if issue.message == "Callout icon missing in Staging"
    ]
    image_findings = [
        {"message": issue.message, "details": issue.details or {}}
        for check in report.checks
        if check.name == "Image Validation"
        for issue in check.issues
    ]
    table_findings = [
        {"message": issue.message, "details": issue.details or {}}
        for check in report.checks
        if check.name == "Table Validation"
        for issue in check.issues
    ]
    format_findings = [
        {"message": issue.message, "details": issue.details or {}}
        for check in report.checks
        if check.name in ("Content Validation", "Encoding Validation")
        for issue in check.issues
        if issue.message in (
            "Bold or italic emphasis removed",
            "Text encoding regression in Staging",
            "Possible text encoding issue",
        )
    ]
    # A list renumbered 1,2,3 -> a,b,c never shows up in the section browser's
    # text diff (the marker is drawn outside the sentence, or stripped from
    # it), so it is carried in explicitly and flagged there in red.
    list_findings = [
        {"message": issue.message, "details": issue.details or {}}
        for check in report.checks
        if check.name == "Alignment Validation"
        for issue in check.issues
        if issue.message == "List marker changed"
    ]
    _p(85, "Building the side-by-side section browser")
    report.section_comparison = build_section_comparison(
        expected, actual, expected_path, actual_path, output_dir,
        callout_icon_findings, image_findings, table_findings, format_findings,
        list_findings,
    )
    _p(95, "Rendering the report")

    expected.close()
    actual.close()
    # The web server runs many comparisons in one process; without this it
    # would keep every uploaded PDF open in pdfplumber for the process's life.
    reset_table_cache()
    reset_paragraph_cache()
    reset_heading_cache()
    from pdfval.ocr import reset_ocr_cache

    reset_ocr_cache()
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.isfile(args.expected):
        print(f"error: expected PDF not found: {args.expected}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.actual):
        print(f"error: actual PDF not found: {args.actual}", file=sys.stderr)
        return 2

    report = run(args.expected, args.actual, args.output)

    paths = generate_reports(report, args.output)

    status = "PASS" if report.passed else "FAIL"
    print(f"Result: {status}")
    print(f"HTML report: {paths['html']}")
    if paths.get("sections"):
        print(f"Side-by-side sections: {paths['sections']}")
    if paths.get("toc"):
        print(f"TOC comparison: {paths['toc']}")
    print(f"JSON report: {paths['json']}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
