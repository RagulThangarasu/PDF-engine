"""Command-line entry point: compares Staging against Production, the baseline."""
from __future__ import annotations

import argparse
import os
import sys

import fitz

from pdfval.extractor import reset_table_cache
from pdfval.models import ValidationReport
from pdfval.report.html_report import generate_reports
from pdfval.report.issues import build_issue_report
from pdfval.validators.chapter import compare_chapters, reset_furniture_cache, validate_chapters
from pdfval.validators.headings import reset_heading_cache
from pdfval.validators.toc import build_toc_report, reset_paragraph_cache


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
    try:
        _p(6, "Comparing the tables of contents")
        report.toc_report = build_toc_report(expected, actual)

        # One engine decides every issue. It reads each top-level chapter whole
        # on both sides - wrapped lines and paragraphs that carry over a page
        # joined first - and compares content, images, hyperlinks, lists, bold
        # and tables. The per-check validators that ran here before (content,
        # images, tables, alignment, links) reported the same differences in
        # their own words and counts and disagreed with it - hyperlinks passed
        # in one report and failed in another - so they no longer run.
        _p(10, "Comparing chapters")
        chapters = compare_chapters(
            expected, actual, expected_path, actual_path,
            progress_cb=lambda i, n, title: _p(10 + 70 * i / max(1, n), f"Comparing “{title}”"),
        )
        report.add(validate_chapters(expected, actual, expected_path, actual_path, chapters=chapters))

        # Data for pdf.html: both documents rendered whole, every issue boxed on
        # them and listed in the nav - drawn while both documents are still open.
        _p(82, "Rendering both documents")
        report.issue_report = build_issue_report(chapters, expected, actual, output_dir)
        # issues.pdf: every issue with its topic, description and both documents'
        # screenshots boxed - to download and share. A failure here must not
        # lose the run, so it is written best-effort.
        _p(90, "Writing the issues PDF")
        try:
            from pdfval.report.issues_pdf import write_issues_pdf

            write_issues_pdf(report.issue_report, expected, actual, output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: issues.pdf not written: {exc}", file=sys.stderr)
        _p(95, "Writing the report")
    finally:
        expected.close()
        actual.close()
        # The web server runs many comparisons in one process; without this it
        # would keep every uploaded PDF open in pdfplumber for the process's life.
        reset_table_cache()
        reset_paragraph_cache()
        reset_heading_cache()
        reset_furniture_cache()
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
    if paths.get("pdfview"):
        print(f"PDF comparison (side by side, every issue boxed): {paths['pdfview']}")
    if paths.get("toc"):
        print(f"TOC comparison: {paths['toc']}")
    print(f"JSON report: {paths['json']}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
