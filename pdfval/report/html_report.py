"""Writes report.html, sections.html, toc.html and report.json from a
ValidationReport. report.pdf is no longer produced - see `ALL_FORMATS`."""
from __future__ import annotations

import json
import os
from typing import Any

import fitz
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pdfval.models import ValidationReport

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
PDF_PAGE_SIZE = "letter"


def _group_by_heading(issues: list[dict]) -> list[tuple[str | None, list[dict]]]:
    """Group issues by their details.heading, preserving first-seen order
    (unlike Jinja's built-in groupby, which sorts alphabetically).
    """
    order: list[str | None] = []
    groups: dict[str | None, list[dict]] = {}
    for issue in issues:
        heading = (issue.get("details") or {}).get("heading")
        if heading not in groups:
            groups[heading] = []
            order.append(heading)
        groups[heading].append(issue)
    return [(heading, groups[heading]) for heading in order]


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["group_by_heading"] = _group_by_heading
    return env


def render_report_html(report: ValidationReport, **extra_context: Any) -> str:
    return _env().get_template("report_template.html").render(report=report.to_dict(), **extra_context)


def render_sections_html(report: ValidationReport, sections: list) -> str:
    """The standalone TOC-navigated side-by-side content browser."""
    return _env().get_template("sections_template.html").render(
        sections=sections,
        expected_path=report.expected_path,
        actual_path=report.actual_path,
    )


def write_sections_html(report: ValidationReport, output_dir: str, sections: list) -> str:
    path = os.path.join(output_dir, "sections.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_sections_html(report, sections))
    return path


def render_toc_html(report: ValidationReport, toc: dict) -> str:
    """The standalone Production-vs-Staging table-of-contents comparison."""
    return _env().get_template("toc_template.html").render(
        toc=toc,
        expected_path=report.expected_path,
        actual_path=report.actual_path,
    )


def write_toc_html(report: ValidationReport, output_dir: str, toc: dict) -> str:
    path = os.path.join(output_dir, "toc.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_toc_html(report, toc))
    return path


def write_json_report(report: ValidationReport, output_dir: str) -> str:
    path = os.path.join(output_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    return path


def write_html_report(
    report: ValidationReport, output_dir: str, html: str | None = None, **extra_context: Any
) -> str:
    if html is None:
        html = render_report_html(report, **extra_context)
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def write_pdf_report(report: ValidationReport, output_dir: str, html: str | None = None) -> str:
    """Paginate the report HTML into report.pdf via PyMuPDF's Story/DocumentWriter."""
    if html is None:
        html = render_report_html(report)

    path = os.path.join(output_dir, "report.pdf")
    raw_path = path + ".raw"
    archive = fitz.Archive(output_dir)  # lets <img src="screenshots/..."> resolve
    mediabox = fitz.paper_rect(PDF_PAGE_SIZE)
    story = fitz.Story(html=html, archive=archive)
    writer = fitz.DocumentWriter(raw_path)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(mediabox)
        story.draw(device)
        writer.end_page()
    writer.close()

    # Story/DocumentWriter embeds images uncompressed; recompress before publishing
    # or a report with many screenshots can balloon to hundreds of MB.
    raw_doc = fitz.open(raw_path)
    raw_doc.save(path, garbage=4, deflate=True, clean=True)
    raw_doc.close()
    os.remove(raw_path)

    return path


# report.pdf is deliberately NOT here: nobody reads it. The three HTML reports
# are what anyone opens - the side-by-side section browser above all - and a
# paginated copy of the findings costs every run a few seconds and a ~7 MB file
# (2.9s / 7.4 MB on a 61-page manual) that is strictly worse to read than the
# report.html it was rendered from. `write_pdf_report` is still callable for
# anyone who explicitly asks for `formats=("pdf",)`; nothing does by default.
ALL_FORMATS = ("json", "html", "sections", "toc")


def generate_reports(
    report: ValidationReport,
    output_dir: str,
    formats: tuple[str, ...] = ALL_FORMATS,
    **extra_context: Any,
) -> dict[str, str]:
    """Write the requested report formats; returns a dict of their paths.

    "sections" writes sections.html - the standalone TOC-navigated side-by-side
    content browser - from `report.section_comparison` (set by `cli.run`);
    skipped silently when that isn't present. "toc" writes toc.html the same
    way from `report.toc_report`.
    """
    html = render_report_html(report, **extra_context)
    section_data = getattr(report, "section_comparison", None)
    toc_data = getattr(report, "toc_report", None)
    writers = {
        "json": lambda: write_json_report(report, output_dir),
        "html": lambda: write_html_report(report, output_dir, html=html),
        "pdf": lambda: write_pdf_report(report, output_dir, html=html),
        "sections": lambda: write_sections_html(report, output_dir, section_data or []),
        "toc": lambda: write_toc_html(report, output_dir, toc_data),
    }
    out: dict[str, str] = {}
    for fmt in formats:
        if fmt not in writers:
            continue
        if fmt == "sections" and section_data is None:
            continue
        if fmt == "toc" and toc_data is None:
            continue
        out[fmt] = writers[fmt]()
    return out
