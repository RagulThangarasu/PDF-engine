"""Data for pdf.html - both PDFs whole, side by side, every finding highlighted.

Every page of Production and Staging is rendered by the PDF engine (PyMuPDF)
into pdfview/, and every finding is placed on those pages as a rectangle in PDF
points. The viewer draws those rectangles as a red background tint over the
text, not a red outline, and positions them as percentages of the page, so they
stay on the text at any zoom.

Line-level text differences come from the section browser's rows. Those carry
the exact line bbox on each side, so the Content Validation text findings
("Changed text", "Added text", ...) are not listed a second time. Every other
check's findings are listed with their own bbox where they have one, and as a
whole-page highlight where they only know the page.
"""
from __future__ import annotations

import os
from typing import Any

import fitz

from pdfval.models import ValidationReport

ZOOM = 1.5
JPEG_QUALITY = 80
SUBDIR = "pdfview"

# Covered line by line (with exact positions) by the section rows.
_TEXT_DIFF_MESSAGES = {"Changed text", "Added text", "Missing text", "Minor text difference"}
_ROW_LABELS = {"change": "Changed text", "prod": "Text in Production only", "stage": "Text in Staging only"}
_SNIPPET = 160


def _render(doc: fitz.Document, output_dir: str, prefix: str) -> list[dict]:
    os.makedirs(os.path.join(output_dir, SUBDIR), exist_ok=True)
    pages = []
    mat = fitz.Matrix(ZOOM, ZOOM)
    for i, page in enumerate(doc):
        name = f"{prefix}_p{i + 1}.jpg"
        page.get_pixmap(matrix=mat, alpha=False).save(
            os.path.join(output_dir, SUBDIR, name), jpg_quality=JPEG_QUALITY
        )
        r = page.rect
        pages.append({"n": i + 1, "src": f"{SUBDIR}/{name}", "w": r.width, "h": r.height})
    return pages


def _spot(page: Any, bbox: Any = None, n_pages: int = 0) -> dict | None:
    """{"page": 1-based, "bbox": [x0,y0,x1,y1] | None}, or None when unusable."""
    if not isinstance(page, int) or not 1 <= page <= n_pages:
        return None
    if bbox is not None:
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            bbox = [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None
        except (TypeError, ValueError):
            bbox = None
    return {"page": page, "bbox": bbox}


def _snip(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= _SNIPPET else text[: _SNIPPET - 1] + "…"


def _row_issues(sections: list[dict], n_exp: int, n_act: int) -> list[dict]:
    from pdfval.report.sections import KIND_CHROME

    out = []
    for sec in sections or []:
        for row in sec.get("rows") or []:
            if row.get("op") == "equal" or row.get("kind") == KIND_CHROME:
                continue
            pl, sl = row.get("prod_loc") or {}, row.get("stage_loc") or {}
            # A side is highlighted only where it actually prints the line; the
            # other side still scrolls to the section so the two stay comparable.
            prod = (_spot((pl.get("page") or 0) + 1, pl.get("bbox"), n_exp)
                    if row.get("prod") and pl.get("bbox") and isinstance(pl.get("page"), int) else None)
            stage = (_spot((sl.get("page") or 0) + 1, sl.get("bbox"), n_act)
                     if row.get("stage") and sl.get("bbox") and isinstance(sl.get("page"), int) else None)
            if not prod and not stage:
                continue
            prod = prod or _spot(sec.get("expected_page"), None, n_exp)
            stage = stage or _spot(sec.get("actual_page"), None, n_act)
            if prod and not (row.get("prod") and prod.get("bbox")):
                prod = {**prod, "context": True}
            if stage and not (row.get("stage") and stage.get("bbox")):
                stage = {**stage, "context": True}
            out.append({
                "group": "Text differences",
                "message": _ROW_LABELS.get(row["op"], "Text differs"),
                "heading": sec.get("heading"),
                "severity": "error",
                "prod_text": _snip(row.get("prod")),
                "stage_text": _snip(row.get("stage")),
                "prod": prod,
                "stage": stage,
            })
    return out


def _bbox_side(message: str, details: dict) -> str:
    """Which document an issue's bbox was measured on."""
    m = message.lower()
    if "margins" in m or "staging only" in m or "only in staging" in m or ("in staging" in m and "missing" not in m):
        return "stage"
    if details.get("expected_page") is None and details.get("actual_page") is not None:
        return "stage"
    return "prod"


def _check_issues(report: ValidationReport, n_exp: int, n_act: int) -> list[dict]:
    out = []
    for check in report.checks:
        for issue in check.issues:
            if check.name == "Content Validation" and issue.message in _TEXT_DIFF_MESSAGES:
                continue
            d = issue.details or {}
            bbox = d.get("bbox")
            side = _bbox_side(issue.message, d)
            exp_page, act_page = d.get("expected_page"), d.get("actual_page")
            if exp_page is None and act_page is None and isinstance(issue.page, int):
                # Only the validator's own 0-based page is known.
                if d.get("side") == "Production":
                    exp_page = issue.page + 1
                elif d.get("side") == "Staging" or issue.message.startswith("Extra page"):
                    act_page = issue.page + 1
                else:
                    exp_page = act_page = issue.page + 1
            if bbox is not None and side == "stage" and act_page is None and isinstance(issue.page, int):
                act_page = issue.page + 1
            prod = _spot(exp_page, bbox if side == "prod" else None, n_exp)
            stage = _spot(act_page, bbox if side == "stage" else None, n_act)
            out.append({
                "group": check.name,
                "message": issue.message,
                "heading": d.get("heading"),
                "severity": issue.severity,
                "confidence": d.get("confidence"),
                "prod": prod,
                "stage": stage,
            })
    return out


def build_pdf_view(
    report: ValidationReport, expected: fitz.Document, actual: fitz.Document, output_dir: str
) -> dict:
    prod_pages = _render(expected, output_dir, "prod")
    stage_pages = _render(actual, output_dir, "stage")
    n_exp, n_act = len(prod_pages), len(stage_pages)
    issues = _row_issues(getattr(report, "section_comparison", None) or [], n_exp, n_act)
    issues += _check_issues(report, n_exp, n_act)
    groups: list[dict] = []
    by_name: dict[str, dict] = {}
    for i, issue in enumerate(issues):
        issue["id"] = i
        g = by_name.get(issue["group"])
        if g is None:
            g = by_name[issue["group"]] = {"name": issue["group"], "ids": []}
            groups.append(g)
        g["ids"].append(i)
    return {"prod_pages": prod_pages, "stage_pages": stage_pages, "issues": issues, "groups": groups}
