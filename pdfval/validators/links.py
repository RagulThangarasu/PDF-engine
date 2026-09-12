"""6. Other (hyperlinks).

A link is reported ONLY when it is objectively broken - when clicking it in
Staging would not do what the document promises:

* Hyperlink not highlighted in STAGE - a live link drawn as ordinary body text,
  so a reader has no way to know it is clickable
* Internal link target does not resolve - points at a page/destination that
  isn't there
* Hyperlink has no usable scheme - a URI a viewer cannot open
* Hyperlink hotspot cannot be clicked - the clickable rectangle is empty, or
  off the page

Deliberately NOT reported, all confirmed false positives:

* a link present in one document and absent in the other. Whether a phrase is
  linked is an editorial choice, not a defect.
* a link that resolves to a different page or section than in Production. The
  click target is fine; the documents simply paginate differently.
* a stale PRINTED page number in cross-reference text ("see page 46"). The link
  itself works - only the typeset number is out of date, and it is out of date
  by design once pagination changes.
"""
from __future__ import annotations

import itertools

import fitz

from pdfval.models import CheckResult, Issue
from pdfval.report import screenshots
from pdfval.validators.headings import resolve_entries
from pdfval.validators.toc import heading_at, looks_like_toc_listing

MAX_SCREENSHOTS = 100
MIN_HOTSPOT_SIDE = 2.0  # points - a hotspot thinner than this cannot be hit

# Schemes a PDF viewer will actually act on. Anything else (or nothing at all)
# leaves the reader with a link that does nothing when clicked.
USABLE_SCHEMES = ("http://", "https://", "mailto:", "ftp://", "ftps://", "tel:", "file://")

# A link is conventionally signalled by colouring its text away from body black
# and/or underlining it. A span that matches the body text in both respects is
# invisible as a link.
BODY_TEXT_MAX_LUMINANCE = 0.35  # near-black
LINK_COLOR_MIN_DISTANCE = 0.2  # how far from black a colour must be to read as a link


def validate_links(
    expected: fitz.Document, actual: fitz.Document, output_dir: str | None = None
) -> CheckResult:
    result = CheckResult(name="Hyperlink Validation")
    counter = itertools.count(1)
    _, entries = resolve_entries(expected, actual)

    for page_index in range(actual.page_count):
        try:
            links = actual[page_index].get_links()
        except Exception:
            continue
        for link in links:
            _check_link(result, actual, page_index, link, entries, output_dir, counter)

    return result


def _check_link(
    result: CheckResult,
    doc: fitz.Document,
    page_index: int,
    link: dict,
    entries,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    rect = link.get("from")
    bbox = (rect.x0, rect.y0, rect.x1, rect.y1) if rect is not None else None
    heading = heading_at(entries, page_index, bbox[1]) if bbox else None
    kind = link.get("kind")

    def report(message: str, severity: str = "error", **extra) -> None:
        details: dict = {}
        if heading:
            details["heading"] = heading
        details["link_text"] = _link_text(doc, page_index, bbox) or "(no text under the link)"
        details.update(extra)
        if bbox:
            details["bbox"] = bbox
        _attach(details, doc, page_index, bbox, output_dir, counter)
        result.issues.append(Issue(severity=severity, page=page_index, message=message, details=details))

    if rect is None or rect.is_empty or rect.width < MIN_HOTSPOT_SIDE or rect.height < MIN_HOTSPOT_SIDE:
        report(
            "Hyperlink hotspot cannot be clicked",
            reason="the clickable area is empty or too small to hit",
            hotspot=f"{rect.width:.1f} x {rect.height:.1f} pt" if rect is not None else "none",
        )
        return
    if not (fitz.Rect(rect) & doc[page_index].rect).is_valid or (fitz.Rect(rect) & doc[page_index].rect).is_empty:
        report(
            "Hyperlink hotspot cannot be clicked",
            reason="the clickable area lies outside the page",
        )
        return

    if kind == fitz.LINK_URI:
        uri = (link.get("uri") or "").strip()
        if not uri or not uri.lower().startswith(USABLE_SCHEMES):
            report(
                "Hyperlink has no usable scheme",
                uri=uri or "(empty)",
                reason="a viewer has no way to open this target",
            )
            return
    elif kind == fitz.LINK_GOTO:
        target = link.get("page", -1)
        if target is None or target < 0 or target >= doc.page_count:
            report(
                "Internal link target does not resolve",
                target_page=target + 1 if isinstance(target, int) and target >= 0 else "unknown",
                document_pages=doc.page_count,
                reason="the link points at a page that does not exist in this document",
            )
            return
    elif kind == fitz.LINK_NAMED:
        name = link.get("name") or link.get("nameddest") or ""
        if not _named_destination_resolves(doc, name):
            report(
                "Internal link target does not resolve",
                destination=name or "(unnamed)",
                reason="the named destination is not defined in this document",
            )
            return

    # A printed table-of-contents entry ("Copyright ...... 3") is conventionally
    # NOT styled as a link even when every row of it is clickable - the dot
    # leader and the page number are what tell the reader it navigates. Flagging
    # those would fill the report with one finding per TOC row and say nothing.
    if looks_like_toc_listing(_link_text(doc, page_index, bbox)):
        return

    if not _looks_like_a_link(doc, page_index, bbox):
        report(
            "Hyperlink not highlighted in STAGE",
            severity="warning",
            reason="the link works, but its text is styled exactly like body text - "
            "nothing tells the reader it is clickable",
        )


def _named_destination_resolves(doc: fitz.Document, name: str) -> bool:
    if not name:
        return False
    try:
        return name in doc.resolve_names()
    except Exception:
        # Older/odd documents may not expose the name tree; don't invent a
        # finding out of an inability to check.
        return True


def _link_text(doc: fitz.Document, page_index: int, bbox: tuple | None) -> str:
    if not bbox:
        return ""
    try:
        text = doc[page_index].get_textbox(fitz.Rect(*bbox))
    except Exception:
        return ""
    return " ".join(text.split())[:200]


def _looks_like_a_link(doc: fitz.Document, page_index: int, bbox: tuple | None) -> bool:
    """True when the text under the hotspot is visually distinguishable as a
    link: coloured away from body black, or underlined.

    A link whose text is plain black with no underline is indistinguishable
    from surrounding prose - it works, but only for a reader who happens to
    hover over it.
    """
    if not bbox:
        return True
    rect = fitz.Rect(*bbox)
    try:
        data = doc[page_index].get_text("dict", clip=rect)
    except Exception:
        return True

    saw_text = False
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                saw_text = True
                if _is_link_colored(span.get("color", 0)):
                    return True
    if not saw_text:
        return True  # nothing under the hotspot to judge (e.g. a linked image)

    return _has_underline(doc, page_index, rect)


def _is_link_colored(color: int) -> bool:
    """PDF span colours arrive as a packed sRGB integer."""
    r = ((color >> 16) & 0xFF) / 255.0
    g = ((color >> 8) & 0xFF) / 255.0
    b = (color & 0xFF) / 255.0
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > BODY_TEXT_MAX_LUMINANCE:
        return True  # lighter than body text - deliberately coloured
    return max(r, g, b) - min(r, g, b) >= LINK_COLOR_MIN_DISTANCE  # tinted rather than neutral


def _has_underline(doc: fitz.Document, page_index: int, rect: fitz.Rect) -> bool:
    """A thin horizontal rule running under the hotspot's text."""
    try:
        drawings = doc[page_index].get_drawings()
    except Exception:
        return True
    for d in drawings:
        r = d.get("rect")
        if r is None or r.height > 2.5 or r.width < rect.width * 0.5:
            continue
        if rect.x0 - 2 <= r.x1 and r.x0 <= rect.x1 + 2 and rect.y0 - 2 <= r.y0 <= rect.y1 + 4:
            return True
    return False


def _attach(
    details: dict,
    doc: fitz.Document,
    page_index: int,
    bbox: tuple | None,
    output_dir: str | None,
    counter: "itertools.count",
) -> None:
    """Only the Staging document is involved in a link finding, so only the
    Staging column gets a screenshot - there is no Production counterpart to
    show, and filling that column with an unrelated page would mislead.
    """
    if not output_dir:
        return
    seq = next(counter)
    if seq > MAX_SCREENSHOTS:
        return
    path = screenshots.capture_region(
        doc, page_index, output_dir, f"link_{seq}_stage", bbox, screenshots.KIND_DIFF, str(seq)
    )
    if path:
        details["stage_screenshot"] = path
        details["stage_screenshot_caption"] = f"Staging - p.{page_index + 1}"
