"""issues.pdf: the issues of a run as a document to download and share.

One section per issue, in the same order and with the same number as pdf.html:

  * the topic it sits under and its title;
  * the description - what differs, in the engine's words - and each change
    listed one per line;
  * below, Production's and Staging's page at that spot, side by side, with the
    issue boxed in the same colour the viewer draws it (red for something
    missing or wrong in Staging, orange for an expected kind of difference to
    confirm).

Built with PyMuPDF straight from the data pdf.html uses, so the two never
disagree. Crops are rendered from the source PDFs, not from the viewer's page
images, and kept as JPEG so a long run stays a reasonable size.
"""
from __future__ import annotations

import os
from html import escape

import fitz

FILENAME = "issues.pdf"

PAGE_W, PAGE_H = 595.0, 842.0          # A4
MARGIN = 36.0
CROP_DPI = 110
CROP_PAD = 40.0                         # points of page around the boxes
CROP_MAX_H = 360.0                      # points of page shown at most
JPEG_QUALITY = 80

RED = (0.86, 0.15, 0.15)
ORANGE = (0.90, 0.51, 0.0)
INK = (0.09, 0.09, 0.11)
MUTED = (0.40, 0.44, 0.50)
RULE = (0.89, 0.90, 0.92)


def _crop_rect(page: fitz.Page, boxes: list[dict]) -> fitz.Rect | None:
    """The part of the page to show: the issue's boxes on its first page, with
    room around them, never taller than CROP_MAX_H."""
    if not boxes:
        return None
    rect = fitz.Rect(boxes[0]["bbox"])
    for b in boxes[1:]:
        r = fitz.Rect(b["bbox"])
        if r.y0 - rect.y1 > CROP_MAX_H:
            break
        rect |= r
    rect = fitz.Rect(page.rect.x0, rect.y0 - CROP_PAD, page.rect.x1, rect.y1 + CROP_PAD) & page.rect
    if rect.height > CROP_MAX_H:
        rect.y1 = rect.y0 + CROP_MAX_H
    return rect


def _shot(doc: fitz.Document, boxes: list[dict], colour: tuple) -> tuple[bytes, fitz.Rect, int] | None:
    """A JPEG of one side's page around the issue, boxes drawn on it."""
    if not boxes:
        return None
    page_no = boxes[0]["page"]
    if not 1 <= page_no <= doc.page_count:
        return None
    on_page = [b for b in boxes if b["page"] == page_no]
    src = fitz.open()
    src.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
    page = src[0]
    for b in on_page:
        r = fitz.Rect(b["bbox"]) + (-2, -2, 2, 2)
        page.draw_rect(r, color=colour, width=1.6)
    clip = _crop_rect(page, on_page)
    pix = page.get_pixmap(clip=clip, dpi=CROP_DPI, alpha=False)
    data = pix.tobytes("jpg", jpg_quality=JPEG_QUALITY)
    src.close()
    return data, clip, page_no


class _Writer:
    """Flows text and images down A4 pages."""

    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page: fitz.Page | None = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN

    def room(self, height: float) -> None:
        if self.y + height > PAGE_H - MARGIN:
            self.new_page()

    def text(self, text: str, size: float = 10, color: tuple = INK, bold: bool = False,
             indent: float = 0.0, gap: float = 3.0) -> None:
        """Wrapped text in the page's flow. Set as HTML so quotes, dashes and
        every script the documents print (Cyrillic, CJK) render - the base-14
        fonts turned “ — › 中 into question marks."""
        if not text:
            return
        rgb = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in color)
        html = (f'<div style="font-family: sans-serif; font-size: {size}pt; color: {rgb}; '
                f'font-weight: {"bold" if bold else "normal"}; line-height: 1.25">{escape(text)}</div>')
        for attempt in (0, 1):
            bottom = PAGE_H - MARGIN
            rect = fitz.Rect(MARGIN + indent, self.y, PAGE_W - MARGIN, bottom)
            if rect.height < size * 2:
                self.new_page()
                continue
            spare, scale = self.page.insert_htmlbox(rect, html, scale_low=1)
            if spare >= 0 and scale >= 1:
                self.y = bottom - spare + gap
                return
            if attempt == 0:
                # Did not fit what was left of the page: the insert drew nothing
                # (scale_low=1 forbids shrinking) - start a new page and retry.
                self.new_page()
        # Longer than a whole page: let it shrink to fit rather than vanish.
        rect = fitz.Rect(MARGIN + indent, self.y, PAGE_W - MARGIN, PAGE_H - MARGIN)
        spare, _ = self.page.insert_htmlbox(rect, html)
        self.y = PAGE_H - MARGIN - max(0.0, spare) + gap

    def rule(self) -> None:
        self.room(10)
        self.page.draw_line((MARGIN, self.y + 4), (PAGE_W - MARGIN, self.y + 4), color=RULE, width=0.8)
        self.y += 12

    def shots(self, left: tuple | None, right: tuple | None, labels: tuple[str, str]) -> None:
        """Production and Staging side by side, scaled to half the width each."""
        col_w = (PAGE_W - 2 * MARGIN - 12) / 2
        scaled = []
        for shot in (left, right):
            if shot is None:
                scaled.append(None)
                continue
            data, clip, page_no = shot
            scale = col_w / max(1.0, clip.width)
            scaled.append((data, clip.height * scale, page_no))
        height = max([s[1] for s in scaled if s] or [40.0])
        self.room(height + 20)
        for k, shot in enumerate(scaled):
            x = MARGIN + k * (col_w + 12)
            label = labels[k] + (f"  ·  p. {shot[2]}" if shot else "  ·  nothing to show")
            self.page.insert_text((x, self.y + 9), label, fontsize=8.5, fontname="hebo", color=MUTED)
            box = fitz.Rect(x, self.y + 14, x + col_w, self.y + 14 + (shot[1] if shot else 40))
            if shot:
                self.page.insert_image(box, stream=shot[0])
            self.page.draw_rect(box, color=RULE, width=0.6)
        self.y += height + 24


def write_issues_pdf(view: dict, expected: fitz.Document, actual: fitz.Document, output_dir: str) -> str:
    """Write issues.pdf into `output_dir` from pdf.html's view data; returns its path."""
    w = _Writer()
    issues = view.get("issues") or []
    w.text("Staging validation — issues", size=18, bold=True, gap=6)
    w.text("Production is the baseline; Staging is checked against it.", size=10, color=MUTED, gap=8)
    counts = "   ".join(f"{c['label']}: {c['count']}" for c in view.get("categories") or [])
    red = sum(1 for it in issues if not it.get("minor"))
    w.text(f"{len(issues)} issues   ·   {red} red (missing or wrong in Staging)   ·   "
           f"{len(issues) - red} orange (to confirm)", size=10.5, bold=True, gap=4)
    w.text(counts, size=9.5, color=MUTED, gap=10)
    w.rule()

    labels = {c["key"]: c["label"] for c in view.get("categories") or []}
    for it in issues:
        colour = ORANGE if it.get("minor") else RED
        w.room(120)
        where = " › ".join(p for p in (it.get("chapter"), it.get("section")) if p)
        w.text(f"#{it['n']}   {labels.get(it['category'], it['category'])}   ·   "
               f"{'Orange' if it.get('minor') else 'Red'}", size=8.5, bold=True, color=colour, gap=1)
        if where:
            w.text(f"Topic: {where}", size=9, color=MUTED, gap=2)
        w.text(it.get("title") or "", size=12, bold=True, gap=3)
        w.text(it.get("description") or "", size=10, gap=4)
        # What differs, one line each - only what the title and description do
        # not already say, each side labelled.
        said = f"{it.get('title') or ''} {it.get('description') or ''}".casefold()
        if it.get("changes"):
            lines = list(dict.fromkeys(it["changes"]))
        else:
            lines = []
            for label, text in (("Production", it.get("comment_prod")), ("Staging", it.get("comment_stage"))):
                core = (text or "").split(": ", 1)[-1].rstrip(".").casefold()
                if text and core and core not in said:
                    lines.append(f"{label}: {text}")
        for k, line in enumerate(lines, 1):
            w.text(f"{k}. {line}", size=9.5, indent=10, gap=1)
        if it.get("occurrences", 1) > 1:
            w.text(f"Same change in {it['occurrences']} places; the first is shown.", size=9, color=MUTED, gap=2)
        w.y += 4
        w.shots(_shot(expected, it.get("prod_boxes") or [], colour),
                _shot(actual, it.get("stage_boxes") or [], colour),
                ("Production", "Staging"))
        w.rule()

    path = os.path.join(output_dir, FILENAME)
    w.doc.save(path, garbage=3, deflate=True)
    w.doc.close()
    return path
