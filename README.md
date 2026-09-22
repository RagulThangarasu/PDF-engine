# PDF Validation Engine

Compares a Staging PDF against a Production PDF — Production is the baseline —
and produces one side-by-side report: both documents whole, every difference
boxed where it is with a comment saying what is wrong, and listed in a nav.

```
expected.pdf (Production) ─┐
                           ├──▶ chapter engine ──▶ pdf.html (+ toc.html, report.json) ──▶ PASS/FAIL
actual.pdf (Staging)      ─┘
```

## What is checked

Each top-level chapter is read whole on both sides and compared as printed.
Every issue is filed under one of six categories:

| Category | Reported |
| --- | --- |
| **Content** | Text missing from Staging, extra in Staging, or worded differently. |
| **Images** | Pictures missing, extra, replaced, moved to another section, oversized, aligned differently (left instead of centre), with black spots, or with a label missing. |
| **Hyperlinks** | Links missing or extra in Staging, going somewhere else, or not working. |
| **Lists** | List items that lost or gained their bullet or number, are numbered differently (1., 2. against a., b.), or are indented or aligned differently within their list. |
| **Bold** | Text that prints bold in Production and visibly lighter in Staging. |
| **Tables** | A different number of rows or columns, merged or split cells, or a header row not repeated on a continued page. |

Rules that keep the report to real differences:

- Lines that wrap differently and paragraphs that continue onto the next page
  are joined before comparing; page numbers, running headers and footers,
  printed “see page N” references, tables of contents and Q&A index pages are
  not compared.
- Bold is judged by how dark the text prints on the page, not by the font's
  weight name — text that prints as dark is fine.
- List indents are compared within each list, so a different page size or
  margin is not an issue.
- Black spots are looked for only on a picture that is confidently the same
  picture in both documents; a shifted line is not a spot.

## Setup

```bash
cd pdf-validation-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m pdfval sample/expected.pdf sample/actual.pdf -o output
```

This writes to `output/`:

- `pdf.html` – both documents side by side: every issue listed in the left nav
  by category, and boxed on the pages with a black comment saying what is wrong
  there. The two documents scroll together heading by heading.
- `pdfview/` – the page images `pdf.html` shows
- `toc.html` – the Production vs Staging table-of-contents comparison
- `report.json` – every issue, with its category, as structured data

Place your own `expected.pdf` and `actual.pdf` in `sample/` (or pass any paths).

## Options

```
python -m pdfval <expected.pdf> <actual.pdf> [-o OUTPUT_DIR] [--dpi 150] [--pixel-threshold 0.02]
```

## Web UI

A browser-based UI lets you upload a Production PDF and a Staging PDF, run the
comparison, and open the side-by-side report.

```bash
python -m pdfval.web
```

Then open http://127.0.0.1:5001, upload both PDFs, and click **Run Comparison**.

> Port 5001, not 5000: macOS runs AirPlay Receiver on port 5000, which answers
> HTTP requests with `403 Access denied`. The server refuses to start on a port
> something else already owns, and says so, rather than leaving you with a 403
> from a stranger.
The results page shows a pass/fail summary per check plus download buttons for
the HTML, PDF and JSON reports. Each run is stored under `runs/<run_id>/`.

```
python -m pdfval.web [--host 127.0.0.1] [--port 5001] [--debug]
```

