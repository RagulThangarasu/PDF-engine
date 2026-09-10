# PDF Validation Engine

Compares an `expected.pdf` against an `actual.pdf` and produces a PASS/FAIL
HTML report plus a machine-readable JSON report.

```
expected.pdf ─┐
              ├──▶ PDF VALIDATION ENGINE ──▶ HTML report ──▶ PASS/FAIL, JSON
actual.pdf   ─┘
```

## Validation checks

1. **Page validation** – page count, missing pages, extra pages
2. **Content validation** – missing/added/changed text, text position
3. **Image validation** – every Production figure is matched against Staging's
   *same section* by rendered appearance, then compared
4. **Table validation** – table structure, contents, and the mandatory repeated
   header row on continuation pages

### How figures are compared

Each figure is fingerprinted from how it renders on the page (a normalised grey
thumbnail plus a 64-bit structural hash), and matched against the figures in
Staging's same section — not against the figure at the same position, and not
against the same page number. That gives:

| Finding | Meaning |
| --- | --- |
| *(none)* | The same picture was found in the section. **Which page it landed on does not matter** — Staging paginates differently, and a figure pushed onto the next page inside its section is expected, not a defect. The per-section summary counts how many moved. |
| `Image missing` | Nothing in the Staging section resembles it. |
| `Image content differs` | A figure sits in its place, both were measured over comparable areas, and their artwork has nothing in common. |
| `Image could not be confirmed identical` | A figure sits in its place but the two documents detected different extents of it, or the two are only partly alike — flagged for a human rather than asserted as a defect. |
| `Image size changed` | The same picture, drawn at a materially different shape or size. |
| `Image outside its section` | The picture is still in Staging, but now sits past the end of its section. |

Plus broken artwork, missing figure labels, missing diagram callout numbers,
missing highlight boxes, and alignment changes.

### How tables are compared

Beyond row/column/cell comparison (rows matched by an anchor cell, never by
position), a table that runs onto another page **must reprint its header row
there**. That rule is checked against Staging on its own, document-wide — it
holds whether or not the table pairs with a Production table, and whether or not
Production drops the header too.

Both checks also emit a per-section/per-table summary table in the report
showing everything that was compared, including what came through clean.

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

- `report.html` – human-readable report
- `report.pdf` – paginated PDF version of the same report
- `report.json` – full structured results
- `diff_images/` – per-page pixel-diff and visual-diff images

Place your own `expected.pdf` and `actual.pdf` in `sample/` (or pass any paths).

## Options

```
python -m pdfval <expected.pdf> <actual.pdf> [-o OUTPUT_DIR] [--dpi 150] [--pixel-threshold 0.02]
```

## Web UI

A browser-based UI lets you upload a Production PDF and a Staging PDF, run the
comparison, and download the generated HTML/PDF/JSON reports.

```bash
python -m pdfval.web
```

Then open http://127.0.0.1:5000, upload both PDFs, and click **Run Comparison**.
The results page shows a pass/fail summary per check plus download buttons for
the HTML, PDF and JSON reports. Each run is stored under `runs/<run_id>/`.

```
python -m pdfval.web [--host 127.0.0.1] [--port 5000] [--debug]
```

