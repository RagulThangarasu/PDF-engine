# PDF comparison engine (JavaScript)

Compares two PDFs and writes one self-contained HTML report. No Python, no
server, no build step.

```bash
cd js-engine
npm install                       # once
node compare.mjs production.pdf staging.pdf -o report.html
```

`production.pdf` is the baseline - what the document should say. `staging.pdf`
is the candidate. Add `--json` to print every finding as JSON as well.

## What it decides is a difference

The rules are the ones the Python engine arrived at, carried over on purpose:

* A line that **wraps** in a different place is not a difference. Two documents
  set to different measures always break lines differently.
* A page that **paginates** differently is not a difference. Pages are paired
  by what they say, never by their number.
* A **word printed more often** on one side is not a difference. Counting word
  frequencies reports "and" 48 times against 42 and calls the rest missing.
* A **sentence one document prints and the other never does** is a difference.
  Matched as runs of six consecutive words, so the two breaking sentences in
  different places does not hide content that is plainly there.
* **Bold** is compared per page pair, read from the font each run is set in.

## Honest limits

The layout reconstruction is weaker than the Python engine's, and that is
where the difference in finding counts comes from:

* **Multi-column pages** are handled but not perfectly. Where two columns share
  a baseline exactly, some lines are still read across the page rather than
  down the column, which invents sentences neither column contains.
* **Tables** are not detected as tables; their cells read as running text.
* **Images, icons, links, shading and list markers** are not compared at all.

Use it for a fast text-and-bold check that runs anywhere. Use the Python
engine when the answer has to be trusted.
