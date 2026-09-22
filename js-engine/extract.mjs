// Reading a PDF into the shape the comparison needs.
//
// pdf.js hands back every text run with its position and font. That is enough
// for the questions this engine asks - what words are on the page, which of
// them are set bold, where each one sits - and it is all done in the browser's
// own PDF renderer, so what is read is what a reader would see.
import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

// A word broken across a line end - "war-" / "ranty" - is one word. Two
// documents set to different measures break lines in different places, and a
// line break is never a difference in what the document says.
const WRAP_HYPHEN = /(\w)[-‐­]\s+(?=[a-z])/g;
// Bullet glyphs belong to the template. A list is a list whichever dot draws it.
const BULLETS = /[•●▪◦‣⁃·∙]/g;
// The page number printed at the foot of every page is furniture.
const PAGE_NUMBER = /^\d{1,4}$/;
// A contents listing - "Copyright ......... 2" - belongs to no section.
const LEADERS = /\.{4,}/;

export function clean(text) {
  return (text || '')
    .replace(WRAP_HYPHEN, '$1')
    .replace(BULLETS, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Every page of the PDF as lines, with the font each run is set in. */
export async function readPdf(path) {
  const data = new Uint8Array(await (await import('node:fs/promises')).readFile(path));
  const doc = await getDocument({ data, useSystemFonts: true }).promise;
  const pages = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    // The operator list is what fills commonObjs, and commonObjs is where the
    // REAL font name lives. getTextContent alone reports an internal id
    // ("g_d0_f1"), which says nothing about weight - so bold went unseen.
    await page.getOperatorList();
    const content = await page.getTextContent();
    const lines = groupIntoLines(content.items, page);
    pages.push({
      number: n,
      size: page.view.slice(2),          // [width, height] in points
      lines,
      text: clean(lines.map((l) => l.text).join(' ')),
      blocks: paragraphs(lines),
      bold: lines.flatMap((l) => l.runs.filter((r) => r.bold).map((r) => r.text.trim()))
                 .filter((t) => t.length > 1),
    });
  }
  await doc.destroy();
  return { path, pageCount: doc.numPages, pages };
}

/** Runs on the same baseline are one line - pdf.js gives them item by item. */
// Points of blank space that mean "a different column", not "a wide space".
// A space in running text is about a quarter of the type size; a column gutter
// is several times that.
const COLUMN_GAP = 24;

function runWidth(run) {
  return run.width ?? run.text.length * 4.2;   // pdf.js gives width when it can
}

function makeLine(y, runs) {
  return { y, x: runs[0].x, runs, text: runs.map((r) => r.text).join('') };
}

function groupIntoLines(items, page) {
  const fontName = (id) => {
    try {
      return page.commonObjs.has(id) ? (page.commonObjs.get(id).name || '') : '';
    } catch {
      return '';
    }
  };
  const rows = new Map();
  for (const it of items) {
    if (!it.str || !it.str.trim()) continue;
    const y = Math.round(it.transform[5] * 2) / 2;   // half-point buckets
    const row = rows.get(y) || { y, runs: [] };
    row.runs.push({
      text: it.str,
      x: it.transform[4],
      width: it.width,
      font: fontName(it.fontName),
      // A bold face says so in its own name. The subset prefix AEM and
      // InDesign add ("BALTMQ+Roboto-Bold") does not get in the way.
      bold: /bold|black|heavy|semib/i.test(fontName(it.fontName)),
    });
    rows.set(y, row);
  }
  // Runs sharing a baseline are NOT always one line. A page set in two
  // columns puts the left column and the right column on the same baseline,
  // and joining them makes a sentence neither column contains: "1.Please read
  // this user manual before you6.In some countries, the line voltage is...".
  // A wide horizontal gap is where one column ends and the next begins.
  const out = [];
  for (const row of [...rows.values()].sort((a, b) => b.y - a.y)) {
    row.runs.sort((a, b) => a.x - b.x);
    let segment = [row.runs[0]];
    for (let i = 1; i < row.runs.length; i++) {
      const prev = row.runs[i - 1];
      const gap = row.runs[i].x - (prev.x + runWidth(prev));
      if (gap > COLUMN_GAP) {
        out.push(makeLine(row.y, segment));
        segment = [];
      }
      segment.push(row.runs[i]);
    }
    if (segment.length) out.push(makeLine(row.y, segment));
  }
  return inReadingOrder(out.filter((l) => l.text.trim()));
}

/**
 * Lines gathered into paragraphs, furniture dropped.
 *
 * Sentences are compared later, and a sentence must not run from the end of
 * one paragraph into the heading below it - so the gap between baselines
 * decides where one paragraph ends.
 */
function paragraphs(lines) {
  const out = [];
  let current = [];
  let lastY = null;
  const gaps = [];
  for (let i = 1; i < lines.length; i++) gaps.push(lastY === null ? 0 : 0) , lastY = 0;
  const leading = medianGap(lines);
  for (const line of lines) {
    const text = line.text.trim();
    if (PAGE_NUMBER.test(text) || LEADERS.test(text)) continue;
    const gap = lastY === null ? 0 : lastY - line.y;
    if (lastY !== null && gap > leading * 1.6 && current.length) {
      out.push(clean(current.join(' ')));
      current = [];
    }
    current.push(text);
    lastY = line.y;
  }
  if (current.length) out.push(clean(current.join(' ')));
  return out.filter(Boolean);
}

/**
 * Lines in the order a reader takes them: down one column, then down the next.
 *
 * Sorted by height alone, a two-column page alternates - one line of the left
 * column, one of the right, all the way down - and every paragraph built from
 * consecutive lines mixes the two: "1.Please read this user manual before
 * you 6.In some countries, the line voltage is...". Neither column ever said
 * that, so it matches nothing in the other document and reads as content
 * missing from it.
 */
function inReadingOrder(lines) {
  if (lines.length < 4) return lines;
  const xs = lines.map((l) => l.x).sort((a, b) => a - b);
  const widest = xs[xs.length - 1] - xs[0];
  // One column unless the line starts fall into two clearly separated bands.
  if (widest < COLUMN_GAP * 3) return lines;
  const split = widestGapPoint(xs);
  if (split === null) return lines;
  const left = lines.filter((l) => l.x < split);
  const right = lines.filter((l) => l.x >= split);
  // A band holding almost nothing is an indent or a figure caption, not a
  // column - ordering by it would scatter the page instead of settling it.
  if (left.length < 3 || right.length < 3) return lines;
  return [...left, ...right];
}

function widestGapPoint(xs) {
  let best = 0;
  let at = null;
  for (let i = 1; i < xs.length; i++) {
    const gap = xs[i] - xs[i - 1];
    if (gap > best) { best = gap; at = (xs[i] + xs[i - 1]) / 2; }
  }
  return best >= COLUMN_GAP * 3 ? at : null;
}

function medianGap(lines) {
  const gaps = [];
  for (let i = 1; i < lines.length; i++) {
    const g = lines[i - 1].y - lines[i].y;
    if (g > 0 && g < 100) gaps.push(g);
  }
  if (!gaps.length) return 14;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)] || 14;
}
