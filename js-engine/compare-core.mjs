// What counts as a difference, and what does not.
//
// The rules here are the ones a year of false findings taught the other
// engine, carried over deliberately rather than rediscovered:
//
//   * A line that wraps elsewhere is not a difference. Two documents set to
//     different measures break lines in different places, always.
//   * A page that paginates elsewhere is not a difference either, so pages
//     are paired by their content, never by their number.
//   * A word one side prints more often is not a difference. Only a SENTENCE
//     one side prints and the other does not is missing content.
//   * A sentence printed somewhere else in the other document has moved, not
//     gone - so every sentence is looked for across the whole document.
import { clean } from './extract.mjs';

const WORD = /[^\W_]+(?:['’][^\W_]+)*/gu;
const SENTENCE_SPLIT = /(?<=[.!?:])\s+(?=[A-ZÀ-ɏ])|\s{2,}/;
const MIN_SENTENCE_WORDS = 4;   // shorter runs are labels, not content
const SAME_SENTENCE = 0.82;     // two sentences this alike are the same one

export const words = (text) => (clean(text).match(WORD) || []).map((w) => w.toLowerCase());
export const key = (text) => words(text).join(' ');

export function sentences(blocks) {
  const out = [];
  for (const block of blocks) {
    for (const piece of String(block).split(SENTENCE_SPLIT)) {
      const t = clean(piece);
      if ((t.match(WORD) || []).length >= MIN_SENTENCE_WORDS) out.push(t);
    }
  }
  return out;
}

/** How alike two strings are, 0..1 - the ratio difflib reports. */
export function ratio(a, b) {
  if (a === b) return 1;
  if (!a || !b) return 0;
  const [short, long] = a.length < b.length ? [a, b] : [b, a];
  if (long.includes(short)) return (2 * short.length) / (a.length + b.length);
  // Matching-block count, the same measure SequenceMatcher reports, computed
  // greedily: exact for identical text, close enough to rank near-misses.
  let matches = 0;
  const seen = new Map();
  for (const w of short.split(' ')) seen.set(w, (seen.get(w) || 0) + 1);
  for (const w of long.split(' ')) {
    const n = seen.get(w);
    if (n) { matches += w.length + 1; seen.set(w, n - 1); }
  }
  return (2 * matches) / (a.length + b.length);
}

/**
 * Sentences `mine` prints that `theirs` does not print ANYWHERE.
 *
 * Looked for across the whole of the other document, not just the page
 * opposite: the two paginate differently, and a paragraph that moved is not
 * a paragraph lost.
 */
const GRAM = 6;          // words in the run matched against the other side
const GRAM_ABSENT = 0.6; // this share of a sentence's runs must be absent

/** Every run of GRAM consecutive words in the document, as a Set. */
export function shingles(blocks) {
  const stream = words(blocks.join(' '));
  const set = new Set();
  for (let i = 0; i + GRAM <= stream.length; i++) {
    set.add(stream.slice(i, i + GRAM).join(' '));
  }
  return set;
}

/**
 * Sentences `mine` prints whose WORDS do not appear in `theirs` anywhere.
 *
 * Matched as runs of words, not as whole sentences. The two documents break
 * sentences and paragraphs in different places - one prints "18. This
 * apparatus must be earthed. 19. To avoid damaging..." as one block where the
 * other splits it in two - so whole-sentence matching fails on content that
 * is plainly there. A run of six consecutive words survives that: it is the
 * same run whichever sentence it ends up in.
 */
export function missingSentences(mineBlocks, theirsBlocks) {
  const theirs = shingles(theirsBlocks);
  const out = [];
  for (const sentence of sentences(mineBlocks)) {
    const stream = words(sentence);
    if (stream.length < GRAM) continue;
    let absent = 0;
    let total = 0;
    for (let i = 0; i + GRAM <= stream.length; i++) {
      total++;
      if (!theirs.has(stream.slice(i, i + GRAM).join(' '))) absent++;
    }
    if (total && absent / total >= GRAM_ABSENT) out.push(sentence);
  }
  return out;
}

/**
 * Which page of `b` mirrors each page of `a`.
 *
 * Paired on what the pages SAY. Paired by number instead, a single inserted
 * page puts every later comparison one page out and every finding after it
 * is noise.
 */
export function pairPages(a, b) {
  const pairs = [];
  let cursor = 0;
  for (const page of a.pages) {
    const mine = key(page.text);
    if (!mine) { pairs.push([page.number, null]); continue; }
    let best = null;
    let bestScore = 0;
    // Only ever forward, and never far: both documents run in the same order.
    for (let j = cursor; j < Math.min(b.pages.length, cursor + 12); j++) {
      const score = ratio(mine, key(b.pages[j].text));
      if (score > bestScore) { bestScore = score; best = j; }
    }
    if (best !== null && bestScore >= 0.45) {
      pairs.push([page.number, b.pages[best].number]);
      cursor = best + 1;
    } else {
      pairs.push([page.number, null]);
    }
  }
  return pairs;
}

/** Every difference between two documents, as findings. */
export function compare(prod, stage) {
  const findings = [];
  const pairs = pairPages(prod, stage);

  if (prod.pageCount !== stage.pageCount) {
    findings.push({
      kind: 'page-count', severity: 'medium', prodPage: null, stagePage: null,
      title: 'Page count differs',
      detail: `Production has ${prod.pageCount} pages, Staging has ${stage.pageCount}. ` +
              `Pages are paired by what they say, so this alone is not a content problem.`,
    });
  }

  // Content, whole-document: a sentence in one and nowhere in the other.
  const prodBlocks = prod.pages.flatMap((p) => p.blocks);
  const stageBlocks = stage.pages.flatMap((p) => p.blocks);
  const lost = missingSentences(prodBlocks, stageBlocks);
  const gained = missingSentences(stageBlocks, prodBlocks);
  for (const sentence of lost) {
    findings.push({
      kind: 'text-missing', severity: 'high',
      prodPage: pageOf(prod, sentence), stagePage: null,
      title: 'Printed in Production, nowhere in Staging',
      detail: sentence, sentence,
    });
  }
  for (const sentence of gained) {
    findings.push({
      kind: 'text-extra', severity: 'high',
      prodPage: null, stagePage: pageOf(stage, sentence),
      title: 'Printed in Staging, nowhere in Production',
      detail: sentence, sentence,
    });
  }

  // Bold, page by page: a phrase set bold on one side and plain on the other.
  for (const [pn, sn] of pairs) {
    if (!sn) {
      findings.push({
        kind: 'page-unmatched', severity: 'high', prodPage: pn, stagePage: null,
        title: 'Page has no counterpart in Staging',
        detail: `Nothing in Staging reads like Production page ${pn}.`,
      });
      continue;
    }
    const p = prod.pages[pn - 1];
    const s = stage.pages[sn - 1];
    const pb = new Set(p.bold.map(key).filter(Boolean));
    const sb = new Set(s.bold.map(key).filter(Boolean));
    const lostBold = [...pb].filter((x) => !sb.has(x));
    const gainedBold = [...sb].filter((x) => !pb.has(x));
    if (lostBold.length) {
      findings.push({
        kind: 'bold-missing', severity: 'medium', prodPage: pn, stagePage: sn,
        title: 'Bold in Production, plain in Staging',
        detail: lostBold.slice(0, 6).map((x) => `“${x}”`).join('; ') +
                (lostBold.length > 6 ? ` and ${lostBold.length - 6} more` : ''),
      });
    }
    if (gainedBold.length) {
      findings.push({
        kind: 'bold-added', severity: 'medium', prodPage: pn, stagePage: sn,
        title: 'Plain in Production, bold in Staging',
        detail: gainedBold.slice(0, 6).map((x) => `“${x}”`).join('; ') +
                (gainedBold.length > 6 ? ` and ${gainedBold.length - 6} more` : ''),
      });
    }
  }
  return { findings, pairs };
}

function pageOf(doc, sentence) {
  const k = key(sentence).slice(0, 60);
  for (const page of doc.pages) if (key(page.text).includes(k)) return page.number;
  return null;
}
