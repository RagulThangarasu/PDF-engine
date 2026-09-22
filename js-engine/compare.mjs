#!/usr/bin/env node
// PDF-to-PDF comparison, in JavaScript, on its own.
//
//   node compare.mjs production.pdf staging.pdf [-o report.html]
//
// Nothing here needs Python or a server: pdf.js reads both files, the rules
// in compare-core.mjs decide what counts as a difference, and the result is
// one HTML file that opens anywhere.
import { argv, exit } from 'node:process';
import { readPdf } from './extract.mjs';
import { compare } from './compare-core.mjs';
import { writeReport } from './report.mjs';

function usage(message) {
  if (message) console.error(`\n  ${message}`);
  console.error(`
  PDF comparison engine (JavaScript)

    node compare.mjs <production.pdf> <staging.pdf> [-o report.html] [--json]

    production.pdf   the baseline - what the document should say
    staging.pdf      the candidate - what it now says
    -o, --out        where to write the report (default: pdf-comparison.html)
    --json           also print every finding as JSON
`);
  exit(message ? 1 : 0);
}

const args = argv.slice(2);
if (!args.length || args.includes('-h') || args.includes('--help')) usage();

let out = 'pdf-comparison.html';
const asJson = args.includes('--json');
const files = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '-o' || args[i] === '--out') out = args[++i];
  else if (!args[i].startsWith('-')) files.push(args[i]);
}
if (files.length !== 2) usage('Give exactly two PDFs: the baseline, then the candidate.');

const started = Date.now();
console.log(`reading ${files[0]}`);
const prod = await readPdf(files[0]);
console.log(`reading ${files[1]}`);
const stage = await readPdf(files[1]);
console.log(`comparing ${prod.pageCount} pages against ${stage.pageCount}`);
const result = compare(prod, stage);
await writeReport(out, prod, stage, result);
if (asJson) console.log(JSON.stringify(result.findings, null, 2));
console.log(`${result.findings.length} differences in ${((Date.now() - started) / 1000).toFixed(1)}s`);
console.log(`report: ${out}`);
