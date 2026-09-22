// The findings as one self-contained HTML file: no server, no assets, opens
// anywhere. Only what differs is marked - everything that matches stays
// plain, or the marking says nothing.
import { writeFile } from 'node:fs/promises';
import { basename } from 'node:path';

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const LABEL = {
  'text-missing': 'Missing from Staging',
  'text-extra': 'Extra in Staging',
  'bold-missing': 'Bold lost in Staging',
  'bold-added': 'Bold added in Staging',
  'page-unmatched': 'Page with no counterpart',
  'page-count': 'Page count differs',
};

export async function writeReport(out, prod, stage, result) {
  const { findings } = result;
  const counts = {};
  for (const f of findings) counts[f.kind] = (counts[f.kind] || 0) + 1;
  const order = { high: 0, medium: 1, low: 2 };
  const sorted = [...findings].sort((a, b) =>
    (order[a.severity] - order[b.severity]) || ((a.prodPage || 0) - (b.prodPage || 0)));

  const pills = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `<span class="pill">${esc(LABEL[k] || k)}: <b>${n}</b></span>`)
    .join(' ');

  const rows = sorted.map((f, i) => `
    <article class="f ${esc(f.severity)}">
      <header><span class="n">${i + 1}</span>
        <span class="kind">${esc(LABEL[f.kind] || f.kind)}</span>
        <span class="where">${f.prodPage ? `Production p.${f.prodPage}` : ''}${
          f.prodPage && f.stagePage ? ' &middot; ' : ''}${
          f.stagePage ? `Staging p.${f.stagePage}` : ''}</span></header>
      <p>${esc(f.detail)}</p>
    </article>`).join('');

  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDF comparison</title><style>
:root{--line:#e6e8ec;--muted:#667085;--high:#c0392b;--med:#8a5a00}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a1a1a;background:#f5f6f8}
.wrap{max-width:1040px;margin:0 auto;padding:22px 22px 120px}
h1{font-size:20px;margin:0 0 4px}
.meta{color:var(--muted);font-size:12px;margin-bottom:12px}
.pill{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;margin:0 6px 6px 0}
.pill.ok{border-color:#bfe3c9;color:#1a7a34}
.f{background:#fff;border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:11px 14px;margin-bottom:10px}
.f.high{border-left-color:var(--high)}.f.medium{border-left-color:var(--med)}
.f header{display:flex;gap:10px;align-items:baseline;margin-bottom:5px;flex-wrap:wrap}
.n{font-weight:700;color:var(--muted)}.kind{font-weight:700}
.where{color:var(--muted);font-size:12px}
.f p{margin:0;overflow-wrap:anywhere}
.ok{color:#1a7a34}
</style></head><body><div class="wrap">
<h1>PDF comparison</h1>
<div class="meta">Production: <b>${esc(basename(prod.path))}</b> (${prod.pageCount} pages)
 &middot; Staging: <b>${esc(basename(stage.path))}</b> (${stage.pageCount} pages)<br>
<b>${findings.length}</b> difference${findings.length === 1 ? '' : 's'}</div>
<div>${pills || '<span class="pill ok">No differences found</span>'}</div>
<p class="meta">Pages are paired by what they say, not by number. A line that wraps
elsewhere, a page that breaks elsewhere and a word printed more often are not
differences; a sentence one document prints and the other never does, is.</p>
${rows || '<p class="ok">The two documents say the same thing.</p>'}
</div></body></html>`;
  await writeFile(out, html, 'utf8');
  return out;
}
