"""Compare a published PDF against the DITA topics it was built from.

The rest of this engine compares two PDFs. This compares a PDF against its
SOURCE: the topics in an AEM Guides (FMDITA) repository. The question is the
same one a reviewer asks - is everything in the PDF in the topics, and is
everything in the topics in the PDF - but the two sides are not alike, so what
can honestly be compared is stated up front:

  * WORDS can be compared. A topic's text and the PDF section's text are the
    same sentences, and anything printed in one and not the other is a real
    gap, whichever side it is on.
  * STRUCTURE can be compared. A note, a list, a table, a heading level: the
    topic marks these up explicitly, the PDF draws them, and both can be
    counted.
  * STYLING cannot. A topic says "this is a note"; the PDF decides it is a
    blue panel with a pencil icon. A difference there is the publishing
    template's doing, not a content defect, and is not reported.

Nothing here runs unless asked: the module is not imported by the comparison
pipeline, and needs credentials for the author instance to fetch anything.
"""
from __future__ import annotations

import base64
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

REQUEST_TIMEOUT = 30

# DITA elements whose text is CONTENT, and the structure each one stands for.
_NOTE_TAGS = {"note"}
_LIST_TAGS = {"ul", "ol", "sl"}
_TABLE_TAGS = {"table", "simpletable"}
_ROW_TAGS = {"row", "strow"}
_TITLE_TAGS = {"title"}
# Markup that is not the topic's own words: metadata, indexing, draft comments.
_SKIP_TAGS = {"prolog", "metadata", "draft-comment", "required-cleanup",
              "indexterm", "data", "resourceid", "titlealts"}


@dataclass
class Topic:
    """One DITA topic, reduced to what can be compared with a PDF."""

    path: str
    title: str
    text: str
    notes: int = 0
    list_items: int = 0
    tables: list = field(default_factory=list)   # rows per table
    xml: str = ""


class AemClient:
    """Reads a DITA map and its topics from an AEM author instance.

    Credentials are never defaulted or guessed - an author instance answers
    401 without them, and trying likely pairs against someone's server is not
    something a tool should do on its own.
    """

    def __init__(self, base: str, user: str, password: str):
        if not (user and password):
            raise ValueError("AEM author needs a user and password (it answers 401 without them)")
        self.base = base.rstrip("/")
        token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
        self._auth = f"Basic {token}"

    def get(self, path: str) -> bytes:
        url = path if path.startswith("http") else f"{self.base}/{path.lstrip('/')}"
        req = urllib.request.Request(url, headers={"Authorization": self._auth})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return resp.read()

    def topic_paths(self, ditamap: str) -> list[str]:
        """Every topic the map references, in the order the map lists them -
        which is the order the PDF prints them in."""
        raw = self.get(ditamap)
        root = _parse(raw)
        base_dir = ditamap.rsplit("/", 1)[0]
        by_guid = self.guid_index(ditamap)
        out: list[str] = []
        for el in root.iter():
            href = el.get("href")
            if not href or el.get("format") not in (None, "dita", "ditamap"):
                continue
            # A GUID reference resolves through the index; anything else is an
            # ordinary relative or absolute path.
            guid = href.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if guid in by_guid:
                path = by_guid[guid]
            elif href.startswith("/"):
                path = href
            else:
                path = _resolve(base_dir, href)
            if path.endswith((".dita", ".xml")) and path not in out:
                out.append(path)
        return out

    def topic(self, path: str) -> Topic:
        return parse_topic(path, self.get(path))

    def guid_index(self, ditamap: str) -> dict:
        """`{GUID: real path}` for every topic stored beside the map.

        A map written by AEM Guides references its topics by GUID -
        `GUID-2c1b1fe5-....dita` - and no file of that name exists anywhere:
        the GUID is the topic's own `id`, and the file is called something
        else entirely (`VS25_UM_V1.00_EN-5.dita`) in a sibling folder. So the
        topics are listed and indexed by the id each one declares.
        """
        import json as _json

        root = ditamap.rsplit("/", 2)[0]          # .../vs25/Maps/x.ditamap -> .../vs25
        index: dict = {}
        for folder in ("Topics", "topics", "Maps"):
            try:
                listing = _json.loads(self.get(f"{root}/{folder}.1.json"))
            except Exception:
                continue
            for name in listing:
                if not name.endswith((".dita", ".xml")):
                    continue
                path = f"{root}/{folder}/{name}"
                try:
                    guid = _parse(self.get(path)).get("id")
                except Exception:
                    continue
                if guid:
                    index.setdefault(guid, path)
        return index


def _resolve(base_dir: str, href: str) -> str:
    """"../Topics/x.dita" against "/content/dam/.../Maps" -> an absolute path."""
    parts = base_dir.split("/")
    for piece in href.split("?")[0].split("#")[0].split("/"):
        if piece in ("", "."):
            continue
        if piece == "..":
            if parts:
                parts.pop()
        else:
            parts.append(piece)
    return "/".join(parts)


def _parse(raw: bytes) -> ET.Element:
    """Parse DITA, tolerating the DOCTYPE declarations AEM writes."""
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1, flags=re.S)
    return ET.fromstring(text)


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""


def parse_topic(path: str, raw: bytes) -> Topic:
    """A topic's title, its words, and a count of the structures it declares."""
    root = _parse(raw)
    title = ""
    words: list[str] = []
    notes = list_items = 0
    tables: list[int] = []

    def walk(el, in_title=False):
        nonlocal title, notes, list_items
        tag = _tag(el)
        if tag in _SKIP_TAGS:
            return
        if tag in _TITLE_TAGS and not title:
            title = " ".join("".join(el.itertext()).split())
            in_title = True
        if tag in _NOTE_TAGS:
            notes += 1
        if tag == "li" or tag == "sli":
            list_items += 1
        if tag in _TABLE_TAGS:
            tables.append(sum(1 for r in el.iter() if _tag(r) in _ROW_TAGS))
        if el.text and not in_title:
            words.append(el.text)
        for child in el:
            walk(child, in_title and _tag(child) in _TITLE_TAGS)
            if child.tail:
                words.append(child.tail)

    walk(root)
    text = clean(" ".join(words))
    return Topic(path=path, title=title, text=text, notes=notes,
                 list_items=list_items, tables=tables, xml=raw.decode("utf-8", "replace"))


# --- the PDF side ------------------------------------------------------------

def pdf_sections(pdf_path: str) -> list[dict]:
    """Each heading in the PDF with the text printed under it, its notes, its
    list items and its tables - the same four things a topic declares."""
    import fitz

    from pdfval.validators.headings import resolve_entries

    doc = fitz.open(pdf_path)
    entries, _ = resolve_entries(doc, doc)
    # A section runs to the next heading AT ITS OWN LEVEL OR ABOVE, so a
    # chapter contains its sub-headings instead of stopping at the first one.
    # A DITA topic is a whole chapter - compared against a chapter's opening
    # paragraph it looks as if the publish dropped every list and note in it.
    spans = []
    for n, e in enumerate(entries):
        end = next((x for x in entries[n + 1:] if x.level <= e.level), None)
        spans.append((e, (e.page, e.y), (end.page, end.y) if end else (doc.page_count, 1e9)))
    out = []
    for entry, start, stop in spans:
        text, notes, items, tables = [], 0, 0, []
        for page in range(start[0], min(stop[0] + 1, doc.page_count)):
            try:
                blocks = doc[page].get_text("dict").get("blocks", [])
            except Exception:
                continue
            for b in blocks:
                y = b.get("bbox", (0, 0, 0, 0))[1]
                if page == start[0] and y < start[1] - 2:
                    continue
                if page == stop[0] and y > stop[1]:
                    continue
                for line in b.get("lines", []):
                    line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                    if line_text.strip():
                        text.append(line_text)
                        if _MARKER_LINE.match(line_text.strip()):
                            items += 1
        try:
            tables = [t.row_count for p in range(start[0], min(stop[0] + 1, doc.page_count))
                      for t in doc[p].find_tables().tables]
        except Exception:
            tables = []
        # The heading itself and the page number printed at the top of the page
        # are furniture, not the section's words - left in, every section
        # opens with a red mark for its own title.
        lines = [ln for ln in text if not _TOC_LEADER.match(ln.strip())]
        while lines and (lines[0].strip().isdigit()
                         or _key(lines[0]) == _key(entry.title)
                         or _key(lines[0]).startswith(_key(entry.title)[:24])):
            lines.pop(0)
        joined = clean(" ".join(lines))
        notes = len(_NOTE_LABEL.findall(joined))
        out.append({"title": entry.title, "page": entry.page + 1, "text": joined,
                    "level": entry.level, "start": start, "stop": stop,
                    "notes": notes, "list_items": items, "tables": tables})
    return out


_MARKER_LINE = re.compile(r"^(?:[•●▪◦]|\d{1,2}[.)])\s*\S")
_NOTE_LABEL = re.compile(r"\b(NOTE|TIP|WARNING|CAUTION|IMPORTANT)\b\s*:", re.IGNORECASE)
# A word broken across a line end - "war-" / "ranty" - is one word, not two,
# and not a difference from the topic that spells it whole.
_WRAP_HYPHEN = re.compile(r"(\w)-\s+(?=[a-z])")
# Bullet glyphs are the TEMPLATE's: a topic says <li>, the PDF draws a dot.
# Marking every one of them red says nothing about the content.
_BULLETS = re.compile("[\u2022\u25cf\u25aa\u25e6\u2023\u2043]")


# A contents listing - "Copyright......... 2" - belongs to no section. The
# pages carry no bookmark of their own, so they fall inside whichever section
# precedes them and read as content that topic failed to write.
_TOC_LEADER = re.compile(r"^.*\.{4,}.*$", re.MULTILINE)


def clean(text: str) -> str:
    """A PDF section's text as the TOPIC would have written it: wrap hyphens
    closed up, bullet glyphs dropped, whitespace normalised. Both sides are
    cleaned the same way, so the comparison is like for like."""
    text = _WRAP_HYPHEN.sub(r"\1", text or "")
    text = _BULLETS.sub(" ", text)
    return " ".join(text.split())


# --- matching topics to sections --------------------------------------------

import difflib  # noqa: E402

TITLE_MATCH = 0.72   # below this two titles are not the same section
WORDS_SHOWN = 15     # of the words only one side has, before "and N more"
_WORD_RE = re.compile(r"[\wÀ-￿]+", re.UNICODE)
_MIN_WORD = 3        # shorter tokens are markers and noise, not content


def diff_parts(a_text: str, b_text: str) -> tuple[list, list]:
    """Both sides as [(text, differs)] runs, word by word.

    The report shows the two texts side by side in full - a reviewer needs to
    read what is there, not a list of loose words - so the difference has to
    be marked IN them. Everything that matches stays plain; only what is on
    one side and not the other is marked, which is what "highlight the issues"
    means when most of the page is identical.
    """
    a_words, b_words = (a_text or "").split(), (b_text or "").split()
    fold_a = [w.casefold().strip(".,;:()\u201c\u201d\"'") for w in a_words]
    fold_b = [w.casefold().strip(".,;:()\u201c\u201d\"'") for w in b_words]
    a_out: list = []
    b_out: list = []

    def add(bag, words, differs):
        if not words:
            return
        text = " ".join(words)
        if bag and bag[-1][1] == differs:
            bag[-1] = (bag[-1][0] + " " + text, differs)
        else:
            bag.append((text, differs))

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=fold_a, b=fold_b, autojunk=False).get_opcodes():
        add(a_out, a_words[i1:i2], tag != "equal")
        add(b_out, b_words[j1:j2], tag != "equal")
    return a_out, b_out


def _key(title: str) -> str:
    return " ".join((title or "").split()).casefold()


def match(topics: list, sections: list) -> list[tuple]:
    """[(topic, section)] - each topic against the PDF section that answers to
    it, or None when the PDF has no such section (and the other way round)."""
    spare = list(sections)
    pairs: list[tuple] = []
    for topic in topics:
        best, score = None, 0.0
        for sec in spare:
            ratio = difflib.SequenceMatcher(None, _key(topic.title), _key(sec["title"])).ratio()
            if ratio > score:
                best, score = sec, ratio
        if best is not None and score >= TITLE_MATCH:
            spare.remove(best)
            pairs.append((topic, best))
        else:
            pairs.append((topic, None))
    pairs += [(None, sec) for sec in spare]
    return pairs


def _words(text: str):
    from collections import Counter

    return Counter(w.casefold() for w in _WORD_RE.findall(text or "") if len(w) >= _MIN_WORD)


def _only_in(mine: str, theirs: str) -> str:
    a, b = _words(mine), _words(theirs)
    short = [w for w, n in a.items() if n > b.get(w, 0)]
    if not short:
        return ""
    shown = ", ".join(f"“{w}”" for w in short[:WORDS_SHOWN])
    return shown + (f" and {len(short) - WORDS_SHOWN} more" if len(short) > WORDS_SHOWN else "")


def _covers(outer: dict, inner: dict) -> bool:
    """`outer`'s span contains `inner`'s, and it is not the same section."""
    if outer is inner or "start" not in outer or "start" not in inner:
        return False
    return outer["start"] <= inner["start"] and inner["stop"] <= outer["stop"]


def compare(topics: list, sections: list) -> list[dict]:
    """Every difference between the topics and the PDF built from them."""
    out: list[dict] = []
    pairs = match(topics, sections)
    matched = [sec for topic, sec in pairs if topic is not None and sec is not None]
    for topic, sec in pairs:
        if topic is not None and sec is None:
            out.append({"kind": "topic-not-published", "severity": "high",
                        "topic": topic.path, "title": topic.title, "page": None,
                        "detail": (f"The topic “{topic.title}” is in the map but no section of "
                                   f"the PDF answers to it - it was not published, or it was published "
                                   f"under a different heading.")})
            continue
        if topic is None and sec is not None:
            # A heading INSIDE a chapter a topic already covers is part of
            # that topic, not a section nobody wrote - only a chapter with no
            # topic at all is a gap.
            if any(_covers(m, sec) for m in matched):
                continue
            out.append({"kind": "section-not-in-topics", "severity": "high",
                        "topic": None, "title": sec["title"], "page": sec["page"],
                        "detail": (f"The PDF prints “{sec['title']}” on page {sec['page']}, and no "
                                   f"topic in the map carries that heading.")})
            continue
        missing = _only_in(sec["text"], topic.text)
        extra = _only_in(topic.text, sec["text"])
        if missing or extra:
            topic_parts, pdf_parts = diff_parts(topic.text, sec["text"])
            said = []
            if missing:
                said.append(f"in the PDF and not in the topic: {missing}")
            if extra:
                said.append(f"in the topic and not printed: {extra}")
            out.append({
                "kind": "words-differ", "severity": "high",
                "topic": topic.path, "title": sec["title"], "page": sec["page"],
                "detail": "; ".join(said).capitalize() + ".",
                "topic_parts": topic_parts, "pdf_parts": pdf_parts,
            })
        # Structure. The topic DECLARES these; the PDF DRAWS them. A count that
        # disagrees means something was dropped or added between the two, which
        # is a content question - unlike how the template chooses to style them,
        # which is not compared at all.
        for what, a, b in (("note", topic.notes, sec["notes"]),
                           ("list item", topic.list_items, sec["list_items"]),
                           ("table", len(topic.tables), len(sec["tables"]))):
            if a != b:
                out.append({"kind": "structure", "severity": "medium",
                            "topic": topic.path, "title": sec["title"], "page": sec["page"],
                            "detail": (f"The topic declares {a} {what}(s); the PDF section prints "
                                       f"{b}. One of them gained or lost {what}s in publishing.")})
        for n, (rows_a, rows_b) in enumerate(zip(topic.tables, sec["tables"]), start=1):
            if rows_a != rows_b:
                out.append({"kind": "structure", "severity": "medium",
                            "topic": topic.path, "title": sec["title"], "page": sec["page"],
                            "detail": (f"Table {n}: {rows_a} rows in the topic against {rows_b} "
                                       f"printed in the PDF.")})
    return out


# --- screenshots and the report ---------------------------------------------

SHOTS = "shots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pdf_shot(pdf_path: str, page: int, out_dir: str, name: str) -> str | None:
    """The PDF page a finding is on, as an image beside the finding."""
    import fitz

    if not page:
        return None
    try:
        doc = fitz.open(pdf_path)
        if not (1 <= page <= doc.page_count):
            return None
        os.makedirs(os.path.join(out_dir, SHOTS), exist_ok=True)
        rel = f"{SHOTS}/{name}.png"
        doc[page - 1].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(
            os.path.join(out_dir, rel))
        return rel
    except Exception:
        return None


def topic_shot(url: str, out_dir: str, name: str, user: str = "", password: str = "") -> str | None:
    """The topic as the author instance previews it.

    Headless Chrome, which is what is on this machine - no driver to install.
    Credentials go in the URL because that is the only way to hand basic auth
    to a headless screenshot, and they are never written to the report.
    """
    if not os.path.exists(CHROME):
        return None
    import subprocess
    from urllib.parse import quote, urlparse

    target = url
    if user and password:
        bits = urlparse(url)
        target = f"{bits.scheme}://{quote(user, safe='')}:{quote(password, safe='')}@{bits.netloc}{bits.path}"
        if bits.query:
            target += f"?{bits.query}"
    os.makedirs(os.path.join(out_dir, SHOTS), exist_ok=True)
    rel = f"{SHOTS}/{name}.png"
    try:
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--screenshot={os.path.join(out_dir, rel)}", "--window-size=1440,1800",
             "--virtual-time-budget=8000", target],
            capture_output=True, timeout=90, check=False,
        )
    except Exception:
        return None
    return rel if os.path.exists(os.path.join(out_dir, rel)) else None


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def write_report(findings: list[dict], out_dir: str, pdf_path: str,
                 ditamap: str, topics: int, sections: int,
                 shots: dict | None = None) -> str:
    """One self-contained HTML file: every difference, with its screenshot."""
    import html
    from collections import Counter

    os.makedirs(out_dir, exist_ok=True)
    shots = shots or {}
    findings = sorted(findings, key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 3), f.get("page") or 0))
    counts = Counter(f["kind"] for f in findings)
    rows = []
    for n, f in enumerate(findings, start=1):
        shot = shots.get(id(f))
        img = (f'<details class="shot"><summary>PDF page {f.get("page")}</summary>'
               f'<img src="{html.escape(shot)}" alt="PDF page {f.get("page")}"></details>'
               if shot else "")
        # The two sides in full, with ONLY what differs marked. A reviewer has
        # to read what is actually there; a list of loose words out of context
        # cannot be checked against the page.
        # PDF on the LEFT (the published page a reviewer is holding), the
        # ditamap topic on the RIGHT (the source it should have come from).
        left = _marked(f.get("pdf_parts"))
        right = _marked(f.get("topic_parts"))
        pair = ""
        if left or right:
            pair = f"""
          <div class="pair">
            <div class="side"><h4>PDF <span>{f'p.{f["page"]}' if f.get('page') else ''}</span></h4>
              <div class="body">{left or _EMPTY}</div></div>
            <div class="side"><h4>Topic <span>(ditamap)</span></h4>
              <div class="body">{right or _EMPTY}</div></div>
          </div>"""
        rows.append(f"""
        <article class="f {html.escape(f['severity'])}">
          <header><span class="n">{n}</span>
            <span class="kind">{html.escape(_KIND_LABEL.get(f['kind'], f['kind']))}</span>
            <span class="where">{html.escape(f.get('title') or '')}
              {f'&middot; PDF p.{f["page"]}' if f.get('page') else ''}</span></header>
          <p class="detail">{html.escape(f['detail'])}</p>
          {f'<p class="topic">{html.escape(f["topic"])}</p>' if f.get('topic') else ''}
          {pair}{img}
        </article>""")
    summary = " ".join(
        f'<span class="pill">{html.escape(_KIND_LABEL.get(k, k))}: <b>{v}</b></span>'
        for k, v in counts.most_common())
    doc = _REPORT_HTML.format(
        pdf=html.escape(os.path.basename(pdf_path)), ditamap=html.escape(ditamap),
        topics=topics, sections=sections, total=len(findings),
        summary=summary or '<span class="pill ok">No differences found</span>',
        rows="\n".join(rows) or '<p class="ok">Every topic matches the PDF section built from it.</p>')
    path = os.path.join(out_dir, "aem-vs-pdf.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


_EMPTY = '<span class="none">\u2014 nothing here \u2014</span>'


def _marked(parts) -> str:
    """Runs of text, with only the differing ones wrapped in a red mark."""
    import html as _h

    if not parts:
        return ""
    out = []
    for text, differs in parts:
        piece = _h.escape(text)
        out.append(f'<mark>{piece}</mark>' if differs else piece)
    return " ".join(out)


_KIND_LABEL = {
    "words-differ": "Wording differs",
    "topic-not-published": "Topic not published",
    "section-not-in-topics": "In the PDF, not in the topics",
    "words-not-in-topic": "Words missing from the topic",
    "words-not-published": "Words not printed in the PDF",
    "structure": "Structure differs",
}

_REPORT_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Topics vs PDF</title><style>
:root {{ --line:#e6e8ec; --muted:#667085; --high:#c0392b; --med:#8a5a00; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font:14px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif; color:#1a1a1a; background:#f5f6f8 }}
.wrap {{ max-width:1380px; margin:0 auto; padding:22px 22px 120px }}
h1 {{ font-size:20px; margin:0 0 4px }}
.meta {{ color:var(--muted); font-size:12px; margin-bottom:12px }}
.pill {{ display:inline-block; background:#fff; border:1px solid var(--line); border-radius:999px;
        padding:3px 10px; font-size:12px; margin:0 6px 6px 0 }}
.pill.ok {{ border-color:#bfe3c9; color:#1a7a34 }}
.f {{ background:#fff; border:1px solid var(--line); border-left-width:4px; border-radius:8px;
     padding:12px 14px; margin-bottom:14px }}
.f.high {{ border-left-color:var(--high) }} .f.medium {{ border-left-color:var(--med) }}
.f header {{ display:flex; gap:10px; align-items:baseline; margin-bottom:6px; flex-wrap:wrap }}
.n {{ font-weight:700; color:var(--muted) }}
.kind {{ font-weight:700 }}
.where {{ color:var(--muted); font-size:12px }}
.detail {{ margin:0 0 8px }}
.topic {{ color:var(--muted); font-size:11px; font-family:ui-monospace,Menlo,monospace;
         overflow-wrap:anywhere; margin:0 0 8px }}
/* The two sides, side by side. Only what differs is red - everything that
   matches stays plain, or the marking says nothing. */
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }}
@media (max-width:900px) {{ .pair {{ grid-template-columns:1fr }} }}
.side {{ border:1px solid var(--line); border-radius:6px; background:#fcfcfd; min-width:0 }}
.side h4 {{ margin:0; padding:6px 10px; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
           color:var(--muted); border-bottom:1px solid var(--line); background:#f7f8fa;
           border-radius:6px 6px 0 0 }}
.side h4 span {{ font-weight:400; text-transform:none; letter-spacing:0 }}
.side .body {{ padding:9px 11px; max-height:340px; overflow:auto; font-size:13px; overflow-wrap:anywhere }}
mark {{ background:#fde0de; color:#8a1711; font-weight:600; border-radius:2px; padding:0 2px }}
.none {{ color:#aab }}
details.shot {{ margin-top:10px }}
details.shot summary {{ cursor:pointer; color:var(--muted); font-size:12px }}
details.shot img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; margin-top:8px }}
.ok {{ color:#1a7a34 }}
</style></head><body><div class="wrap">
<h1>Topics vs published PDF</h1>
<div class="meta">PDF: <b>{pdf}</b> &middot; map: {ditamap}<br>
{topics} topics compared against {sections} PDF sections &middot; <b>{total}</b> differences</div>
<div>{summary}</div>
<p class="meta">Both sides are shown in full; <mark>only what differs is marked</mark>. Words and
structure are compared - how the publishing template STYLES a note, a list or a heading is not.</p>
{rows}
</div></body></html>"""


# --- command line ------------------------------------------------------------

def run(pdf_path: str, ditamap: str, out_dir: str, base: str = "", user: str = "",
        password: str = "", topics_dir: str = "", preview_shots: bool = False,
        progress=print) -> str:
    """Fetch the topics, compare them to the PDF, write the report. Returns
    the report's path."""
    if topics_dir:
        topics = []
        for name in sorted(os.listdir(topics_dir)):
            if name.endswith((".dita", ".xml")):
                with open(os.path.join(topics_dir, name), "rb") as fh:
                    topics.append(parse_topic(os.path.join(topics_dir, name), fh.read()))
    else:
        client = AemClient(base, user, password)
        progress(f"reading the map: {ditamap}")
        paths = client.topic_paths(ditamap)
        progress(f"{len(paths)} topics referenced")
        topics = []
        for n, path in enumerate(paths, start=1):
            try:
                topics.append(client.topic(path))
            except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError) as exc:
                progress(f"  [{n}/{len(paths)}] could not read {path}: {exc}")
        progress(f"{len(topics)} topics read")
    sections = pdf_sections(pdf_path)
    progress(f"{len(sections)} sections in the PDF")
    findings = compare(topics, sections)
    progress(f"{len(findings)} differences")
    shots = {}
    for n, f in enumerate(findings, start=1):
        rel = pdf_shot(pdf_path, f.get("page"), out_dir, f"pdf_{n}")
        if rel:
            shots[id(f)] = rel
    if preview_shots and base and user:
        for n, topic in enumerate(topics, start=1):
            topic_shot(f"{base.rstrip('/')}{topic.path}", out_dir, f"topic_{n}", user, password)
    path = write_report(findings, out_dir, pdf_path, ditamap, len(topics), len(sections), shots)
    progress(f"report: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="pdfval.aem_compare",
        description="Compare a published PDF against the DITA topics it was built from.")
    p.add_argument("pdf", help="the published PDF")
    p.add_argument("-o", "--output", default="aem-report", help="output directory")
    p.add_argument("--ditamap", default="", help="path to the .ditamap in the repository")
    p.add_argument("--base", default=os.environ.get("AEM_BASE", ""),
                   help="author instance, e.g. http://host:4502 (env AEM_BASE)")
    p.add_argument("--user", default=os.environ.get("AEM_USER", ""), help="AEM user (env AEM_USER)")
    p.add_argument("--password", default=os.environ.get("AEM_PASSWORD", ""),
                   help="AEM password (env AEM_PASSWORD)")
    p.add_argument("--topics-dir", default="",
                   help="compare against .dita files in this folder instead of AEM")
    p.add_argument("--preview-shots", action="store_true",
                   help="also screenshot each topic's preview (needs Chrome)")
    args = p.parse_args(argv)
    if not args.topics_dir and not (args.base and args.user and args.password and args.ditamap):
        p.error("give --topics-dir, or --base --user --password --ditamap "
                "(the author instance answers 401 without credentials)")
    try:
        run(args.pdf, args.ditamap, args.output, args.base, args.user, args.password,
            args.topics_dir, args.preview_shots)
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
