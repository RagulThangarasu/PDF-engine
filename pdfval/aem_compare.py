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
        raw_blocks: list[str] = []
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
                # One BLOCK is one paragraph. Its lines are joined: a sentence
                # wrapped over three lines is one sentence, not three
                # fragments, and a sentence never runs on into the paragraph
                # below it either.
                block_lines = []
                for line in b.get("lines", []):
                    line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                    if line_text.strip():
                        text.append(line_text)
                        block_lines.append(line_text)
                        if _MARKER_LINE.match(line_text.strip()):
                            items += 1
                joined_block = clean(" ".join(block_lines))
                if joined_block and not _TOC_LEADER.match(joined_block) \
                        and not _PAGE_NUMBER.match(joined_block):
                    raw_blocks.append(joined_block)
        try:
            tables = [t.row_count for p in range(start[0], min(stop[0] + 1, doc.page_count))
                      for t in doc[p].find_tables().tables]
        except Exception:
            tables = []
        # The heading itself and the page number printed at the top of the page
        # are furniture, not the section's words - left in, every section
        # opens with a red mark for its own title.
        lines = [ln for ln in text
                 if not _TOC_LEADER.match(ln.strip()) and not _PAGE_NUMBER.match(ln.strip())]
        while lines and (lines[0].strip().isdigit()
                         or _key(lines[0]) == _key(entry.title)
                         or _key(lines[0]).startswith(_key(entry.title)[:24])):
            lines.pop(0)
        joined = clean(" ".join(lines))
        blocks = raw_blocks
        notes = len(_NOTE_LABEL.findall(joined))
        out.append({"title": entry.title, "page": entry.page + 1, "text": joined,
                    "blocks": blocks,
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
# The page number printed at the foot of every page, and the date stamp beside
# it, belong to the template, not to any section.
_PAGE_NUMBER = re.compile(r"^\d{1,3}$|^\d{4}/\d{2}/\d{2}$")


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
# Letters and digits only, plus an inner apostrophe. A quote or bracket
# carried along with a word produced chips like `"the` - two spellings of
# the same word, and a difference that is not one.
_WORD_RE = re.compile(r"[^\W_]+(?:[\u2019'][^\W_]+)*", re.UNICODE)
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


# A difference is a SENTENCE one side prints and the other does not - not a
# word one side prints more often. Counting words said "and" appears 48 times
# in the PDF and 42 in the topic, and called the other six a gap: 108 of 203
# "differences" on one chapter were words BOTH sides print. A reviewer cannot
# act on that, and worse, the real gaps are buried in it.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z\u00c0-\u024f])|\s{2,}")
_MIN_SENTENCE_WORDS = 4     # shorter runs are labels and fragments, not content
_SAME_SENTENCE = 0.80       # two sentences this alike are the same sentence


def sentences(text, blocks: list | None = None) -> list:
    """The text as sentences, short fragments dropped.

    Wrapping, hyphens and bullet glyphs are already closed up by `clean`, so
    a sentence reads the same whichever side it came from. Given `blocks`,
    each one is split on its own - a sentence never runs from the end of a
    paragraph into the heading below it.
    """
    out = []
    source = blocks if blocks else [text]
    for chunk in source:
        out += _split_one(chunk)
    return out


def _split_one(text: str) -> list:
    out = []
    for piece in _SENTENCE_SPLIT.split(clean(text) or ""):
        piece = piece.strip()
        if len(_WORD_RE.findall(piece)) >= _MIN_SENTENCE_WORDS:
            out.append(piece)
    return out


def _sentence_key(text: str) -> str:
    return " ".join(w.casefold() for w in _WORD_RE.findall(text or ""))


def only_in(mine, theirs, mine_blocks=None, theirs_blocks=None) -> list:
    """Sentences `mine` has that `theirs` does not print anywhere.

    Matched against EVERY sentence on the other side, not the one opposite
    it: the two documents order and chunk their content differently, and a
    paragraph that moved is not a paragraph lost.
    """
    theirs_keys = [_sentence_key(x) for x in sentences(theirs, theirs_blocks)]
    theirs_blob = " ".join(theirs_keys)
    out = []
    for sentence in sentences(mine, mine_blocks):
        key = _sentence_key(sentence)
        if not key:
            continue
        if key in theirs_blob:                      # printed verbatim somewhere
            continue
        best = max((difflib.SequenceMatcher(None, key, k).ratio() for k in theirs_keys),
                   default=0.0)
        if best < _SAME_SENTENCE:
            out.append(sentence)
    return out


def _only_in(mine: str, theirs: str) -> str:
    """The same gaps as one sentence, for the finding's headline."""
    found = only_in(mine, theirs)
    if not found:
        return ""
    shown = "; ".join(f"\u201c{_clip(x, 70)}\u201d" for x in found[:3])
    return shown + (f" and {len(found) - 3} more" if len(found) > 3 else "")


def _covers(outer: dict, inner: dict) -> bool:
    """`outer`'s span contains `inner`'s, and it is not the same section."""
    if outer is inner or "start" not in outer or "start" not in inner:
        return False
    return outer["start"] <= inner["start"] and inner["stop"] <= outer["stop"]


_PDF_PATH = [""]   # the PDF the sections came from, for the style comparison


def _all_topic_text(topics: list) -> str:
    """Every topic's words as one blob, for the question "does the map carry
    this anywhere at all?". A paragraph the PDF prints under one heading and
    a topic carries under another has MOVED, not gone - and the two sides
    chunk their chapters differently by design."""
    return " ".join(_sentence_key(t.text) for t in topics)


def compare(topics: list, sections: list, pdf_path: str = "") -> list[dict]:
    """Every difference between the topics and the PDF built from them."""
    _PDF_PATH[0] = pdf_path or _PDF_PATH[0]
    everywhere = _all_topic_text(topics)
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
        blocks = sec.get("blocks")
        missing_words = [x for x in only_in(sec["text"], topic.text, blocks, None)
                         if _sentence_key(x) not in everywhere]
        extra_words = only_in(topic.text, sec["text"], None, blocks)
        missing = "; ".join(f"\u201c{_clip(x, 70)}\u201d" for x in missing_words[:3]) + (
            f" and {len(missing_words) - 3} more" if len(missing_words) > 3 else "")
        extra = "; ".join(f"\u201c{_clip(x, 70)}\u201d" for x in extra_words[:3]) + (
            f" and {len(extra_words) - 3} more" if len(extra_words) > 3 else "")
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
                "missing_words": missing_words, "extra_words": extra_words,
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
        # Styles: what the page renders against what the topic declares. Only
        # where the PDF can actually be read - with no page to look at there
        # is no evidence either way, and every <b> in the topic would read as
        # markup the template dropped.
        if _PDF_PATH[0]:
            try:
                out += compare_roles(topic, sec, _PDF_PATH[0])
            except Exception as exc:  # noqa: BLE001 - never cost the report
                # its other findings, but never hide the reason either: a
                # silent `pass` here hid a NameError through a whole run.
                print(f"  style check skipped for {topic.title}: {type(exc).__name__}: {exc}")
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


HIGHLIGHT_WORDS = 30   # marked on one page before the page is all red anyway
HIGHLIGHT_MIN = 4      # shorter words match everywhere and mark nothing useful


def pdf_shot(pdf_path: str, page: int, out_dir: str, name: str,
             words: list | None = None) -> str | None:
    """The PDF page a finding is on, with the differing words marked ON it.

    A picture of a page says nothing by itself - the reviewer still has to
    hunt through it for the problem. Marking the words this finding is about
    where they are actually printed is the difference between showing a page
    and showing a finding.
    """
    import fitz

    if not page:
        return None
    try:
        doc = fitz.open(pdf_path)
        if not (1 <= page <= doc.page_count):
            return None
        pg = doc[page - 1]
        for word in (words or [])[:HIGHLIGHT_WORDS]:
            if len(word) < HIGHLIGHT_MIN:
                continue
            try:
                for rect in pg.search_for(word)[:8]:
                    ann = pg.add_highlight_annot(rect)
                    ann.set_colors(stroke=(0.99, 0.72, 0.70))
                    ann.update()
            except Exception:
                continue
        os.makedirs(os.path.join(out_dir, SHOTS), exist_ok=True)
        rel = f"{SHOTS}/{name}.png"
        pg.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False).save(
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
                 shots: dict | None = None, topic_list: list | None = None) -> str:
    """One self-contained HTML file, organised the way a reviewer works: topic
    by topic, each with its verdict, every differing word named in full, and
    both sides as pictures."""
    import html
    from collections import Counter, OrderedDict

    os.makedirs(out_dir, exist_ok=True)
    shots = shots or {}
    counts = Counter(f["kind"] for f in findings)

    # Group by topic. A reviewer opens one topic, fixes it, opens the next -
    # findings scattered by severity across fourteen topics cannot be worked.
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for t in topic_list or []:
        groups[t.path] = {"title": t.title, "path": t.path, "findings": [], "topic": t}
    for f in findings:
        key = f.get("topic") or "\u2014 no topic \u2014"
        g = groups.setdefault(key, {"title": f.get("title") or key, "path": f.get("topic"),
                                    "findings": [], "topic": None})
        g["findings"].append(f)

    blocks = []
    for key, g in groups.items():
        fs = g["findings"]
        clean_topic = not fs
        badge = ('<span class="verdict ok">no differences</span>' if clean_topic else
                 f'<span class="verdict bad">{len(fs)} issue{"s" if len(fs) != 1 else ""}</span>')
        tshot = shots.get(("topic", g["path"])) if g["path"] else None
        rows = []
        for n, f in enumerate(fs, start=1):
            rows.append(_finding_html(f, n, shots, html))
        blocks.append(f"""
      <section class="topic-block {'clean' if clean_topic else 'has-issues'}">
        <h2>{html.escape(g['title'] or '')} {badge}</h2>
        {f'<p class="path">{html.escape(g["path"])}</p>' if g.get('path') else ''}
        {"".join(rows) or '<p class="ok">Everything this topic declares is printed in the PDF, and everything the PDF prints here is in the topic.</p>'}
      </section>""")

    summary = " ".join(
        f'<span class="pill">{html.escape(_KIND_LABEL.get(k, k))}: <b>{v}</b></span>'
        for k, v in counts.most_common())
    doc = _REPORT_HTML.format(
        pdf=html.escape(os.path.basename(pdf_path)), ditamap=html.escape(ditamap),
        topics=topics, sections=sections, total=len(findings),
        summary=summary or '<span class="pill ok">No differences found</span>',
        rows="\n".join(blocks) or '<p class="ok">Every topic matches the PDF section built from it.</p>')
    path = os.path.join(out_dir, "aem-vs-pdf.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def _finding_words(f: dict) -> list:
    """The words THIS finding is about - what gets marked on both pictures."""
    words = list(f.get("missing_words") or []) + list(f.get("extra_words") or [])
    for phrase in f.get("phrases") or []:
        words += [w for w in _WORD_RE.findall(phrase) if len(w) >= HIGHLIGHT_MIN]
    seen, out = set(), []
    for w in words:
        if w.casefold() not in seen:
            seen.add(w.casefold())
            out.append(w)
    return out


def _chips(words: list, css: str, label: str, html) -> str:
    """Every word, named. Not a sample - the list is the finding."""
    if not words:
        return ""
    chips = "".join(f'<span class="chip {css}">{html.escape(w)}</span>' for w in words)
    return f'<div class="chips"><span class="chip-label">{label} ({len(words)})</span>{chips}</div>'


def _finding_html(f: dict, n: int, shots: dict, html) -> str:
    shot = shots.get(id(f))
    tshot = shots.get(("topic", id(f)))
    views = ""
    if shot or tshot:
        left = (f'<img src="{html.escape(shot)}" alt="PDF page {f.get("page")}">' if shot else _EMPTY)
        right = (f'<img src="{html.escape(tshot)}" alt="topic render">' if tshot else _EMPTY)
        views = f'''<div class="pair views">
            <div class="side"><h4>PDF <span>{f"p.{f['page']}" if f.get("page") else ""} - as printed</span></h4>
              <div class="shotbody">{left}</div></div>
            <div class="side"><h4>Topic <span>(ditamap) - as its markup declares</span></h4>
              <div class="shotbody">{right}</div></div>
          </div>'''
    # The complete lists, every word named.
    lists = _chips(f.get("missing_words") or [], "pdf", "In the PDF, missing from the topic", html)
    lists += _chips(f.get("extra_words") or [], "topic", "In the topic, not printed in the PDF", html)
    if f.get("phrases"):
        where = "rendered by the PDF, not marked up in the topic" if f.get("in_pdf_only") \
            else f"marked <{f.get('role')}> in the topic, rendered plain by the PDF"
        lists += _chips(f["phrases"], "pdf" if f.get("in_pdf_only") else "topic", where, html)
    left_words = _marked(f.get("pdf_parts"))
    right_words = _marked(f.get("topic_parts"))
    words = ""
    if left_words or right_words:
        words = f'''<details class="words"><summary>read the two texts side by side</summary>
          <div class="pair">
            <div class="side"><h4>PDF <span>{f"p.{f['page']}" if f.get("page") else ""}</span></h4>
              <div class="body">{left_words or _EMPTY}</div></div>
            <div class="side"><h4>Topic <span>(ditamap)</span></h4>
              <div class="body">{right_words or _EMPTY}</div></div>
          </div></details>'''
    return f"""
        <article class="f {html.escape(f['severity'])}">
          <header><span class="n">{n}</span>
            <span class="kind">{html.escape(_KIND_LABEL.get(f['kind'], f['kind']))}</span>
            <span class="where">{html.escape(f.get('title') or '')}
              {f'&middot; PDF p.{f["page"]}' if f.get('page') else ''}</span></header>
          <p class="detail">{html.escape(f['detail'])}</p>
          {lists}{views}{words}
        </article>"""


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
/* Per-topic blocks: a reviewer opens one topic, fixes it, opens the next. */
.topic-block {{ margin:0 0 22px }}
.topic-block h2 {{ font-size:16px; margin:0 0 2px; display:flex; gap:10px; align-items:center }}
.topic-block.clean h2 {{ color:var(--muted) }}
.verdict {{ font-size:11px; font-weight:700; border-radius:999px; padding:2px 9px }}
.verdict.bad {{ background:#fdecec; color:var(--high) }}
.verdict.ok {{ background:#e7f6ec; color:#1a7a34 }}
.path {{ color:var(--muted); font-size:11px; font-family:ui-monospace,Menlo,monospace;
        overflow-wrap:anywhere; margin:0 0 8px }}
/* Every differing word named - the list IS the finding, never a sample. */
.chips {{ margin:6px 0 10px; line-height:2.1 }}
.chip-label {{ font-size:11px; color:var(--muted); text-transform:uppercase;
              letter-spacing:.04em; margin-right:8px }}
.chip {{ display:inline-block; border-radius:4px; padding:1px 7px; margin:0 4px 3px 0;
        font-size:12px; font-weight:600 }}
.chip.pdf {{ background:#fde0de; color:#8a1711 }}
.chip.topic {{ background:#fff0d6; color:#7a4a00 }}
.views {{ margin:10px 0 }}
.shotbody {{ padding:8px; background:#fff; max-height:520px; overflow:auto }}
.shotbody img {{ width:100%; border:1px solid var(--line); border-radius:4px; display:block }}
details.words {{ margin-top:8px }}
details.words summary {{ cursor:pointer; color:var(--muted); font-size:12px; margin-bottom:6px }}
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
    findings = compare(topics, sections, pdf_path)
    progress(f"{len(findings)} differences")
    # Each finding's OWN words are marked on both of its pictures, so the two
    # point at the same thing. A page picture and a topic picture that merely
    # sit beside each other leave the reviewer to find the problem twice.
    shots = {}
    by_path = {t.path: t for t in topics}
    progress("drawing each finding on both sides")
    for n, f in enumerate(findings, start=1):
        words = _finding_words(f)
        rel = pdf_shot(pdf_path, f.get("page"), out_dir, f"pdf_{n}", words)
        if rel:
            shots[id(f)] = rel
        topic = by_path.get(f.get("topic"))
        if topic is not None:
            rel = topic_render_shot(topic, out_dir, f"topic_{n}", words)
            if rel:
                shots[("topic", id(f))] = rel
    path = write_report(findings, out_dir, pdf_path, ditamap, len(topics), len(sections),
                        shots, topic_list=topics)
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




# --- styles: what the PDF RENDERS against what the topic DECLARES -----------
#
# Comparing two PDFs means comparing inferred against inferred - both sides are
# pixels, so bold is measured against bold and a note against a note. A topic
# is not pixels. It DECLARES what a thing is (<b>, <note>, <xref>, <image>,
# <li>) and says nothing about how it should look; the PDF is the evidence of
# how the template rendered that declaration. So the question is not "do the
# two look alike" but:
#
#     is every style the PDF renders justified by markup in the topic,
#     and does every markup in the topic show up styled in the PDF?
#
# A phrase bold on the page and plain in the topic is styling with no source
# behind it - it will vanish the next time the manual is rebuilt. A phrase
# <b> in the topic and plain on the page is markup the template dropped. Both
# are defects, and neither is visible to a word-by-word comparison, which sees
# the same words on both sides and passes.
_ROLE_TAGS = {
    "bold": ("b", "uicontrol", "cmdname", "wintitle"),
    "note": ("note",),
    "link": ("xref",),
    "image": ("image",),
    "list item": ("li", "sli"),
}
_ROLE_MIN_CHARS = 2        # a one-character run is punctuation, not a phrase
_ROLE_SHOWN = 8            # phrases named in a finding before "and N more"


def topic_roles(topic_xml: str) -> dict:
    """`{role: [phrase, ...]}` - what the topic DECLARES, by markup."""
    out: dict = {r: [] for r in _ROLE_TAGS}
    try:
        root = _parse(topic_xml.encode("utf-8") if isinstance(topic_xml, str) else topic_xml)
    except Exception:
        return out
    for el in root.iter():
        tag = _tag(el)
        for role, tags in _ROLE_TAGS.items():
            if tag not in tags:
                continue
            if role == "image":
                href = el.get("href") or ""
                out[role].append(href.rsplit("/", 1)[-1])
            else:
                phrase = " ".join("".join(el.itertext()).split())
                if len(phrase) >= _ROLE_MIN_CHARS:
                    out[role].append(phrase)
    return out


def pdf_roles(pdf_path: str, section: dict) -> dict:
    """`{role: [phrase, ...]}` - what the PDF section RENDERS.

    Read the way the rest of the engine reads a page: bold from the font the
    span is actually set in, notes from the callout label, links from the
    page's own annotations, pictures from the artwork regions. Nothing here
    trusts the topic - that is the whole point of comparing them.
    """
    import fitz

    from pdfval.validators.inventory import page_inventory

    doc = fitz.open(pdf_path)
    start, stop = section.get("start"), section.get("stop")
    out: dict = {r: [] for r in _ROLE_TAGS}
    if not (start and stop):
        return out
    for page in range(start[0], min(stop[0] + 1, doc.page_count)):
        try:
            blocks = doc[page].get_text("dict").get("blocks", [])
        except Exception:
            continue
        for b in blocks:
            y = b.get("bbox", (0, 0, 0, 0))[1]
            if (page == start[0] and y < start[1] - 2) or (page == stop[0] and y > stop[1]):
                continue
            for line in b.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans)
                if _MARKER_LINE.match(text.strip()):
                    out["list item"].append(clean(text))
                if _NOTE_LABEL.search(text):
                    out["note"].append(clean(text))
                # Bold runs, joined where they sit next to each other: a
                # phrase set bold across two spans is one phrase.
                run: list[str] = []
                for s in spans:
                    if "bold" in (s.get("font") or "").casefold():
                        run.append(s.get("text", ""))
                    elif run:
                        _add_run(out["bold"], run)
                        run = []
                _add_run(out["bold"], run)
        try:
            out["link"] += [(l.get("uri") or "internal") for l in doc[page].get_links()
                            if l.get("kind") != 0]
            out["image"] += [f"{round(r[0])}x{round(r[1])}"
                             for r in page_inventory(doc, page)["images"]]
        except Exception:
            pass
    return out


def _add_run(bag: list, run: list) -> None:
    phrase = " ".join("".join(run).split())
    if len(phrase) >= _ROLE_MIN_CHARS:
        bag.append(phrase)


def _phrase_key(text: str) -> str:
    """Two phrases are the same phrase when their words are."""
    return " ".join(_WORD_RE.findall((text or "").casefold()))


def compare_roles(topic, section: dict, pdf_path: str) -> list[dict]:
    """Every style the PDF renders without markup behind it, and every markup
    the topic declares that the PDF did not render."""
    declared = topic_roles(topic.xml)
    rendered = pdf_roles(pdf_path, section)
    out: list[dict] = []
    for role in ("bold", "note", "link"):
        # Phrases, matched on their words. An <image> is matched by count
        # instead (a GUID filename and a rendered size share nothing), and a
        # list item's text is already compared word for word by the wording
        # check - here only its COUNT says whether the marker survived.
        d = {_phrase_key(p): p for p in declared[role] if _phrase_key(p)}
        r = {_phrase_key(p): p for p in rendered[role] if _phrase_key(p)}
        lost = [d[k] for k in d.keys() - r.keys()]
        extra = [r[k] for k in r.keys() - d.keys()]
        if lost:
            out.append(_role_finding(topic, section, role, lost, published=False))
        if extra:
            out.append(_role_finding(topic, section, role, extra, published=True))
    for role in ("image", "list item"):
        a, b = len(declared[role]), len(rendered[role])
        if a != b:
            out.append({
                "kind": "style-count", "severity": "medium",
                "topic": topic.path, "title": section["title"], "page": section["page"],
                "detail": (f"The topic declares {a} {role}(s); the PDF section renders {b}. "
                           f"{'The template dropped some' if a > b else 'The page shows more than the topic declares'} - "
                           f"check which."),
                "topic_parts": [], "pdf_parts": [],
            })
    return out


def _role_finding(topic, section: dict, role: str, phrases: list, published: bool) -> dict:
    shown = "; ".join(f"“{_clip(p)}”" for p in phrases[:_ROLE_SHOWN])
    more = f" and {len(phrases) - _ROLE_SHOWN} more" if len(phrases) > _ROLE_SHOWN else ""
    if published:
        detail = (f"The PDF renders these as {role} with no <{_ROLE_TAGS[role][0]}> behind them in the "
                  f"topic: {shown}{more}. Styling with no source is lost the next time the manual is "
                  f"rebuilt from these topics.")
    else:
        detail = (f"The topic marks these <{_ROLE_TAGS[role][0]}> and the PDF renders them plain: "
                  f"{shown}{more}. The template did not carry the markup through.")
    return {
        "kind": f"style-{role}", "severity": "high" if role != "link" else "medium",
        "topic": topic.path, "title": section["title"], "page": section["page"],
        "detail": detail, "topic_parts": [], "pdf_parts": [],
        "phrases": list(phrases), "role": role, "in_pdf_only": published,
    }


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_KIND_LABEL.update({
    "style-bold": "Bold differs from the topic's markup",
    "style-note": "Note differs from the topic's markup",
    "style-link": "Link differs from the topic's markup",
    "style-count": "Styled element count differs",
})


# --- seeing both sides ------------------------------------------------------
#
# A list of differences in words is not enough to review a page by: a reviewer
# has to SEE what is on it. So each finding carries two pictures - the PDF page
# as it prints, and the topic rendered from its own markup - side by side.
#
# The topic render is NOT a screenshot of the AEM editor. That editor needs a
# session login, and what it shows is mostly its own panels and toolbars. This
# renders the markup itself: every <title>, <note>, <b>, <ul>, <table> drawn as
# what it declares itself to be. Images resolve by UUID inside AEM and cannot
# be fetched by path, so each one is drawn as a labelled placeholder - the
# topic says a picture belongs there, and that is what can honestly be shown.

_TOPIC_CSS = """
body { margin:0; font:14px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif; color:#1a1a1a;
       background:#fff; padding:26px 30px; }
h1 { font:600 23px/1.3 Poppins,Segoe UI,Arial,sans-serif; color:#4b2e83; margin:0 0 14px }
h2 { font:600 18px/1.3 Poppins,Segoe UI,Arial,sans-serif; color:#4b2e83; margin:20px 0 8px }
h3 { font:600 15px/1.3 Poppins,Segoe UI,Arial,sans-serif; color:#4b2e83; margin:16px 0 6px }
p { margin:0 0 10px }
b, .uicontrol { font-weight:700 }
.note { border:1px solid #cdddf7; background:#eef4fe; border-radius:6px; padding:9px 12px 9px 40px;
        margin:10px 0; position:relative; font-size:13px }
.note::before { content:"\270e"; position:absolute; left:12px; top:8px; width:20px; height:20px;
        border-radius:50%; background:#4b2e83; color:#fff; text-align:center; line-height:20px;
        font-size:12px }
.note.warning::before { content:"!"; background:#a01515 } .note.warning { border-color:#e8a3a3; background:#fdf0f0 }
ul, ol { margin:8px 0 12px 22px; padding:0 } li { margin:0 0 5px }
a, .xref { color:#2d6cdf; text-decoration:underline }
table { border-collapse:collapse; margin:10px 0; font-size:13px; width:100% }
td, th { border:1px solid #d7dbe0; padding:5px 8px; text-align:left; vertical-align:top }
th, .thead td { background:#efefef; font-weight:600 }
.img { border:1px dashed #b9c0cc; border-radius:6px; background:#fafbfc; color:#7a8595;
       padding:14px; margin:10px 0; text-align:center; font-size:11px;
       font-family:ui-monospace,Menlo,monospace; overflow-wrap:anywhere }
"""

_BLOCK_AS = {"p": "p", "li": "li", "ul": "ul", "ol": "ol", "sl": "ul", "sli": "li",
             "table": "table", "simpletable": "table", "row": "tr", "strow": "tr",
             "entry": "td", "stentry": "td", "thead": "tbody", "tbody": "tbody",
             "b": "b", "uicontrol": "b", "cmdname": "b", "wintitle": "b",
             "xref": "span", "i": "i", "codeph": "code"}


def render_topic_html(topic, words: list | None = None) -> str:
    """The topic drawn as what its markup declares it to be, with the words a
    finding is about marked - the same marking the PDF picture carries, so
    the two pictures point at the same thing."""
    import html as _h

    try:
        root = _parse(topic.xml.encode("utf-8"))
    except Exception:
        return f"<!doctype html><meta charset=utf-8><style>{_TOPIC_CSS}</style><p>(topic could not be parsed)</p>"
    depth = [0]

    def walk(el) -> str:
        tag = _tag(el)
        if tag in _SKIP_TAGS:
            return ""
        inner = _h.escape(el.text or "")
        for child in el:
            inner += walk(child) + _h.escape(child.tail or "")
        if tag == "title":
            depth[0] += 1
            return f"<h{min(depth[0], 3)}>{inner}</h{min(depth[0], 3)}>"
        if tag == "note":
            kind = (el.get("type") or "note").lower()
            css = "note warning" if kind in ("warning", "caution", "danger", "important") else "note"
            return f'<div class="{css}"><b>{kind.upper()}:</b> {inner}</div>'
        if tag == "image":
            href = (el.get("href") or "").rsplit("/", 1)[-1]
            return f'<div class="img">&#128247; image declared by the topic<br>{_h.escape(href)}</div>'
        if tag == "xref":
            return f'<span class="xref">{inner}</span>'
        as_tag = _BLOCK_AS.get(tag)
        if as_tag:
            return f"<{as_tag}>{inner}</{as_tag}>"
        return inner

    body = walk(root)
    if words:
        pattern = "|".join(re.escape(w) for w in sorted(
            {w for w in words if len(w) >= HIGHLIGHT_MIN}, key=len, reverse=True)[:HIGHLIGHT_WORDS])
        if pattern:
            # Only text, never inside a tag, or the markup is rewritten too.
            body = re.sub(rf"(?<![<\w])({pattern})(?![\w>])",
                          r"<mark>\1</mark>", body, flags=re.IGNORECASE)
    return (f"<!doctype html><meta charset=utf-8><title>{_h.escape(topic.title)}</title>"
            f"<style>{_TOPIC_CSS}mark{{background:#fde0de;color:#8a1711;"
            f"border-radius:2px;padding:0 2px}}</style>{body}")


def topic_render_shot(topic, out_dir: str, name: str, words: list | None = None) -> str | None:
    """Render the topic to HTML and photograph it, so the report can show it."""
    if not os.path.exists(CHROME):
        return None
    import subprocess

    os.makedirs(os.path.join(out_dir, SHOTS), exist_ok=True)
    html_path = os.path.join(out_dir, SHOTS, f"{name}.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_topic_html(topic, words))
    rel = f"{SHOTS}/{name}.png"
    try:
        subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--screenshot={os.path.join(out_dir, rel)}", "--window-size=900,1300",
             f"file://{os.path.abspath(html_path)}"],
            capture_output=True, timeout=60, check=False)
    except Exception:
        return None
    return rel if os.path.exists(os.path.join(out_dir, rel)) else None


if __name__ == "__main__":
    raise SystemExit(main())
