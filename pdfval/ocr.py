"""Optional OCR of a page's artwork, used to read text that lives inside a
figure rather than in the PDF's text layer.

A caption or a callout label baked into an illustration is invisible to
`get_text`, so a comparison that only reads the text layer cannot tell a figure
that lost its label from one that never had a text-layer label at all. OCR
fills that gap - but only where it is available: Tesseract is an optional
system dependency, and every entry point here degrades to "no OCR words found"
rather than failing a run when it is missing.
"""
from __future__ import annotations

import os
import re
import shutil

import fitz

OCR_DPI = 200

# Tesseract's language data isn't always on TESSDATA_PREFIX, and PyMuPDF needs
# an explicit path. These are where the usual package managers put it.
_TESSDATA_CANDIDATES = (
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/usr/share/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
)

_tessdata: str | None | bool = False  # False = not yet resolved


def tessdata_dir() -> str | None:
    """Resolve Tesseract's language-data directory once, or None if OCR is
    unavailable on this machine.
    """
    global _tessdata
    if _tessdata is not False:
        return _tessdata  # type: ignore[return-value]

    _tessdata = None
    if shutil.which("tesseract"):
        env = os.environ.get("TESSDATA_PREFIX")
        candidates = ([env] if env else []) + list(_TESSDATA_CANDIDATES)
        for path in candidates:
            if path and os.path.isdir(path) and os.path.isfile(os.path.join(path, "eng.traineddata")):
                _tessdata = path
                break
    return _tessdata


def available() -> bool:
    return tessdata_dir() is not None


# Tesseract's own default is "eng" - reading a Traditional Chinese callout
# page with it produces near-total garbage, since a Latin-only model has no
# CJK glyphs to recognise at all. Every OCR call here instead picks the
# script actually printed on the page, straight from its own PDF text layer
# (already extracted, so no extra render), and only ever falls back to
# English when nothing else in the packs installed on this machine matches.
_HANGUL_RE = re.compile(r"[가-힣]")
_KANA_RE = re.compile(r"[぀-ヿ]")
_HAN_RE = re.compile(r"[一-鿿]")
_SIMPLIFIED_ONLY_RE = re.compile(r"[产装内会与关国东车义为]")  # simplified forms with no traditional-form overlap
_SCRIPT_LANGS = {
    "hangul": ("kor",),
    "kana": ("jpn",),
    # chi_tra and chi_sim both loaded at once measurably hedges toward
    # simplified output even on an all-Traditional page (Tesseract picks
    # whichever combined model scores higher character by character) - so
    # only the variant this page's OWN text layer actually uses is loaded.
    "han_simplified": ("chi_sim",),
    "han_traditional": ("chi_tra",),
}


def _detect_script(text: str) -> str | None:
    if _HANGUL_RE.search(text):
        return "hangul"
    if _KANA_RE.search(text):
        return "kana"
    if _HAN_RE.search(text):
        return "han_simplified" if _SIMPLIFIED_ONLY_RE.search(text) else "han_traditional"
    return None


def _ocr_language(text: str) -> str:
    """The Tesseract language string for text written in this script - its
    own packs first, "eng" always appended so Latin words, model numbers and
    punctuation mixed into the same page are still read.  A pack this
    installation does not have (a language pack is optional, same as
    Tesseract itself) is silently left out rather than passed to Tesseract,
    which would otherwise fail the whole OCR call over one missing language.
    """
    script = _detect_script(text)
    if script is None:
        return "eng"
    tessdata = tessdata_dir()
    have = lambda lang: tessdata and os.path.isfile(os.path.join(tessdata, f"{lang}.traineddata"))
    packs = [lang for lang in _SCRIPT_LANGS[script] if have(lang)]
    packs.append("eng")
    return "+".join(packs)


# OCR is the slowest thing in a run (~0.3s a page). Content Validation, Image
# Validation and the section browser each build their own `PageWords` and would
# otherwise OCR the same page several times over - on a big manual that alone
# pushed a web run past the point where it looked hung. Cache the raw Tesseract
# read per (document-id, page) so every `PageWords` shares it. `reset_ocr_cache`
# is called at the end of each run (ids can then be safely reused).
_OCR_RAW_CACHE: dict[tuple[int, int], list[tuple]] = {}
# A hard ceiling so a pathological document can't spend minutes in OCR; once
# hit, later pages just return no OCR words (the checks degrade gracefully).
OCR_PAGE_BUDGET = 220
_ocr_pages_done = 0


def reset_ocr_cache() -> None:
    global _ocr_pages_done
    _SCANNED_CACHE.clear()
    _TEXT_DICT_CACHE.clear()
    _OCR_RAW_CACHE.clear()
    _ocr_pages_done = 0


class PageWords:
    """Per-page word lists, memoized: the words in the PDF's own text layer,
    and (when OCR is available) the words Tesseract reads off the rendered
    page. Rendering and OCR-ing a page is expensive, so it happens at most
    once per page and only when something actually asks for artwork words.
    """

    def __init__(self, doc: fitz.Document):
        self._doc = doc
        self._native: dict[int, list[tuple]] = {}
        self._ocr: dict[int, list[tuple]] = {}
        self._ocr_raw: dict[int, list[tuple]] = {}

    def native_words(self, page: int) -> list[tuple]:
        """[(x0, y0, x1, y1, word), ...] from the PDF text layer."""
        if page not in self._native:
            try:
                self._native[page] = [
                    (w[0], w[1], w[2], w[3], w[4]) for w in self._doc[page].get_text("words")
                ]
            except Exception:
                self._native[page] = []
        return self._native[page]

    def _run_ocr(self, page: int) -> list[tuple]:
        if page in self._ocr_raw:
            return self._ocr_raw[page]
        global _ocr_pages_done
        key = (id(self._doc), page)
        shared = _OCR_RAW_CACHE.get(key)
        if shared is not None:
            self._ocr_raw[page] = shared
            return shared
        result: list[tuple] = []
        if available() and _ocr_pages_done < OCR_PAGE_BUDGET:
            try:
                page_obj = self._doc[page]
                # The page's own text layer already tells us what script is
                # printed here - read straight off it, no extra render, and
                # right even for a page OCR is being asked to double-check
                # BECAUSE the text layer might be wrong (a wrong character is
                # still the right script almost always).
                language = _ocr_language(page_obj.get_text("text"))
                tp = page_obj.get_textpage_ocr(
                    flags=0, language=language, dpi=OCR_DPI, full=True, tessdata=tessdata_dir()
                )
                result = [
                    (w[0], w[1], w[2], w[3], w[4]) for w in page_obj.get_text("words", textpage=tp)
                ]
            except Exception:
                result = []
            _ocr_pages_done += 1
        _OCR_RAW_CACHE[key] = result
        self._ocr_raw[page] = result
        return result

    def ocr_words(self, page: int) -> list[tuple]:
        """Every word Tesseract reads off the rendered page, UNFILTERED - use
        this (not `artwork_words`) when checking "is this content genuinely
        visible on the page in any form". `artwork_words` subtracts any word
        whose text matches ANY native word ANYWHERE on the page (not just
        nearby it), so a short, common word genuinely baked into a figure
        (e.g. "status", "icon") gets wrongly discarded whenever an unrelated
        native sentence elsewhere on the same page happens to use the same
        word - this method skips that subtraction entirely.
        """
        return self._run_ocr(page)

    def artwork_words(self, page: int) -> list[tuple]:
        """Words that OCR finds on the rendered page but that the text layer
        does NOT already account for - i.e. text baked into the artwork.

        Subtracting the text layer is what keeps this from re-reporting every
        ordinary paragraph on the page as if it were part of a figure.
        """
        if page in self._ocr:
            return self._ocr[page]
        ocr_words = self._run_ocr(page)
        if not ocr_words:
            self._ocr[page] = []
            return self._ocr[page]

        explained = {normalize_word(w[4]) for w in self.native_words(page)}
        explained.discard("")
        self._ocr[page] = [w for w in ocr_words if normalize_word(w[4]) not in explained]
        return self._ocr[page]


def normalize_word(word: str) -> str:
    """Fold a word to a comparison key. OCR routinely disagrees with the text
    layer about case and edge punctuation on text that is otherwise identical,
    so neither is allowed to matter. NFKC decomposition also folds ligatures
    ("conﬁrm" -> "confirm") and full-width glyphs, which the text layer and OCR
    frequently disagree on.
    """
    import unicodedata

    word = unicodedata.normalize("NFKC", word)
    return "".join(ch for ch in word.lower() if ch.isalnum())


def words_in(words: list[tuple], bbox: tuple, margin: float = 0.0) -> list[str]:
    """The words whose centre falls inside `bbox` (grown by `margin`)."""
    x0, y0, x1, y1 = bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin
    out = []
    for wx0, wy0, wx1, wy1, text in words:
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append(text)
    return out


# A page is "scanned" when one embedded image covers most of it and the text
# layer carries next to nothing: its words exist only as pixels, so every
# text comparison has to go through OCR or it silently compares nothing.
SCANNED_RASTER_FRAC = 0.75   # share of the page one image must cover
SCANNED_MAX_NATIVE_WORDS = 8  # a stray page number / stamp is still "no text layer"
_SCANNED_CACHE: dict[tuple, bool] = {}


def is_scanned_page(doc: fitz.Document, page: int) -> bool:
    key = (id(doc), doc.page_count, page)
    if key in _SCANNED_CACHE:
        return _SCANNED_CACHE[key]
    result = False
    try:
        page_obj = doc[page]
        area = max(1.0, page_obj.rect.width * page_obj.rect.height)
        biggest = 0.0
        for info in page_obj.get_image_info():
            bbox = info.get("bbox")
            if bbox:
                r = fitz.Rect(bbox) & page_obj.rect
                biggest = max(biggest, r.width * r.height / area)
        if biggest >= SCANNED_RASTER_FRAC:
            words = [w for w in page_obj.get_text("words") if w[4].strip()]
            result = len(words) <= SCANNED_MAX_NATIVE_WORDS
    except Exception:
        result = False
    _SCANNED_CACHE[key] = result
    return result


def scanned_pages(doc: fitz.Document) -> list[int]:
    return [i for i in range(doc.page_count) if is_scanned_page(doc, i)]


def ocr_lines(words: list[tuple]) -> list[tuple]:
    """Group OCR words into printed lines: [(bbox, text), ...] in reading
    order. Words belong to one line when their vertical extents mostly
    overlap and they sit left-to-right without a column-sized gap."""
    bands: list[list[tuple]] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        if not str(w[4]).strip():
            continue
        band = bands[-1] if bands else None
        if band is not None:
            b0, b1 = min(x[1] for x in band), max(x[3] for x in band)
            h = min(b1 - b0, w[3] - w[1])
            if h > 0 and min(b1, w[3]) - max(b0, w[1]) >= 0.5 * h:
                band.append(w)
                continue
        bands.append([w])
    out = []
    for band in bands:
        band.sort(key=lambda w: w[0])
        height = max(1.0, sum(w[3] - w[1] for w in band) / len(band))
        pieces: list[list[tuple]] = [[band[0]]]
        for w in band[1:]:
            # a gap this wide is a column gutter, not a word space
            if w[0] - pieces[-1][-1][2] > 3 * height:
                pieces.append([w])
            else:
                pieces[-1].append(w)
        for row in pieces:
            bbox = (min(w[0] for w in row), min(w[1] for w in row),
                    max(w[2] for w in row), max(w[3] for w in row))
            out.append((bbox, " ".join(str(w[4]) for w in row)))
    out.sort(key=lambda r: (round(r[0][1], 1), r[0][0]))
    return out


_TEXT_DICT_CACHE: dict[tuple, dict] = {}


def page_text_dict(doc: fitz.Document, page: int) -> dict:
    """`page.get_text("dict")`, except that a scanned page answers with its
    OCR reading in the same shape (one block per line, one span per line) -
    so heading lookup finds a title that exists only as pixels. Font size is
    estimated from the line's height; weight is unknown and left regular."""
    if not is_scanned_page(doc, page):
        return doc[page].get_text("dict")
    key = (id(doc), doc.page_count, page)
    if key in _TEXT_DICT_CACHE:
        return _TEXT_DICT_CACHE[key]
    blocks = []
    for bbox, text in ocr_lines(PageWords(doc).ocr_words(page)):
        size = max(1.0, (bbox[3] - bbox[1]) * 0.8)
        span = {"text": text, "bbox": bbox, "size": size, "flags": 0, "font": "OCR",
                "color": 0, "origin": (bbox[0], bbox[3])}
        blocks.append({"type": 0, "bbox": bbox, "lines": [{"bbox": bbox, "spans": [span]}]})
    rect = doc[page].rect
    result = {"width": rect.width, "height": rect.height, "blocks": blocks}
    _TEXT_DICT_CACHE[key] = result
    return result
