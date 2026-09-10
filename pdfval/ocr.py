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
                tp = page_obj.get_textpage_ocr(flags=0, dpi=OCR_DPI, full=True, tessdata=tessdata_dir())
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
