"""Language-neutral text helpers shared by the validators.

The validation engine extracts a PDF's embedded text with PyMuPDF, which
already works for any script the document's fonts carry (Latin, Cyrillic,
Greek, CJK, Kana, Hangul, Arabic, Hebrew, Devanagari and other Indic scripts,
Thai, ...). The *processing* on top of that text - deciding where one sentence
ends and the next begins, whether a wrapped line is a continuation, what a
callout label or a "page 12" cross-reference looks like - was written for
English only. This module keeps that logic Unicode-aware so a non-English
manual isn't drowned in false "changed text" findings.

Nothing here needs an extra install; it is pure `unicodedata` + `re`.
"""
from __future__ import annotations

import re
import unicodedata

# --- Sentence terminators -------------------------------------------------

# Terminators that in most scripts are followed by a space before the next
# sentence (Latin, Cyrillic, Greek, ...).
WESTERN_TERMINATORS = ".!?"

# Terminators from scripts that usually run sentences together with no space:
# ideographic full stop / bang / question (CJK), half-width ideographic stop,
# Devanagari/Bengali danda + double danda, Arabic question mark, Urdu full
# stop, Armenian full stop, Ethiopic full stop, interrobang.
CJK_TERMINATORS = "。！？．｡।॥؟۔։።‽"

ALL_TERMINATORS = WESTERN_TERMINATORS + CJK_TERMINATORS

# Closing quotes / brackets that may sit between a terminator and the space.
CLOSERS = "\"'’”»›)\\]〉》」』）］⦆"
_CLOSERS_SET = set(CLOSERS)
_WESTERN_SET = set(WESTERN_TERMINATORS)
_CJK_TERM_SET = set(CJK_TERMINATORS)

# Ranges whose letters have no upper/lower case, so "the next char is not
# uppercase" is not a usable "new sentence starts here" signal for them.
_CJK_START_RE = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯ｦ-ﾟ]"
)

# Full-width ASCII (U+FF01-FF5E) -> ASCII, ideographic + other wide spaces ->
# a normal space, CJK commas -> ASCII comma. Sentence-ending wide punctuation
# (! ? . 。) is deliberately *not* folded here - `split_sentences` needs to
# still recognise it as a terminator that has no following space.
_WIDTH_FOLD = {
    0x3000: " ", 0x2003: " ", 0x2002: " ", 0x2007: " ", 0x2008: " ",
    0x3001: ",", 0xFF0C: ",", 0xFF1B: ";", 0xFF1A: ":",
    0xFF08: "(", 0xFF09: ")", 0xFF0F: "/", 0xFF05: "%", 0xFF06: "&",
    0xFF0D: "-", 0x2010: "-", 0x2011: "-", 0x30FC: "-",
}


def fold_digits(text: str) -> str:
    """Map every Unicode decimal digit (Arabic-Indic, Devanagari, Thai, ...)
    to its ASCII equivalent so "٣" and "3" compare equal and page-number /
    list-marker detection works regardless of the document's numeral system.
    """
    out = []
    for ch in text:
        d = unicodedata.decimal(ch, None)
        out.append(str(d) if d is not None else ch)
    return "".join(out)


# Characters from scripts that never put a space between words inside a
# sentence (CJK ideographs, Kana, Hangul, half-width Katakana). A space sitting
# directly between two of these is always a PDF text-extraction artefact - the
# text layer broke a run mid-word - and it lands at different offsets in two
# exports of the same document, so otherwise identical sentences compare
# unequal unless it is removed.
_CJK_WORDLESS = (
    "぀-ヿ㐀-䶿一-鿿豈-﫿가-힯ｦ-ﾝ"
)
# Punctuation that, sandwiched in a CJK run, still counts as "inside CJK text"
# for dropping a stray space beside it ("方式, 包括" -> "方式,包括", "部件 (劳务"
# -> "部件(劳务", "和/ 或" -> "和/或"). A digit-led list marker ("1. 项目") is
# deliberately NOT covered - the char before its dot is a digit, not CJK, so
# that space survives and the marker is still recognised.
_CJK_MID_PUNCT = r",\uff0c\u3001;\uff1b:\uff1a\u3002\uff0e\uff01\uff1f!?/\uff0f)\uff09\u3011\u300d\u300f(\uff08\u3010\u300c\u300e"
_C = "[" + _CJK_WORDLESS + "]"
_P = "[" + _CJK_MID_PUNCT + "]"
_CJK_INTERIOR_SPACE_RE = re.compile(
    # CJK _ CJK   |   CJK-punct _ CJK   |   CJK _ punct-CJK
    r"(?<=" + _C + r")\s+(?=" + _C + r")"
    r"|(?<=" + _C + _P + r")\s+(?=" + _C + r")"
    r"|(?<=" + _C + r")\s+(?=" + _P + r"\s*" + _C + r")"
)


def fold_cjk_spaces(text: str) -> str:
    """Drop whitespace that sits inside a run of CJK / Kana / Hangul text.
    Those scripts don't space their words, so the gap is an extraction
    artefact, not content. A Latin run embedded in the text ("Bluetooth SIG,
    Inc.") keeps its spaces - neither side of the gap there is a
    wordless-script character.
    """
    return _CJK_INTERIOR_SPACE_RE.sub("", text)


def fold_width(text: str) -> str:
    """Fold full-width Latin letters/digits/punctuation and wide spaces to
    their ASCII forms (leaving sentence-ending wide punctuation alone).
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in _WIDTH_FOLD:
            out.append(_WIDTH_FOLD[cp])
        elif 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A or 0xFF10 <= cp <= 0xFF19:
            out.append(chr(cp - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def is_caseless_letter(ch: str) -> bool:
    return ch.isalpha() and not ch.isupper() and not ch.islower()


def starts_lower_or_caseless(ch: str) -> bool:
    """True when `ch` is a lowercase letter, a caseless-script letter, or a
    combining mark - i.e. it does not look like the first character of a new
    sentence / heading.
    """
    if not ch:
        return False
    return ch.islower() or is_caseless_letter(ch) or unicodedata.combining(ch) != 0


def ends_with_terminator(text: str) -> bool:
    """Does `text` end with sentence-ending punctuation (any script), allowing
    a trailing closing quote / bracket and the ASCII ``:`` / ``;``?
    """
    s = text.rstrip()
    if not s:
        return False
    if s[-1] in _CLOSERS_SET and len(s) >= 2:
        s = s[:-1].rstrip()
    return bool(s) and (s[-1] in ALL_TERMINATORS or s[-1] in ":;")


def _is_digit_marker(text: str, dot_index: int) -> bool:
    """The ``.`` at `dot_index` is a standalone list marker ("1.", "2.") - a
    lone 1-2 digit run at the start of the string or right after whitespace /
    "(" - rather than a real sentence end.
    """
    if text[dot_index] != ".":
        return False
    k = dot_index - 1
    digits = 0
    while k >= 0 and text[k].isdigit() and digits < 2:
        k -= 1
        digits += 1
    if digits == 0:
        return False
    return k < 0 or text[k].isspace() or text[k] == "("


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences across scripts.

    * A Western terminator (``. ! ?``) splits when it is followed by
      whitespace and the next sentence starts with something other than a
      lowercase / caseless letter, OR when it is immediately followed (no
      space) by a CJK character.
    * A CJK-style terminator (``。 ！ ？`` danda, Arabic ``؟`` ...) splits
      whether or not a space follows, as long as the next character is not
      itself another terminator or a closer.
    * A lone ``digit + .`` list marker never triggers a split.
    """
    parts: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in _WESTERN_SET and ch not in _CJK_TERM_SET:
            i += 1
            continue
        j = i + 1
        while j < n and text[j] in _CLOSERS_SET:
            j += 1
        k = j
        while k < n and text[k].isspace():
            k += 1
        if k >= n:
            break
        nxt = text[k]
        split_here = False
        if ch in _WESTERN_SET:
            if not _is_digit_marker(text, i):
                if k > j and not starts_lower_or_caseless(nxt):
                    split_here = True
                elif k == j and _CJK_START_RE.match(nxt):
                    split_here = True
        else:  # CJK-style terminator
            if nxt not in _CJK_TERM_SET and nxt not in _CLOSERS_SET:
                split_here = True
        if split_here:
            piece = text[start:j].strip()
            if piece:
                parts.append(piece)
            start = k
            i = k
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


# --- Callout labels -----------------------------------------------------

# Leading "NOTE:" / "注意：" style callout labels in the languages a
# hardware manual is commonly localised into. Matched only with an explicit
# separator (ASCII or full-width colon, or a dash) so ordinary prose that
# happens to start with one of these words is not mistaken for a label.
_CALLOUT_WORDS = [
    # English
    "tip", "tips", "note", "notes", "warning", "caution", "important",
    "attention", "info", "information", "hint", "remark", "danger",
    # German
    "hinweis", "achtung", "warnung", "wichtig", "tipp", "anmerkung", "vorsicht", "gefahr",
    # French
    "remarque", "attention", "avertissement", "avis", "conseil", "astuce", "important", "danger",
    # Spanish / Portuguese
    "nota", "aviso", "advertencia", "importante", u"atención", "consejo", u"precaución",
    u"atenção", "dica", "cuidado", "perigo",
    # Italian
    "avviso", "avvertenza", "attenzione", "suggerimento", "importante", "pericolo",
    # Dutch
    "opmerking", "waarschuwing", "belangrijk", "voorzichtig", "gevaar",
    # Nordic
    "obs", "merk", "huom", "notera", "vigtigt", u"tärkeää",
    # Japanese
    u"注記", u"注意", u"警告", u"重要", u"ヒント",
    u"メモ", u"参考", u"危険", u"お知らせ",
    # Chinese
    u"注意", u"警告", u"警告", u"重要", u"提示",
    u"说明", u"备注", u"注", u"危险", u"警示",
    # Korean
    u"참고", u"주의", u"경고", u"중요", u"팁", u"알림",
]
_CALLOUT_LABEL_RE = re.compile(
    r"^\s*(" + "|".join(sorted({re.escape(w) for w in _CALLOUT_WORDS}, key=len, reverse=True)) + r")"
    r"\s*[:：]\s*",
    re.IGNORECASE,
)


def match_callout_label(text: str) -> str | None:
    m = _CALLOUT_LABEL_RE.match(text)
    return m.group(1).upper() if m else None


def strip_callout_label(text: str) -> str:
    return _CALLOUT_LABEL_RE.sub("", text, count=1)


# --- Page cross-references --------------------------------------------------

# "page" in the languages above, plus common abbreviations. Used to drop
# "... see page 12" / "第 12 頁" style cross-references, whose number is
# expected to differ between two documents with different pagination.
_PAGE_WORDS_LATIN = [
    "page", "pages", u"pág", "pag", "pagina", u"página", "seite", "seiten",
    "side", "sida", "sivu", "oldal", "strona", "str", u"straně", u"σελίδα",
    u"стр", u"страница", u"страніці",
]
_PAGE_CONNECTIVES = [
    "on", "see", "refer to", "at", "siehe", "voir", "ver", "vedi", "vd", "zie",
    u"см", u"смтре",
]
_PAGE_REFERENCE_WITH_NUM_RE = re.compile(
    r"\s*(?:\b(?:" + "|".join(map(re.escape, _PAGE_CONNECTIVES)) + r")\s+)?"
    r"\b(?:" + "|".join(map(re.escape, _PAGE_WORDS_LATIN)) + r")\.?\s*\d{1,4}\b",
    re.IGNORECASE,
)
# CJK: "第12頁" / "12 ページ" / "12页" / "12쪽" (number may lead or follow).
_PAGE_REFERENCE_CJK_RE = re.compile(
    r"\s*(?:第\s*)?\d{1,4}\s*(?:ページ|頁|页|페이지|쪽)"
)
_DANGLING_PAGE_REFERENCE_RE = re.compile(
    r"\s*\b(?:" + "|".join(map(re.escape, _PAGE_CONNECTIVES)) + r")\s+"
    r"(?:" + "|".join(map(re.escape, _PAGE_WORDS_LATIN)) + r")\b\s*(?=[.,;:，；]|$)",
    re.IGNORECASE,
)


def strip_page_references(text: str) -> str:
    text = _PAGE_REFERENCE_WITH_NUM_RE.sub("", text)
    text = _PAGE_REFERENCE_CJK_RE.sub("", text)
    return _DANGLING_PAGE_REFERENCE_RE.sub("", text)
