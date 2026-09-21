"""A stable identity for an open PDF, for use as a cache key.

`id(doc)` is the obvious key and the wrong one: CPython reuses the id of a
freed object, so a document opened after another was closed can be handed the
closed one's cached pages, table regions or artwork boxes. The symptom is not a
crash - it is a run that quietly reports something it measured on a different
document, and two runs of the same pair that disagree.

Every document is stamped once with a number that is never reused, and the
caches key on that instead.
"""
from __future__ import annotations

import itertools

_COUNTER = itertools.count(1)
_ATTR = "_pdfval_doc_key"


def doc_key(doc) -> int:
    """A number unique to this open document, stable for its whole life."""
    key = getattr(doc, _ATTR, None)
    if key is None:
        key = next(_COUNTER)
        try:
            setattr(doc, _ATTR, key)
        except Exception:
            # A document that will not take the stamp falls back to its id,
            # which is the old behaviour - never worse than before.
            return id(doc)
    return key
