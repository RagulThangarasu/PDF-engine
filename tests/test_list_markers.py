"""Behaviour tests for the list-marker check.

Run with:  python -m pytest tests            (or)  python tests/test_list_markers.py

A procedure numbered 1,2,3 in Production and lettered a,b,c in Staging is a
real regression - every "repeat step 3" around it stops resolving - and it is
invisible to the text diff, because the marker is either stripped from the
sentence or drawn outside it entirely. These fixtures pin down both halves of
that: it IS reported when the style changes, and it is NOT reported merely
because the two documents draw an identical marker differently.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from pdfval.validators import validate_alignment  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture(name: str) -> str:
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        from tests import make_fixtures

        make_fixtures.main()
    return path


def _marker_issues(prod: str, stage: str) -> list:
    expected = fitz.open(_fixture(prod))
    actual = fitz.open(_fixture(stage))
    try:
        result = validate_alignment(expected, actual, None)
    finally:
        expected.close()
        actual.close()
    return [i for i in result.issues if i.message == "List marker changed"]


def test_renumbered_list_is_reported_once_for_the_whole_list():
    issues = _marker_issues("list_prod.pdf", "list_stage_lettered.pdf")
    assert len(issues) == 1, [i.details for i in issues]
    issue = issues[0]
    # A renumbered procedure is a mismatch, not advisory layout drift.
    assert issue.severity == "error"
    details = issue.details
    assert details["items"] == 5
    assert details["heading"] == "Setting the display time"
    assert "numbered" in details["expected_marker"] and "1." in details["expected_marker"]
    assert "lettered" in details["actual_marker"] and "a." in details["actual_marker"]
    assert "Production numbers" in details["changed"]
    assert "Staging letters" in details["changed"]


def test_same_numbering_drawn_as_detached_markers_is_not_reported():
    # Staging draws "1." as its own text object beside the step instead of at
    # the start of its line. Same numbering, so nothing to report - this is the
    # case that must not become a false positive now that detached markers are
    # picked up at all.
    assert _marker_issues("list_prod.pdf", "list_stage_detached.pdf") == []


def test_numbered_list_reduced_to_bullets_is_reported():
    issues = _marker_issues("list_prod.pdf", "list_stage_bulleted.pdf")
    assert len(issues) == 1, [i.details for i in issues]
    assert issues[0].details["items"] == 5
    assert "bulleted" in issues[0].details["actual_marker"]


def test_identical_documents_report_nothing():
    assert _marker_issues("list_prod.pdf", "list_prod.pdf") == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
