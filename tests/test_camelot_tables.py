"""Tables read a second time with Camelot: a row Staging drops and a value it
changes are both found, row by row, with a "Table issue:" comment."""
import fitz
import pytest

from pdfval.validators import camelot_tables as ct
from pdfval.validators.chapter import KIND_TABLE, Element

pytestmark = pytest.mark.skipif(not ct.available(), reason="Camelot not installed")

_ROWS = [["Brightness", "3000 ANSI lumens"], ["Resolution", "1920 x 1080"], ["Throw ratio", "1.2"],
         ["Weight", "2.5 kg"], ["Noise level", "28 dB"]]


def _build(path, rows):
    doc = fitz.open()
    page = doc.new_page()
    y = 170
    for row in [["Item", "Value"]] + rows:
        for x, cell in zip((77, 225), row):
            page.insert_text((x, y + 15), cell, fontsize=10)
        page.draw_rect(fitz.Rect(72, y, 520, y + 22), color=(0, 0, 0), width=0.8)
        page.draw_line((220, y), (220, y + 22), color=(0, 0, 0), width=0.8)
        y += 22
    doc.save(path)
    return fitz.open(path)


def test_camelot_finds_missing_row_and_changed_value(tmp_path):
    ct.reset_cache()
    stage_rows = [r for r in _ROWS if r[0] != "Throw ratio"]
    stage_rows[-1] = ["Noise level", "32 dB"]
    prod = _build(str(tmp_path / "p.pdf"), _ROWS)
    stage = _build(str(tmp_path / "s.pdf"), stage_rows)
    at = lambda page, y: "t0"
    diffs = ct.table_changes(ct.extract(prod, str(tmp_path / "p.pdf")), ct.extract(stage, str(tmp_path / "s.pdf")),
                             ((0, 0), None), ((0, 0), None), at, at, Element, KIND_TABLE, lambda t: set())
    by = {d["type"]: d for d in diffs}
    assert set(by) == {"table-row-missing", "table-cell"}
    assert "Throw ratio" in by["table-row-missing"]["summary"]
    assert by["table-cell"]["gone"] == ["28"] and by["table-cell"]["extra"] == ["32"]
    assert all(d["summary"].startswith("Table issue:") for d in diffs)

    # Already named by another finding: not reported twice.
    covered = lambda t: {"throw", "ratio", "1", "2", "28", "32"}
    assert ct.table_changes(ct.extract(prod, str(tmp_path / "p.pdf")), ct.extract(stage, str(tmp_path / "s.pdf")),
                            ((0, 0), None), ((0, 0), None), at, at, Element, KIND_TABLE, covered) == []


@pytest.mark.parametrize("prod, stage, expected", [
    ('See "Package contents".', "See Package contents.", ["“\"” missing in Staging"]),
    ("Insert the drive.", "Insert the drive", ["“.” missing in Staging"]),
    ("Press OK. Then wait", "Press OK.Then wait", ["space missing in Staging"]),
    ("from 0 m - 2000 m", "from 0 m – 2000 m", ["“-” missing in Staging, “–” extra in Staging"]),
    # never differences: wrapping, curly vs straight quotes, bullets, callout labels, page refs
    ("on,\nthen off.", "on, then off.", []),
    ("“Wired projection”", '"Wired projection"', []),
    ("online. • Use", "online. Use", []),
    ("power- saving mode", "power-saving mode", []),
    ("Tip To do", "Tip: To do", []),
    ('drive" on page 63)', 'drive")', []),
])
def test_punctuation_quotes_and_spaces(prod, stage, expected):
    from pdfval.validators.chapter import mark_changes

    found = mark_changes(prod, stage)
    assert len(found) == len(expected) or (expected and len(found) >= 1)
    for want in expected:
        assert any(want in f for f in found), found
