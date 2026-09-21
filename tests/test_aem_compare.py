"""Comparing a published PDF against the DITA topics it was built from."""
import os

import pytest

from pdfval import aem_compare as A

_TOPIC = b"""<?xml version="1.0"?>
<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
<topic id="t1">
  <title>Connecting the Receiver</title>
  <prolog><metadata><othermeta name="skip" content="not content"/></metadata></prolog>
  <body>
    <p>Connect the HDMI cable to the HDMI out port on the Receiver.</p>
    <note>Use the supplied adapter.</note>
    <ul><li>first</li><li>second</li></ul>
    <table><row/><row/><row/></table>
  </body>
</topic>"""


def test_a_topic_is_reduced_to_its_words_and_its_structure():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    assert t.title == "Connecting the Receiver"
    assert "HDMI out port" in t.text
    assert t.notes == 1 and t.list_items == 2 and t.tables == [3]


def test_the_title_and_metadata_are_not_counted_as_body_words():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    assert "not content" not in t.text
    assert not t.text.startswith("Connecting the Receiver")


def _section(title, text, notes=0, items=0, tables=()):
    return {"title": title, "page": 7, "text": text, "notes": notes,
            "list_items": items, "tables": list(tables)}


def test_words_in_the_pdf_and_not_the_topic_are_reported():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver",
                   t.text + " Then slide the power switch to the on position.",
                   notes=1, items=2, tables=[3])
    kinds = {f["kind"] for f in A.compare([t], [sec])}
    assert "words-not-in-topic" in kinds
    assert "words-not-published" not in kinds


def test_words_in_the_topic_and_not_the_pdf_are_reported():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver", "Connect the HDMI cable.", notes=1, items=2, tables=[3])
    assert any(f["kind"] == "words-not-published" for f in A.compare([t], [sec]))


def test_a_topic_with_no_section_and_a_section_with_no_topic():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    other = _section("Something else entirely", "unrelated words here")
    kinds = {f["kind"] for f in A.compare([t], [other])}
    assert kinds >= {"topic-not-published", "section-not-in-topics"}


def test_a_structure_count_that_disagrees_is_reported():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver", t.text, notes=0, items=2, tables=[3])
    notes = [f for f in A.compare([t], [sec]) if f["kind"] == "structure"]
    assert notes and "note" in notes[0]["detail"]


def test_a_matching_topic_and_section_report_nothing():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver", t.text, notes=1, items=2, tables=[3])
    assert A.compare([t], [sec]) == []


def test_credentials_are_required_and_never_defaulted():
    """An author instance answers 401 without them; guessing at likely pairs
    against someone's server is not something a tool should do on its own."""
    with pytest.raises(ValueError):
        A.AemClient("http://host:4502", "", "")


def test_a_relative_topic_href_resolves_against_the_map():
    assert A._resolve("/content/dam/x/Maps", "../Topics/a.dita") == "/content/dam/x/Topics/a.dita"


def test_the_report_is_one_self_contained_file(tmp_path):
    findings = [{"kind": "words-not-in-topic", "severity": "high", "topic": "/x/t1.dita",
                 "title": "Connecting the Receiver", "page": 7, "detail": "Printed in the PDF: x."}]
    path = A.write_report(findings, str(tmp_path), "manual.pdf", "/x/map.ditamap", 1, 1)
    html = open(path, encoding="utf-8").read()
    assert os.path.basename(path) == "aem-vs-pdf.html"
    assert "Connecting the Receiver" in html and "manual.pdf" in html
    assert "<style>" in html          # no external assets to lose
