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


def _section(title, text, notes=0, items=0, tables=(), start=(0, 0.0), stop=(0, 1e9)):
    return {"title": title, "page": 7, "text": text, "notes": notes, "level": 1,
            "start": start, "stop": stop, "list_items": items, "tables": list(tables)}


def test_words_in_the_pdf_and_not_the_topic_are_reported():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver",
                   t.text + " Then slide the power switch to the on position.",
                   notes=1, items=2, tables=[3])
    found = [f for f in A.compare([t], [sec]) if f["kind"] == "words-differ"]
    assert found and "in the pdf and not in the topic" in found[0]["detail"].lower()
    assert "in the topic and not printed" not in found[0]["detail"].lower()


def test_words_in_the_topic_and_not_the_pdf_are_reported():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver", "Connect the HDMI cable.", notes=1, items=2, tables=[3])
    found = [f for f in A.compare([t], [sec]) if f["kind"] == "words-differ"]
    assert found and "in the topic and not printed" in found[0]["detail"].lower()


def test_both_sides_come_back_with_only_the_differences_marked():
    """The report shows each side in full - a reviewer reads what is there -
    so the difference is marked IN the text, not listed away from it."""
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver", t.text + " Slide the power switch on.",
                   notes=1, items=2, tables=[3])
    f = [x for x in A.compare([t], [sec]) if x["kind"] == "words-differ"][0]
    assert any(not differs for _, differs in f["topic_parts"])      # matching text stays plain
    marked = [text for text, differs in f["pdf_parts"] if differs]
    assert marked and "power switch" in " ".join(marked)


def test_identical_text_marks_nothing():
    a, b = A.diff_parts("the same words on both sides", "the same words on both sides")
    assert not any(d for _, d in a) and not any(d for _, d in b)


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


def test_a_sub_heading_inside_a_matched_chapter_is_not_a_gap():
    """A topic is a whole chapter. The headings inside it belong to that
    topic - reported separately, every sub-heading of the manual reads as
    content nobody wrote."""
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    chapter = _section("Connecting the Receiver", t.text, notes=1, items=2, tables=[3],
                       start=(4, 0.0), stop=(9, 0.0))
    inside = _section("Video input via HDMI", "connect the hdmi cable",
                      start=(5, 100.0), stop=(6, 0.0))
    kinds = [f["kind"] for f in A.compare([t], [chapter, inside])]
    assert "section-not-in-topics" not in kinds


def test_a_chapter_with_no_topic_at_all_is_still_a_gap():
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    chapter = _section("Connecting the Receiver", t.text, notes=1, items=2, tables=[3],
                       start=(4, 0.0), stop=(9, 0.0))
    elsewhere = _section("Troubleshooting", "something else entirely",
                         start=(40, 0.0), stop=(50, 0.0))
    kinds = [f["kind"] for f in A.compare([t], [chapter, elsewhere])]
    assert "section-not-in-topics" in kinds


def test_the_style_check_needs_a_pdf_to_look_at():
    """With no page to read there is no evidence either way - asserting the
    template dropped every <b> in the topic would be a guess."""
    t = A.parse_topic("/x/t1.dita", _TOPIC)
    sec = _section("Connecting the Receiver", t.text, notes=1, items=2, tables=[3])
    assert not [f for f in A.compare([t], [sec]) if f["kind"].startswith("style-")]


def test_markup_the_topic_declares_is_read_back_by_role():
    roles = A.topic_roles(_TOPIC.decode())
    assert roles["note"] == ["Use the supplied adapter."]
    assert roles["list item"] == ["first", "second"]
    assert roles["bold"] == [] and roles["link"] == []
