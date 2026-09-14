"""What each finding MEANS, in one place.

Both reports explain a finding the same way, because they are read together:
report.html lists every finding with its explanation, and the section browser
shows the same issue in the section it belongs to. Keeping two copies of the
wording guaranteed they would drift, and the section browser's copy was the
terse one - a reader who only opens the browser (most of them) got "Image
missing" and nothing else.

`what` says what the finding is and, where it matters, why it is reported at
all; `fix` says what to do about it. Looked up by message, longest key first,
so a message that extends another ("Table heading missing" vs "Table heading
missing in continuation") still finds its own entry.
"""
from __future__ import annotations

ISSUE_HELP: dict[str, dict[str, str]] = {
    'Page count mismatch': {
        "what": 'The two PDFs have a different total number of pages.',
        "fix": 'Confirm whether pages were intentionally added or removed; otherwise regenerate Staging from the same source as Production.',
    },
    'Missing page': {
        "what": 'A page present in Production could not be found in Staging.',
        "fix": 'Add the missing page back into Staging.',
    },
    'Extra page': {
        "what": 'Staging has a page that is not present in Production.',
        "fix": 'Remove the extra page, or confirm it was an intentional addition.',
    },
    'Missing text': {
        "what": 'This text exists in Production but could not be found in Staging.',
        "fix": 'Add the missing text back into Staging.',
    },
    'Added text': {
        "what": 'This text appears in Staging but not in Production.',
        "fix": 'Remove the extra text from Staging, or confirm it was an intentional addition.',
    },
    'Changed text': {
        "what": 'The wording differs between Production and Staging for this section.',
        "fix": 'Align the Staging text with Production (or confirm the change was intended).',
    },
    'Minor text difference': {
        "what": 'The two sentences are near-identical and the wording is still present in both documents - they differ only by spacing, punctuation, or a list-marker glyph that one export captured as text and the other did not. Shown for completeness; never fails the run.',
        "fix": 'Usually nothing - confirm the difference is cosmetic. If a bullet or a whole list marker was genuinely dropped, restore it in Staging.',
    },
    'Possible text encoding issue': {
        "what": 'Unusual or garbled characters were found - usually a sign a font was extracted with the wrong encoding. The count shows how many times each character occurs, and the snippets show where.',
        "fix": 'Check the font embedding/encoding of the source document and re-export the PDF. Characters listed as not present in the other document were introduced by this export.',
    },
    'Text formatting differs': {
        "what": 'The wording is identical but the styling is not - the font family, size, bold or italic changed between Production and Staging.',
        "fix": 'Restore the Staging styling to match Production, or confirm the restyle was intended.',
    },
    'Image alignment changed': {
        "what": 'A figure that was left/centre/right aligned in Production sits in a different alignment in Staging. Small positional drift is ignored - only a change of alignment class is reported.',
        "fix": 'Restore the figure to its original alignment, or confirm the change was intended.',
    },
    'Diagram callout number missing': {
        "what": 'A numbered leader-line callout printed on a Production diagram could not be found anywhere in the corresponding Staging section (or the page either side of it).',
        "fix": 'Restore the missing callout number(s) on the Staging diagram.',
    },
    'Diagram callouts stripped from figure': {
        "what": 'The Production diagram has a run of numbered callouts drawn on it with leader lines; the Staging version of the same diagram has none of them. The numbers still exist in the legend list beside the figure, but a reader can no longer point at a number on the picture and see which part it is.',
        "fix": 'Re-add the numbered leader-line callouts to the Staging diagram so it matches the legend.',
    },
    'Broken image': {
        "what": 'Artwork in Staging is unusable: its embedded image data will not decode, or the region renders as a blank flat colour.',
        "fix": 'Re-embed the image in Staging from a working source file.',
    },
    'Image highlight box missing': {
        "what": 'A coloured highlight box drawn on a Production figure is not present on the corresponding Staging figure.',
        "fix": 'Re-add the highlight box, or confirm it was intentionally removed.',
    },
    'Table header row not repeated on continuation page': {
        "what": 'This Staging table runs onto another page, and that page does not reprint the header row - so the reader is left with columns of values and nothing saying what they are. The header is mandatory on every continuation page, whatever Production does, so this is checked against Staging on its own.',
        "fix": 'Repeat the header row at the top of every page the table continues onto.',
    },
    'Table split across pages': {
        "what": 'A page break splits this table across more pages in Staging than in Production, so it no longer reads as one table.',
        "fix": 'Keep the table together the way Production does - or, if the split is intended, make sure its header row repeats on every continuation page.',
    },
    'Table columns differ': {
        "what": 'The two tables do not have the same number of columns.',
        "fix": 'Reconcile the column counts between the two documents.',
    },
    'Table column layout differs': {
        "what": 'The tables have the same columns, but the column widths/positions changed - a column was widened or narrowed.',
        "fix": 'Restore the Production column widths, or confirm the relayout was intended.',
    },
    'Table cell layout differs': {
        "what": 'A row holds the same words but they are no longer divided into the same cells - cells were merged or split.',
        "fix": 'Restore the original cell structure for that row.',
    },
    'Table breaking the margins': {
        "what": 'The table extends past the page margins in Staging.',
        "fix": 'Resize or reflow the table so it fits within the page margins.',
    },
    'Table heading missing': {
        "what": "A header cell present in Production is missing from Staging. The row was located by its anchor cell, and only that cell's own words were compared.",
        "fix": 'Restore the missing header text.',
    },
    'Table cell missing': {
        "what": "Part of a cell's content is gone in Staging. The row was located by its anchor cell (not by position), and the cell's words were compared as an unordered set - so a reordering alone is never reported.",
        "fix": 'Restore the missing words listed above to that cell.',
    },
    'List marker changed': {
        "what": 'The same list items are marked differently in the two documents - Production numbers a procedure 1., 2., 3. where Staging letters it a., b., c. (or drops it to a bullet). The step text is identical, which is why the content diff shows nothing: only the markers changed. It is reported as a mismatch because it changes what the reader is told - any "repeat step 3" around the list no longer resolves, and the marker is often drawn as its own text object beside the step, so it is invisible in the text comparison.',
        "fix": "Restore Production's marker style for these list items in Staging, or renumber the cross-references that point at them.",
    },
    'List marker size changed': {
        "what": "The same list items use the same kind of marker on both sides (e.g. a bullet on both), but the marker itself prints at a noticeably different size in Staging - larger or smaller than its Production counterpart. The item's own text is unaffected; only the marker glyph's size changed.",
        "fix": "Restore Production's marker size for these list items in Staging, or confirm the size change was intended.",
    },
    'List alignment/indent changed': {
        "what": "A list item's indent moved - it changed nesting level, or lost its hanging indent.",
        "fix": 'Restore the original indent, or confirm the nesting change was intended.',
    },
    'Text alignment changed': {
        "what": 'The same text is aligned differently (left / center / right / justified).',
        "fix": 'Restore the Production alignment, or confirm the change was intended.',
    },
    'Paragraph merged with heading': {
        "what": "A numbered item's description has been pulled up onto the same line as its bold label, where Production keeps the label on its own line.",
        "fix": 'Put the description back on its own line beneath the label.',
    },
    'Hyperlink not highlighted in STAGE': {
        "what": 'The link works, but its text is styled exactly like body text - not coloured and not underlined - so a reader has no way to tell it is clickable.',
        "fix": 'Restore the link styling (colour and/or underline) in Staging.',
    },
    'Internal link target does not resolve': {
        "what": 'The link points at a page or named destination that does not exist in this document, so clicking it goes nowhere.',
        "fix": 'Repoint the link at a destination that exists, or remove it.',
    },
    'Hyperlink has no usable scheme': {
        "what": 'The link URI has no scheme a PDF viewer can act on, so clicking it does nothing.',
        "fix": 'Give the URI a usable scheme (https://, mailto:, ...).',
    },
    'Hyperlink hotspot cannot be clicked': {
        "what": 'The clickable rectangle is empty, too small to hit, or lies outside the page.',
        "fix": 'Resize the link hotspot so it covers its text.',
    },
    'Callout label differs': {
        "what": 'The same sentence is marked with a different callout type (e.g. NOTE vs WARNING) between Production and Staging.',
        "fix": 'Confirm the correct callout type and align Staging with Production (or vice versa if intended).',
    },
    'Callout icon missing in Staging': {
        "what": 'Production marks this NOTE / TIP / WARNING / CAUTION with a filled, coloured badge to the left of the callout body. In Staging the badge is gone or is rendered as a flat ~13pt black outline glyph with no colour, so the visual callout cue is effectively lost even though the callout text is present.',
        "fix": 'Restore the coloured callout icon in Staging so the note/tip/warning is visually flagged as it is in Production.',
    },
    'List formatting differs': {
        "what": 'The same list item is styled differently between Production and Staging (e.g. numbered 1,2,3 vs lettered a,b,c vs a bullet).',
        "fix": 'Align the Staging list style with Production (or confirm the change was intended).',
    },
    'Image missing': {
        "what": 'No figure anywhere in the Staging section resembles this Production figure. Figures are matched by appearance across the whole section, so a figure that merely moved to another page inside the section is never reported here.',
        "fix": 'Re-add the missing figure to the Staging section.',
    },
    'Image content differs': {
        "what": "Staging shows a figure in this figure's place, the two were measured over comparable areas, and their artwork has nothing in common - the picture was replaced, or its contents changed. Compare the two images below.",
        "fix": 'Restore the Production figure, or confirm the replacement was intended.',
    },
    'Image could not be confirmed identical': {
        "what": "A figure sits in this figure's place in Staging, but it could not be confirmed to be the same picture - either the two documents detected different extents of it (one crop caught the leader lines or the figure beside it) or the two are only partly alike. No defect is being asserted; this needs a human eye.",
        "fix": 'Compare the two images below and confirm whether the Staging figure is correct.',
    },
    'Image size changed': {
        "what": 'The same picture, rendered at a materially different width and/or height in Staging (more than 15% and at least 12pt on that axis). Small scaling differences from re-export are ignored.',
        "fix": 'Restore the Production dimensions, or confirm the resize was intended.',
    },
    'Image width changed': {
        "what": 'The same picture, rendered materially wider or narrower in Staging (more than 15% and at least 12pt), with its height unchanged. Small scaling differences from re-export are ignored.',
        "fix": 'Restore the Production width, or confirm the resize was intended.',
    },
    'Image outside its section': {
        "what": 'The figure is still in Staging, but it now sits past the end of the section it belongs to. Moving to a later page WITHIN the section is fine and is never reported; leaving the section is not.',
        "fix": 'Move the figure back under its own heading.',
    },
    'Image label missing': {
        "what": "A caption or label word on a Production figure could not be found in Staging's own lines for the same section, including text OCR reads out of the artwork itself.",
        "fix": 'Restore the missing label text on the Staging figure.',
    },
    'Table row missing': {
        "what": 'A row present in Production could not be found in Staging. The row was located by its anchor cell rather than by position, so an inserted row elsewhere does not cause this.',
        "fix": 'Add the missing row back into the Staging table.',
    },
    'Underline missing in Staging': {
        "what": 'A drawn underline rule sits under this text in Production; the same words print with no rule under them in Staging. A PDF has no font flag for underline, so this is judged from the drawing itself, not a style name.',
        "fix": 'Restore the underline rule in Staging, or confirm removing it was intended.',
    },
    'Underline added in Staging': {
        "what": 'These words print plain in Production, but Staging draws a rule under them.',
        "fix": 'Remove the underline in Staging, or confirm adding it was intended.',
    },
}

# Longest first: a message that starts with another message's key must match
# its own, more specific entry.
_KEYS_BY_LENGTH = sorted(ISSUE_HELP, key=len, reverse=True)


def help_for(message: str) -> dict[str, str] | None:
    """The explanation for a finding, or None when it has none written yet."""
    if not message:
        return None
    if message in ISSUE_HELP:
        return ISSUE_HELP[message]
    for key in _KEYS_BY_LENGTH:
        if message.startswith(key):
            return ISSUE_HELP[key]
    return None


def what(message: str, default: str = "") -> str:
    entry = help_for(message)
    return entry["what"] if entry else default


def fix(message: str, default: str = "") -> str:
    entry = help_for(message)
    return entry["fix"] if entry else default
