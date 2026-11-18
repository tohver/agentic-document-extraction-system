"""Prompts, built once from `fields.py`.

The system prompts are static text (no dates, ids or per-document content), so they sit before
the document behind a cache breakpoint and are read from the prompt cache on every later call.
"""

from typing import Final

from policy_extractor.fields import CONFIDENCE_RUBRIC, GENERAL_RULES, render_definitions

EXTRACTOR_SYSTEM: Final = f"""\
You extract insurance policy declarations into a structured record. The documents are plain
text from PDFs or OCR: columns may be flattened, dot leaders and spacing are irregular, and
layouts differ by carrier.

Return the record and exactly one evidence entry for every field listed below.

General rules:
{GENERAL_RULES}

Field definitions:
{render_definitions()}

Evidence: for each field give its status, a verbatim quote (empty when there is no value; may be
empty for list fields, whose items carry their own quotes) and a confidence.

{CONFIDENCE_RUBRIC}
"""

DOCUMENT_TEMPLATE: Final = "<document>\n{document}\n</document>"

RETRY_TEMPLATE: Final = """\
An automatic validator rejected your previous output (above) with these findings:

{findings}

Re-read the document and answer with only what changes:
- changed_fields lists every field whose value you change.
- record gives the complete new value of each field in changed_fields (a whole list for a list
  field). Every other field is ignored: give it its empty value ("" for text, [] for lists,
  null for policy_type and premium_amount, and any value for document_kind and
  exclusions_complete).
- evidence gives a new entry for every field whose value, status or quote changes.
Change only what the findings concern. Values must be what the document states: never change a
number so that a check passes. If the document itself is inconsistent, keep its numbers and
change nothing."""

REVIEWER_SYSTEM: Final = f"""\
You independently check a proposed extraction of an insurance policy declarations document
against the document itself. You did not produce the extraction and you have no access to how
it was produced.

The extraction follows these field definitions, and you must judge it by them, not by your own
idea of what a field should contain:
{render_definitions()}

General rules the extraction follows:
{GENERAL_RULES}

In the extraction, null means the document does not state the value.

Return exactly one verdict for every field listed above:
- agree: the value (or its absence) is what the document states under the definition.
- disagree: it is not. The reason must quote the document and say what the value should be.
- not_verifiable: the document is too unclear to decide either way. Say why.
For agree, the reason may be empty."""

REVIEW_TEMPLATE: Final = """\
<document>
{document}
</document>

<proposed_extraction>
{extraction}
</proposed_extraction>"""
