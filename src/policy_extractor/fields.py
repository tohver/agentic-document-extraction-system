"""What every field means: the single source of truth for both prompts (CLAUDE.md §4, §5).

The extractor and the reviewer read the same definitions. In the prototype the reviewer never saw
how a field was defined and disputed correct values; sharing this text is the fix.
"""

from typing import Final

GENERAL_RULES: Final = """\
- Extract only what the document states. If a field is not stated, give it no value ("" for text,
  [] for lists, null for policy_type and premium_amount) and set its evidence status to
  not_stated. Never guess, infer a default, or use today's date.
- If the document points to an attachment that is not included (for example "see attached
  Schedule A" and the schedule is absent), give the field no value and set its status to
  referenced_not_attached.
- If the value is present but unreadable, give it no value and set its status to illegible.
- Money and percentages are decimal strings without currency symbols or thousands separators:
  "$ 1,847.62" -> "1847.62". Discounts shown in parentheses are negative: "$ (100.00)" -> "-100.00".
- Dates are ISO YYYY-MM-DD. "03/15/2025" -> "2025-03-15".
- A quote is text copied verbatim from the document (whitespace may differ). It must contain the
  number or words the value was read from. Keep quotes short: one line or one clause.
- Do not extract addresses, vehicles, VINs, garage ZIP codes, agents or producers, payment plans,
  scheduled art items, or underwriting offices."""

CONFIDENCE_RUBRIC: Final = """\
Confidence is the probability that the field is exactly right, including its status.
- 0.95-1.00: the document states it plainly in one place; no interpretation needed.
- 0.80-0.94: you had to interpret layout, OCR noise, or choose between candidates.
- below 0.80: you are unsure; give the low number rather than an optimistic one.
A field with status not_stated is "right" if the document really says nothing about it."""

FIELD_DEFINITIONS: Final[dict[str, str]] = {
    "document_kind": (
        "policy_declarations for an insurance policy declarations page or renewal declarations "
        "/ renewal offer that states the policy's terms. Anything else (claim letters, billing "
        "statements, cancellation or change notices, marketing) is not_a_policy."
    ),
    "policy_number": (
        'The policy number as printed, with internal spaces removed: "POL - 2025 - 007" -> '
        '"POL-2025-007".'
    ),
    "carrier": (
        "The insurance company named as insurer, usually in the header. Not the agent, producer, "
        "agency, broker or underwriting office. If the document names a different company as the "
        "one the policy is underwritten by ('Underwritten by: X Insurance Co.'), that company is "
        "the carrier."
    ),
    "policyholder": (
        "The named insured exactly as written, including joint insureds "
        '("Gregory & Patricia Lindholm") and trusts.'
    ),
    "policy_type": (
        "auto for personal auto, home for homeowners, umbrella for personal umbrella / excess "
        "liability, other for any specialty product (fine-art floater, watercraft, inland "
        "marine). Never force a specialty product into a standard type."
    ),
    "policy_type_detail": (
        'The product as written, e.g. "Fine Art Floater". Required when policy_type is other; '
        "otherwise empty."
    ),
    "policy_form": (
        "The policy form code if stated (HO-3, HO-5, FA-IM-22, UMB-PL-0922, MFD-AU-04). Empty if "
        "none is stated."
    ),
    "effective_date": ("Start of the policy term. Not an issue, print, or 'file generated' date."),
    "expiration_date": "End of the policy term.",
    "premium_amount": (
        "Total premium for the policy term. Not an installment amount (for example 'payable in "
        "monthly installments of $116.51' is not the premium)."
    ),
    "premium_components": (
        "Only when the document itemises the premium: one item per line that adds up to the "
        "stated premium, discounts negative. Empty if the premium is not itemised."
    ),
    "deductibles": (
        "One item per deductible as stated. Comprehensive and Collision are separate items even "
        "when equal. A percentage deductible ('5% of the Coverage A limit') sets percent and "
        "percent_of (the coverage it is a percentage of, e.g. 'Coverage A'); amount stays null "
        "unless the document also states the computed amount. 'NIL' or 'zero deductible' is "
        "amount 0. An umbrella self-insured retention (SIR / retained limit) is a deductible with "
        "applies_to 'self-insured retention'. labelled_primary is true only if the document itself "
        "calls this deductible primary, standard, base, or all other perils (AOP); a heading such "
        "as 'DEDUCTIBLES (Primary)' or a sentence such as 'Standard deductible applies at ...' "
        "labels every deductible it covers. A table with a deductible column gives one item per "
        "row that has a deductible. If the document "
        "states a primary or standard deductible on its own line in addition to per-coverage "
        "deductibles (e.g. 'DEDUCTIBLE (Primary): $500'), that line is its own item with "
        "applies_to 'primary'. A deductible stated once for several coverages ('$1,000 for both "
        "Comprehensive and Collision') is one item per coverage."
    ),
    "coverage_limits": (
        "Every limit of the policy's own coverages. coverage is the name as written (e.g. "
        "'Coverage A - Dwelling'). basis is per_person, per_accident, per_occurrence, aggregate, "
        "scheduled_total or other. Split limits become separate items: '100,000 / 300,000' for "
        "bodily injury -> per_person 100000 and per_accident 300000; '50/100 (in thousands)' -> "
        "50000 and 100000; a unit stated on one line of a table ('in thousands') applies to the "
        "other lines of that table written the same way, such as a following '50/100'. "
        "Use per_person, per_accident, per_occurrence or aggregate only when the "
        "document says so (or for the two parts of a split bodily-injury limit); a single limit "
        "with no stated basis (Coverage A-F, PIP, medical payments) is other. A coverage shown "
        "only as YES or INCLUDED, or as a deductible line without a limit, is not a limit item. "
        "A fine-art total scheduled value is scheduled_total. Valuation wording ('replacement "
        "cost', 'Agreed Value, All-Risk') goes in note. Underlying insurance an umbrella requires "
        "is not a coverage limit of this policy."
    ),
    "underlying_requirements": (
        "Umbrella only: each underlying insurance the insured must keep, with its limits as "
        "written ('250/500/100 BI/PD'). Empty for other policy types."
    ),
    "endorsements": (
        "Every endorsement listed. A code such as AU-21, HO-04-90 or UMB-12 goes in form_code, "
        "not in name. A plain money limit goes in limit_amount; any other limit ('$40/day, 30 day "
        "max', '25% / 90 days', '25% of Cov A') goes in limit_text. When limit_amount holds the "
        "whole limit, limit_text is empty. If the endorsements are only "
        "referenced to an attachment that is not included, the status is referenced_not_attached."
    ),
    "exclusions": (
        "One item per listed exclusion, text as written, joined into one line if it wraps."
    ),
    "exclusions_complete": (
        "false when the document says the exclusions list is partial (e.g. 'partial list, see "
        "policy form PA-100'); true otherwise. This answers only whether the document says so: "
        "if it contains no such statement, true is certain, even though a full policy form "
        "would list more exclusions. Its quote is the exclusions heading."
    ),
}

# Derivation rules (computed in Python by derive.py, never asked of the model).
PRIMARY_LIMIT_RULE: Final = {
    "auto": "bodily injury limit with basis per_accident",
    "home": "Coverage A (dwelling)",
    "umbrella": "limit with basis per_occurrence",
    "other": "scheduled_total, else the single aggregate limit",
}
PRIMARY_DEDUCTIBLE_RULE: Final = (
    "the deductible labelled primary; else the only deductible; else the common amount if all "
    "are equal; for umbrella the self-insured retention; otherwise ambiguous (None)"
)


def render_definitions() -> str:
    return "\n".join(f"- {name}: {text}" for name, text in FIELD_DEFINITIONS.items())
