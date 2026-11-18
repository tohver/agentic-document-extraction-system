# Dataset

22 synthetic insurance documents in `policies/` and one hand-written gold label per document in
`gold/`. All carriers, people, addresses and numbers are fictional.

- `POL-2025-001` to `010` are the seed set from an earlier prototype.
- `POL-2025-011` to `022` were added to cover second layouts per policy type and the traps below.

Each gold label holds the values a careful person would extract under the field definitions in
`src/policy_extractor/fields.py` (same schema as the extraction, without quotes and confidence),
the expected routing outcome, and a one-line `why`. Gold was written from the documents
themselves, not from the prototype's old `MANIFEST.json`, which was internally inconsistent.

## Expected outcomes

| Outcome | Documents |
|---|---|
| auto_approve | 001, 002, 004, 005, 006, 007, 011, 012, 013, 018, 021, 022 |
| human_review | 003 (partial exclusions), 008 and 014 (specialty product), 010 and 019 (document discrepancy) |
| escalated: missing_source | 009 and 020 (endorsement schedule not attached), 015 (no premium) |
| escalated: not_a_policy | 016 (claim letter), 017 (mid-term change notice) |

## Traps

| Trap | Document | What the pipeline must do |
|---|---|---|
| Premium components that do not add up (2,350.00 vs 2,400.00) | 010 | Keep the document's numbers, flag a document discrepancy, route to review |
| Expiration date equal to the effective date | 019 | Same: a consistency finding that survives a retry unchanged |
| Endorsements referenced to an attachment that is missing | 009, 020 | Escalate as missing_source after one call |
| Premium not stated | 015 | Escalate as missing_source after one call |
| Not a declarations page | 016, 017 | Escalate as not_a_policy |
| Split limits, "in thousands", on one line only | 003 | 50/100 on the next line is also in thousands |
| Installment amount next to the premium | 003, 021 | The premium is the term total |
| Partial exclusions list ("partial list, see policy form PA-100") | 003 | exclusions_complete false, route to review |
| Percentage deductibles, with and without the computed amount | 004, 005, 012 | percent and percent_of; a stated amount must match the computation |
| Deductible column per coverage row | 004 | One deductible item per row |
| Separate comprehensive and collision deductibles, none labelled | 002, 009, 011, 021 | Primary deductible is ambiguous: None, not a guess |
| Primary deductible on its own line, or as a heading | 001, 019 | labelled_primary |
| Self-insured retention on umbrellas | 006, 007, 013, 020 | A deductible; the umbrella's primary deductible |
| Underlying insurance requirements | 006, 007, 013, 020 | Not limits of this policy |
| Specialty products | 008 (fine-art floater), 014 (watercraft) | policy_type other with detail |
| Carrier: underwriting brand in the header vs "Underwritten by" | 008 | The company the policy is underwritten by |
| Spaced policy number ("POL - 2025 - 007") | 007 | Spaces removed |
| "File generated" date | 010 | Not the effective date |
| OCR character noise (0 for O, 1 and l for l and I) | 011 | Read through it |
| Prose layout, dates in words | 012, 021 | Same fields, ISO dates |
| "$1M" endorsement limit | 006 | 1000000 |
| Trust as named insured | 008, 022 | As written |

## Gold-label decisions worth knowing

Two labels were corrected after the first evaluation showed the model was right and the label
was not: POL-2025-004's deductible column gives one item per coverage row (the definitions
already said so), and POL-2025-008's carrier is the company named in "Underwritten by", not the
underwriting brand in the header. The other disagreements in that run were extractor errors and
stayed as labelled. Because the same 22 documents were used to tune the field definitions, the
reported accuracy is an optimistic estimate; a held-out set is the next step.
