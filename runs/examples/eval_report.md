# Evaluation report

- Documents: 22 (20 with an extraction scored field by field)
- Routing outcome as expected: 17/22
- Escalation category as expected: 5/5
- Auto-approved: 7; auto-approved with a wrong field: 0
- Field accuracy overall: 99.7%; on fields the document states: 99.6%; mean confidence 0.96; Brier 0.005
- Cost: $0.8099 ($0.0368 per document); tokens: 50427 input, 49713 output

## Routing: expected vs actual

| Expected \ Actual | auto_approve | human_review | escalated |
|---|---|---|---|
| auto_approve | 7 | 5 | 0 |
| human_review | 0 | 5 | 0 |
| escalated | 0 | 0 | 5 |

## Field accuracy

"Where stated" counts only documents whose gold says the field has a value; fields that are empty by construction are easy to get right and inflate the plain figure.

| Field | Accuracy | Where stated | n stated |
|---|---|---|---|
| document_kind | 100.0% | 100.0% | 20 |
| policy_number | 100.0% | 100.0% | 20 |
| carrier | 100.0% | 100.0% | 20 |
| policyholder | 100.0% | 100.0% | 20 |
| policy_type | 100.0% | 100.0% | 20 |
| policy_type_detail | 100.0% | 100.0% | 2 |
| policy_form | 95.0% | 92.3% | 13 |
| effective_date | 100.0% | 100.0% | 20 |
| expiration_date | 100.0% | 100.0% | 20 |
| premium_amount | 100.0% | 100.0% | 19 |
| premium_components | 100.0% | 100.0% | 2 |
| deductibles | 100.0% | 100.0% | 20 |
| coverage_limits | 100.0% | 100.0% | 20 |
| underlying_requirements | 100.0% | 100.0% | 4 |
| endorsements | 100.0% | 100.0% | 18 |
| exclusions | 100.0% | 100.0% | 20 |
| exclusions_complete | 100.0% | 100.0% | 20 |

## Derived fields (computed in Python from extracted vs gold values)

| Derived field | Accuracy |
|---|---|
| primary_limit | 100.0% |
| primary_deductible | 100.0% |
| term_months | 100.0% |
| percentage_deductibles | 100.0% |

## Reviewer value

| Extractor field was | Reviewer agreed | Reviewer disagreed |
|---|---|---|
| correct | 287 | 0 (false dispute) |
| wrong | 0 (missed) | 1 (caught) |

Not verifiable: 1.

## Calibration by policy type x field (flag: confidence exceeds accuracy by 0.15 or more)

| Policy type | Field | n | Mean confidence | Accuracy | Brier | Flag |
|---|---|---|---|---|---|---|
| auto | policy_form | 8 | 0.94 | 0.88 | 0.104 |  |

Cells with every field correct are omitted from this table.

## Per document

| Document | Expected | Actual | Wrong fields |
|---|---|---|---|
| POL-2025-001 | auto_approve | auto_approve | - |
| POL-2025-002 | auto_approve | human_review | - |
| POL-2025-003 | human_review | human_review | policy_form |
| POL-2025-004 | auto_approve | human_review | - |
| POL-2025-005 | auto_approve | human_review | - |
| POL-2025-006 | auto_approve | human_review | - |
| POL-2025-007 | auto_approve | human_review | - |
| POL-2025-008 | human_review | human_review | - |
| POL-2025-009 | escalated (missing_source) | escalated (missing_source) | - |
| POL-2025-010 | human_review | human_review | - |
| POL-2025-011 | auto_approve | auto_approve | - |
| POL-2025-012 | auto_approve | auto_approve | - |
| POL-2025-013 | auto_approve | auto_approve | - |
| POL-2025-014 | human_review | human_review | - |
| POL-2025-015 | escalated (missing_source) | escalated (missing_source) | - |
| POL-2025-016 | escalated (not_a_policy) | escalated (not_a_policy) | - |
| POL-2025-017 | escalated (not_a_policy) | escalated (not_a_policy) | - |
| POL-2025-018 | auto_approve | auto_approve | - |
| POL-2025-019 | human_review | human_review | - |
| POL-2025-020 | escalated (missing_source) | escalated (missing_source) | - |
| POL-2025-021 | auto_approve | auto_approve | - |
| POL-2025-022 | auto_approve | auto_approve | - |
