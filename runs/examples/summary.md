# Run summary

- Documents: 22
- auto_approve: 4
- spot_check: 3
- human_review: 10
- escalated: 5
  - escalated as not_a_policy: 2
  - escalated as missing_source: 3
- Model calls: 45 (extraction retries: 6)
- Tokens: 50427 input, 49713 output, 223272 cache read, 0 cache write
- Cost: $0.8099 total, $0.0368 per document

| Document | Type | Outcome | Reasons |
|---|---|---|---|
| POL-2025-001 | auto | auto_approve | - |
| POL-2025-002 | auto | human_review | low_confidence:coverage_limits |
| POL-2025-003 | auto | human_review | low_confidence:carrier, low_confidence:coverage_limits, reviewer_disagreement:policy_form, exclusions_partial |
| POL-2025-004 | home | human_review | low_confidence:deductibles |
| POL-2025-005 | home | human_review | low_confidence:deductibles, low_confidence:exclusions_complete |
| POL-2025-006 | umbrella | human_review | low_confidence:carrier |
| POL-2025-007 | umbrella | human_review | low_confidence:carrier, low_confidence:effective_date, low_confidence:expiration_date, low_confidence:endorsements |
| POL-2025-008 | other | human_review | low_confidence:coverage_limits, policy_type_other |
| POL-2025-009 | auto | escalated (missing_source) | missing_source:referenced_not_attached:endorsements |
| POL-2025-010 | home | human_review | reviewer_not_verifiable:premium_components, document_discrepancy_consistency:components_sum_mismatch:premium_amount |
| POL-2025-011 | auto | spot_check | spot_check_sample |
| POL-2025-012 | home | auto_approve | - |
| POL-2025-013 | umbrella | spot_check | spot_check_sample |
| POL-2025-014 | other | human_review | policy_type_other |
| POL-2025-015 | auto | escalated (missing_source) | missing_source:premium_amount_absent:premium_amount |
| POL-2025-016 | - | escalated (not_a_policy) | not_a_policy:not_a_policy:document_kind |
| POL-2025-017 | - | escalated (not_a_policy) | not_a_policy:not_a_policy:document_kind |
| POL-2025-018 | home | auto_approve | - |
| POL-2025-019 | auto | human_review | low_confidence:expiration_date, document_discrepancy_consistency:term_not_positive:expiration_date |
| POL-2025-020 | umbrella | escalated (missing_source) | missing_source:referenced_not_attached:endorsements |
| POL-2025-021 | auto | auto_approve | - |
| POL-2025-022 | home | spot_check | spot_check_sample |
