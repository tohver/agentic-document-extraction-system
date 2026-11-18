"""Outcomes and deterministic routing (CLAUDE.md §6, §7).

`route` is a pure function of explicit signals. Model confidence is one signal and can never
approve a record on its own: the reviewer (a disagreement or a field it could not verify),
unresolved findings, the policy type and a partial exclusions list all have a veto. Every signal
that fires is listed by name in `Decision.reasons`.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel

from policy_extractor.config import Settings
from policy_extractor.models import ALL_FIELDS, PolicyExtraction, PolicyType, ReviewOutput, Verdict
from policy_extractor.validate import Category, Finding


class Outcome(StrEnum):
    AUTO_APPROVE = "auto_approve"
    SPOT_CHECK = "spot_check"
    HUMAN_REVIEW = "human_review"
    ESCALATED = "escalated"


class Escalation(StrEnum):
    """Why a document was escalated. Processing failures are never labelled missing_source."""

    NOT_A_POLICY = "not_a_policy"
    MISSING_SOURCE = "missing_source"
    RETRIES_EXHAUSTED = "retries_exhausted"
    MODEL_REFUSAL = "model_refusal"
    TRUNCATED = "truncated"
    API_ERROR = "api_error"
    BATCH_ITEM_FAILED = "batch_item_failed"


class Decision(BaseModel):
    outcome: Outcome
    escalation: Escalation | None = None
    reasons: list[str]
    # The reviewer's own words for every field it disputed, so a person sees why.
    reviewer_reasons: dict[str, str] = {}


def route(
    extraction: PolicyExtraction,
    findings: Sequence[Finding],
    review: ReviewOutput | None,
    settings: Settings,
) -> Decision:
    """Approve only when no signal fires. `review=None` means the review produced no verdicts."""
    reasons: list[str] = []
    reviewer_reasons: dict[str, str] = {}
    for field in ALL_FIELDS:
        evidence = extraction.evidence_for(field)
        if evidence is None:
            reasons.append(f"missing_confidence:{field}")
        elif evidence.confidence < settings.confidence_threshold:
            reasons.append(f"low_confidence:{field}")
    if review is None:
        reasons.append("reviewer_unavailable")
    else:
        for field in ALL_FIELDS:
            verdict = review.verdict_for(field)
            if verdict is None:
                reasons.append(f"reviewer_no_verdict:{field}")
            elif verdict.verdict is Verdict.DISAGREE:
                reasons.append(f"reviewer_disagreement:{field}")
                reviewer_reasons[field] = verdict.reason
            elif verdict.verdict is Verdict.NOT_VERIFIABLE:
                # "The document is too unclear to decide" is not a confirmation.
                reasons.append(f"reviewer_not_verifiable:{field}")
                reviewer_reasons[field] = verdict.reason
    for finding in findings:
        prefix = "document_discrepancy" if finding.document_discrepancy else "unresolved"
        reasons.append(f"{prefix}_{finding.category.value}:{finding.code}:{finding.field}")
    record = extraction.record
    if record.policy_type is PolicyType.OTHER:
        reasons.append("policy_type_other")
    if not record.exclusions_complete:
        reasons.append("exclusions_partial")
    unique = list(dict.fromkeys(reasons))
    return Decision(
        outcome=Outcome.HUMAN_REVIEW if unique else Outcome.AUTO_APPROVE,
        reasons=unique,
        reviewer_reasons=reviewer_reasons,
    )


def escalate(category: Escalation, findings: Sequence[Finding] = (), detail: str = "") -> Decision:
    """The pipeline cannot produce a record a person could approve; a person must act."""
    reasons = [f"{f.category.value}:{f.code}:{f.field}" for f in findings]
    if category is Escalation.RETRIES_EXHAUSTED:
        reasons = [r for r in reasons if r.startswith(Category.FORMAT.value)] or reasons
    if detail:
        reasons.append(detail)
    return Decision(outcome=Outcome.ESCALATED, escalation=category, reasons=reasons)
