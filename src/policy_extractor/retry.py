"""Validation-driven retry (CLAUDE.md §6).

The finding category decides the next step:
- `not_a_policy`, `missing_source`: escalate at once. The information is not in the document, so
  asking again can only invite invention.
- `format`, `grounding`: retry with the findings, up to `max_retries`. A format problem that
  survives every retry escalates as `retries_exhausted`; a grounding problem goes to review.
- `consistency`: retry once. If the same mismatch comes back with the same numbers, the document
  itself is inconsistent: keep its values, flag the discrepancy, and let routing send it to
  review.

A retry answers with a patch (only the fields it changes), applied to the previous output. It
may change only fields that a format or grounding finding named. Every other change against the
first reading, such as dropping a premium line so the components add up, is a `changed_on_retry`
grounding finding: a retry never gets to "fix" numbers the document states.

Whatever happens, the full finding history and every call are returned. Nothing ends in a silent
pass, and an API error mid-way keeps what was already extracted and paid for.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from policy_extractor.config import Settings
from policy_extractor.costs import CallRecord, record_call
from policy_extractor.extract import extract, extract_patch, extraction_request, retry_request
from policy_extractor.llm import STOP_UNPARSEABLE, LLMCallError, LLMClient, LLMRequest
from policy_extractor.models import PolicyExtraction, apply_patch
from policy_extractor.route import Escalation
from policy_extractor.validate import (
    CHANGED_ON_RETRY,
    Category,
    Finding,
    changed_on_retry,
    validate,
)

Status = Literal["ok", "escalated"]


@dataclass
class ExtractionOutcome:
    """`ok`: a record to review (maybe with unresolved findings). `escalated`: see `escalation`."""

    status: Status
    extraction: PolicyExtraction | None
    findings: list[Finding]
    escalation: Escalation | None = None
    history: list[list[Finding]] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    failure: str | None = None
    # The API's own message for an api_error, so the run log says what was wrong, not only that.
    failure_detail: str | None = None


def failure_category(stop_reason: str | None) -> Escalation:
    if stop_reason == "refusal":
        return Escalation.MODEL_REFUSAL
    if stop_reason is not None and stop_reason.startswith("batch_"):
        return Escalation.BATCH_ITEM_FAILED
    if stop_reason in ("max_tokens", STOP_UNPARSEABLE):
        return Escalation.TRUNCATED
    return Escalation.API_ERROR


def _attempt(
    client: LLMClient,
    request: LLMRequest,
    previous: PolicyExtraction | None,
    settings: Settings,
    calls: list[CallRecord],
) -> tuple[PolicyExtraction | None, str | None]:
    """One model call: the first reading, or a patch over `previous`. Returns the extraction (None
    if the call produced nothing usable) and the stop reason."""
    if previous is None:
        response = extract(client, request)
        calls.append(record_call(request, response, settings))
        return (None if response.failed else response.parsed), response.stop_reason
    patch = extract_patch(client, request)
    calls.append(record_call(request, patch, settings))
    if patch.failed or patch.parsed is None:
        return None, patch.stop_reason
    return apply_patch(previous, patch.parsed), patch.stop_reason


def run_extraction(client: LLMClient, document_text: str, settings: Settings) -> ExtractionOutcome:
    request = extraction_request(document_text, settings)
    history: list[list[Finding]] = []
    calls: list[CallRecord] = []
    baseline: PolicyExtraction | None = None
    previous: PolicyExtraction | None = None
    consistency_retried = False
    retries = 0
    while True:
        try:
            extraction, stop_reason = _attempt(client, request, previous, settings, calls)
        except LLMCallError as error:
            return ExtractionOutcome(
                "escalated",
                previous,
                history[-1] if history else [],
                Escalation.API_ERROR,
                history,
                calls,
                failure=f"api_error:{error.kind}",
                failure_detail=str(error),
            )
        if extraction is None:
            return ExtractionOutcome(
                status="escalated",
                extraction=previous,
                findings=history[-1] if history else [],
                escalation=failure_category(stop_reason),
                history=history,
                calls=calls,
                failure=stop_reason or "no_output",
            )
        findings = validate(extraction, document_text, settings)
        if baseline is not None:
            findings = _with_drift(
                findings, changed_on_retry(baseline, extraction, _allowed(history))
            )
        findings = _mark_discrepancies(findings, history)
        history.append(findings)
        categories = {f.category for f in findings}
        for terminal, escalation in (
            (Category.NOT_A_POLICY, Escalation.NOT_A_POLICY),
            (Category.MISSING_SOURCE, Escalation.MISSING_SOURCE),
        ):
            if terminal in categories:
                return ExtractionOutcome(
                    "escalated", extraction, findings, escalation, history, calls
                )
        if not _worth_retrying(findings, consistency_retried):
            return ExtractionOutcome("ok", extraction, findings, None, history, calls)
        if retries >= settings.max_retries:
            if Category.FORMAT in categories:
                return ExtractionOutcome(
                    "escalated", extraction, findings, Escalation.RETRIES_EXHAUSTED, history, calls
                )
            return ExtractionOutcome("ok", extraction, findings, None, history, calls)
        consistency_retried |= Category.CONSISTENCY in categories
        retries += 1
        baseline = baseline or extraction
        previous = extraction
        request = retry_request(document_text, extraction, findings, settings)


def _allowed(history: Sequence[Sequence[Finding]]) -> set[str]:
    """Fields a retry may change: those named by a format or grounding finding so far."""
    return {
        f.field
        for attempt in history
        for f in attempt
        if f.category in (Category.FORMAT, Category.GROUNDING) and f.code != CHANGED_ON_RETRY
    }


def _with_drift(findings: Sequence[Finding], drift: Sequence[Finding]) -> list[Finding]:
    """Add the drift findings. A field the first reading had and the retry dropped is drift, not
    information missing from the document, so it must not escalate as `missing_source`."""
    drifted = {f.field for f in drift}
    kept = [
        f for f in findings if not (f.category is Category.MISSING_SOURCE and f.field in drifted)
    ]
    return [*kept, *drift]


def _worth_retrying(findings: Sequence[Finding], consistency_retried: bool) -> bool:
    return any(
        f.category in (Category.FORMAT, Category.GROUNDING)
        or (
            f.category is Category.CONSISTENCY
            and not consistency_retried
            and not f.document_discrepancy
        )
        for f in findings
    )


def _mark_discrepancies(
    findings: Sequence[Finding], history: Sequence[Sequence[Finding]]
) -> list[Finding]:
    """A consistency mismatch that any earlier attempt reported with identical values is the
    document's, not ours."""
    seen = {
        (f.field, f.code, f.observed)
        for attempt in history
        for f in attempt
        if f.category is Category.CONSISTENCY
    }
    return [
        f.model_copy(update={"document_discrepancy": True})
        if f.category is Category.CONSISTENCY and (f.field, f.code, f.observed) in seen
        else f
        for f in findings
    ]
