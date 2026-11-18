"""Orchestration for one document and for a whole run (CLAUDE.md §3).

Data flow: extract -> derive -> validate -> (retry | escalate) -> review -> route, then
spot-check sampling over the run. Extraction and review are separate phases so the Batches path
can run all extractions first and all reviews second. An infrastructure error on one document
becomes an `api_error` escalation for that document; the run continues.
"""

from collections import Counter
from collections.abc import Sequence
from decimal import Decimal

from pydantic import BaseModel

from policy_extractor.config import Settings
from policy_extractor.costs import CallRecord, record_call
from policy_extractor.derive import Derived, derive
from policy_extractor.llm import LLMCallError, LLMClient
from policy_extractor.models import PolicyExtraction, ReviewOutput
from policy_extractor.retry import ExtractionOutcome, run_extraction
from policy_extractor.review import review, review_request
from policy_extractor.route import Decision, Escalation, Outcome, escalate, route
from policy_extractor.runlog import RunLog
from policy_extractor.sampling import select_spot_checks
from policy_extractor.validate import Finding


class DocumentResult(BaseModel):
    """Everything known about one document, and why it ended as it did."""

    doc_id: str
    policy_type: str | None
    decision: Decision
    extraction: PolicyExtraction | None
    derived: Derived | None
    findings: list[Finding]
    finding_history: list[list[Finding]]
    review: ReviewOutput | None
    calls: list[CallRecord]


def extraction_phase(
    client: LLMClient, doc_id: str, document_text: str, settings: Settings, log: RunLog
) -> ExtractionOutcome:
    log.event("document_started", doc_id=doc_id, characters=len(document_text))
    outcome = run_extraction(client, document_text, settings)
    for attempt, call in enumerate(outcome.calls):
        log.event("llm_call", doc_id=doc_id, attempt=attempt, call=call)
    for attempt, findings in enumerate(outcome.history):
        log.event("validation", doc_id=doc_id, attempt=attempt, findings=findings)
    log.event(
        "extraction_finished",
        doc_id=doc_id,
        status=outcome.status,
        escalation=outcome.escalation,
        attempts=len(outcome.calls),
        failure=outcome.failure,
        failure_detail=outcome.failure_detail,
    )
    return outcome


def review_phase(
    client: LLMClient,
    doc_id: str,
    document_text: str,
    outcome: ExtractionOutcome,
    settings: Settings,
    log: RunLog,
) -> DocumentResult:
    calls = list(outcome.calls)
    review_output: ReviewOutput | None = None
    extraction = outcome.extraction
    if outcome.status == "escalated" or extraction is None:
        decision = escalate(
            outcome.escalation or Escalation.API_ERROR,
            outcome.findings,
            ""
            if not outcome.failure
            else outcome.failure
            if outcome.failure.startswith("api_error:")
            else f"stop_reason:{outcome.failure}",
        )
    else:
        request = review_request(document_text, extraction, settings)
        review_error: LLMCallError | None = None
        try:
            response = review(client, request)
        except LLMCallError as error:
            # The extraction is valid and paid for: keep it and let a person do the review.
            review_error = error
            log.event("review_error", doc_id=doc_id, error=error.kind, message=str(error))
        else:
            call = record_call(request, response, settings)
            calls.append(call)
            log.event("llm_call", doc_id=doc_id, attempt=0, call=call)
            review_output = None if response.failed else response.parsed
            log.event(
                "review", doc_id=doc_id, review=review_output, stop_reason=response.stop_reason
            )
        decision = route(extraction, outcome.findings, review_output, settings)
        if review_error is not None:
            decision = decision.model_copy(
                update={"reasons": [*decision.reasons, f"reviewer_api_error:{review_error.kind}"]}
            )
    log.event(
        "decision",
        doc_id=doc_id,
        outcome=decision.outcome,
        escalation=decision.escalation,
        reasons=decision.reasons,
    )
    return DocumentResult(
        doc_id=doc_id,
        policy_type=_policy_type(extraction),
        decision=decision,
        extraction=extraction,
        derived=None if extraction is None else derive(extraction.values()),
        findings=outcome.findings,
        finding_history=outcome.history,
        review=review_output,
        calls=calls,
    )


def _policy_type(extraction: PolicyExtraction | None) -> str | None:
    if extraction is None or extraction.record.policy_type is None:
        return None
    return extraction.record.policy_type.value


def process_document(
    client: LLMClient, doc_id: str, document_text: str, settings: Settings, log: RunLog
) -> DocumentResult:
    """Never raises for an API failure. A failure during extraction becomes this document's
    `api_error` escalation (with the calls already made); a failure of the review call sends the
    extracted record to human review. The `except` below is the last line of defence."""
    try:
        outcome = extraction_phase(client, doc_id, document_text, settings, log)
        return review_phase(client, doc_id, document_text, outcome, settings, log)
    except LLMCallError as error:
        log.event("document_error", doc_id=doc_id, error=error.kind, message=str(error))
        return api_error_result(doc_id, error)


def api_error_result(doc_id: str, error: LLMCallError) -> DocumentResult:
    return DocumentResult(
        doc_id=doc_id,
        policy_type=None,
        decision=escalate(Escalation.API_ERROR, detail=f"api_error:{error}"),
        extraction=None,
        derived=None,
        findings=[],
        finding_history=[],
        review=None,
        calls=[],
    )


def apply_spot_checks(
    results: Sequence[DocumentResult], settings: Settings, log: RunLog
) -> list[DocumentResult]:
    """Promote a per-policy-type sample of auto-approved records to `spot_check`."""
    approved = {
        r.doc_id: r.policy_type for r in results if r.decision.outcome is Outcome.AUTO_APPROVE
    }
    selected = select_spot_checks(approved, settings.spot_check_share, settings.sampling_seed)
    for doc_id in sorted(selected):
        log.event("spot_check", doc_id=doc_id, policy_type=approved[doc_id])
    return [
        r.model_copy(
            update={"decision": Decision(outcome=Outcome.SPOT_CHECK, reasons=["spot_check_sample"])}
        )
        if r.doc_id in selected
        else r
        for r in results
    ]


class RunSummary(BaseModel):
    documents: int
    outcomes: dict[str, int]
    escalations: dict[str, int]
    finding_codes: dict[str, int]
    calls: int
    retries: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: Decimal
    unpriced_calls: int


def summarise(results: Sequence[DocumentResult]) -> RunSummary:
    calls = [call for result in results for call in result.calls]
    codes: Counter[str] = Counter(
        f"{f.category.value}:{f.code}"
        for result in results
        for attempt in result.finding_history
        for f in attempt
    )
    return RunSummary(
        documents=len(results),
        outcomes={o.value: sum(r.decision.outcome is o for r in results) for o in Outcome},
        escalations={e.value: sum(r.decision.escalation is e for r in results) for e in Escalation},
        finding_codes=dict(sorted(codes.items())),
        calls=len(calls),
        retries=sum(max(0, sum(c.role == "extractor" for c in r.calls) - 1) for r in results),
        input_tokens=sum(c.usage.input_tokens for c in calls),
        output_tokens=sum(c.usage.output_tokens for c in calls),
        cache_read_input_tokens=sum(c.usage.cache_read_input_tokens for c in calls),
        cache_creation_input_tokens=sum(c.usage.cache_creation_input_tokens for c in calls),
        cost_usd=sum((c.cost_usd for c in calls if c.cost_usd is not None), Decimal(0)),
        unpriced_calls=sum(c.cost_usd is None for c in calls),
    )


def render_summary(summary: RunSummary, results: Sequence[DocumentResult]) -> str:
    per_document = summary.cost_usd / summary.documents if summary.documents else Decimal(0)
    escalations = {k: v for k, v in summary.escalations.items() if v}
    lines = [
        "# Run summary",
        "",
        f"- Documents: {summary.documents}",
        *(f"- {name}: {count}" for name, count in summary.outcomes.items()),
        *(f"  - escalated as {name}: {count}" for name, count in escalations.items()),
        f"- Model calls: {summary.calls} (extraction retries: {summary.retries})",
        f"- Tokens: {summary.input_tokens} input, {summary.output_tokens} output, "
        f"{summary.cache_read_input_tokens} cache read, "
        f"{summary.cache_creation_input_tokens} cache write",
        f"- Cost: ${summary.cost_usd:.4f} total, ${per_document:.4f} per document"
        + (
            f" ({summary.unpriced_calls} calls had no configured price)"
            if summary.unpriced_calls
            else ""
        ),
        "",
        "| Document | Type | Outcome | Reasons |",
        "|---|---|---|---|",
        *(
            f"| {r.doc_id} | {r.policy_type or '-'} | "
            f"{r.decision.outcome.value}"
            f"{f' ({r.decision.escalation.value})' if r.decision.escalation else ''} | "
            f"{', '.join(r.decision.reasons) or '-'} |"
            for r in results
        ),
        "",
    ]
    return "\n".join(lines)
