"""A retry answers with a patch, may change only what a finding named, and an API error mid-way
keeps what was already extracted."""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from policy_extractor.llm import LLMCallError, LLMRequest, LLMResponse
from policy_extractor.models import (
    ExtractionPatch,
    FieldName,
    PremiumComponent,
    apply_patch,
)
from policy_extractor.retry import run_extraction
from policy_extractor.route import Escalation
from tests.fakes import DOCUMENT, FakeClient, make_extraction, patch, respond, settings

SETTINGS = settings()
PREMIUM_LINES = "Base $ 1,000.00\nFee $ 200.00\nCredit $ (100.00)\n"
BASE = PremiumComponent(name="Base", amount=Decimal("1000.00"), quote="Base $ 1,000.00")
FEE = PremiumComponent(name="Fee", amount=Decimal("200.00"), quote="Fee $ 200.00")
CREDIT = PremiumComponent(name="Credit", amount=Decimal("-100.00"), quote="Credit $ (100.00)")


def codes(findings) -> set[tuple[str, str, str]]:
    return {(f.category.value, f.code, f.field) for f in findings}


def test_a_patch_changes_only_the_fields_it_names():
    previous = make_extraction(effective_date="01/01/2026")
    merged = apply_patch(previous, patch(effective_date="2026-01-01"))
    assert merged == make_extraction()


def test_fields_outside_changed_fields_are_ignored_and_an_empty_change_applies():
    # The filler (empty lists, exclusions_complete false) must not leak into the record.
    unchanged = apply_patch(make_extraction(), patch())
    assert unchanged == make_extraction()
    cleared = apply_patch(make_extraction(), patch(policy_form="", endorsements=[]))
    assert (cleared.record.policy_form, cleared.record.endorsements) == ("", [])
    flipped = apply_patch(
        make_extraction(),
        ExtractionPatch(
            changed_fields=[FieldName.EXCLUSIONS_COMPLETE],
            record=patch().record,
            evidence=[],
        ),
    )
    assert flipped.record.exclusions_complete is False


def test_a_retry_answer_is_a_patch_over_the_previous_output():
    bad = make_extraction(effective_date="01/01/2026")
    client = FakeClient([respond(bad), respond(patch(effective_date="2026-01-01"))])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert (outcome.status, outcome.findings) == ("ok", [])
    assert outcome.extraction == make_extraction()
    assert "changed_fields" in client.requests[1].messages[2].content


def test_dropping_a_premium_line_so_the_total_adds_up_is_caught_not_accepted():
    document = DOCUMENT + PREMIUM_LINES
    stated = make_extraction(premium_components=[BASE, FEE, CREDIT])  # sums to 1100, not 1200
    dropped = patch(premium_components=[BASE, FEE])  # now sums to 1200
    client = FakeClient([respond(stated), respond(dropped), respond(dropped)])

    outcome = run_extraction(client, document, SETTINGS)

    assert outcome.status == "ok"  # routed to a person, not escalated
    assert ("grounding", "changed_on_retry", "premium_components") in codes(outcome.findings)
    assert ("consistency", "components_sum_mismatch", "premium_amount") in codes(outcome.history[0])


def test_a_retry_that_restores_the_documents_values_ends_as_a_document_discrepancy():
    document = DOCUMENT + PREMIUM_LINES
    stated = make_extraction(premium_components=[BASE, FEE, CREDIT])
    client = FakeClient(
        [
            respond(stated),
            respond(patch(premium_components=[BASE, FEE])),
            respond(patch(premium_components=[BASE, FEE, CREDIT])),
        ]
    )

    outcome = run_extraction(client, document, SETTINGS)

    [finding] = outcome.findings
    assert (finding.code, finding.document_discrepancy) == ("components_sum_mismatch", True)
    assert outcome.extraction is not None
    assert len(outcome.extraction.record.premium_components) == 3


def test_a_value_dropped_on_retry_is_drift_not_missing_source():
    bad = make_extraction(effective_date="01/01/2026")
    client = FakeClient(
        [
            respond(bad),
            respond(patch(effective_date="2026-01-01", premium_amount=None)),
            respond(patch(premium_amount=Decimal("1200.00"))),
        ]
    )

    outcome = run_extraction(client, DOCUMENT, SETTINGS)

    assert (outcome.status, outcome.escalation, outcome.findings) == ("ok", None, [])
    assert ("grounding", "changed_on_retry", "premium_amount") in codes(outcome.history[1])
    assert "missing_source" not in {f.category.value for f in outcome.history[1]}


class FailsOnRetry:
    """The first call answers; the retry hits an API error the SDK could not absorb."""

    def __init__(self, first: BaseModel) -> None:
        self.first = first
        self.calls = 0

    def parse(self, request: LLMRequest, output_type: type[BaseModel]) -> LLMResponse[Any]:
        self.calls += 1
        if self.calls == 1:
            return respond(self.first)
        raise LLMCallError("rate_limited", "429 after SDK retries")


def test_an_api_error_on_a_retry_keeps_the_first_reading_and_its_cost():
    bad = make_extraction(effective_date="01/01/2026")
    outcome = run_extraction(FailsOnRetry(bad), DOCUMENT, SETTINGS)
    assert (outcome.status, outcome.escalation) == ("escalated", Escalation.API_ERROR)
    assert outcome.failure == "api_error:rate_limited"
    assert outcome.failure_detail == "rate_limited: 429 after SDK retries"
    assert outcome.extraction == bad
    assert [call.role for call in outcome.calls] == ["extractor"]
    assert [f.code for f in outcome.findings] == ["date_not_iso"]
