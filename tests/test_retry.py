"""Retry and escalation (§6): what each finding category makes the pipeline do next."""

from decimal import Decimal

from policy_extractor.models import DocumentKind, FieldStatus, PremiumComponent
from policy_extractor.retry import run_extraction
from policy_extractor.route import Escalation
from policy_extractor.validate import Category
from tests.fakes import DOCUMENT, FakeClient, make_extraction, respond, settings

SETTINGS = settings()


def test_missing_premium_escalates_after_one_call():
    client = FakeClient([respond(make_extraction(premium_amount=None))])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert (outcome.status, outcome.escalation) == ("escalated", Escalation.MISSING_SOURCE)
    assert len(client.requests) == 1
    assert [f.code for f in outcome.findings] == ["premium_amount_absent"]


def test_a_referenced_but_missing_schedule_escalates_after_one_call():
    extraction = make_extraction(
        endorsements=[], statuses={"endorsements": FieldStatus.REFERENCED_NOT_ATTACHED}
    )
    client = FakeClient([respond(extraction)])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert outcome.escalation is Escalation.MISSING_SOURCE
    assert len(client.requests) == 1


def test_not_a_policy_escalates_after_one_call():
    extraction = make_extraction(document_kind=DocumentKind.NOT_A_POLICY, premium_amount=None)
    client = FakeClient([respond(extraction)])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert outcome.escalation is Escalation.NOT_A_POLICY
    assert len(client.requests) == 1


def test_a_format_problem_is_retried_with_the_previous_output_and_findings():
    bad = make_extraction(effective_date="01/01/2026")
    client = FakeClient([respond(bad), respond(make_extraction())])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)

    assert (outcome.status, outcome.findings) == ("ok", [])
    first, retry = client.requests
    assert len(retry.messages) == 3
    assert retry.messages[:1] == first.messages
    assert retry.messages[1].role == "assistant"
    assert retry.messages[1].content == bad.model_dump_json()
    assert "date_not_iso" in retry.messages[2].content
    assert [len(h) for h in outcome.history] == [1, 0]


def test_a_format_problem_that_survives_every_retry_escalates_as_retries_exhausted():
    bad = make_extraction(effective_date="01/01/2026")
    client = FakeClient([respond(bad)] * (SETTINGS.max_retries + 1))
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert (outcome.status, outcome.escalation) == ("escalated", Escalation.RETRIES_EXHAUSTED)
    assert len(client.requests) == SETTINGS.max_retries + 1


def test_a_grounding_problem_that_survives_retries_goes_to_review_not_escalation():
    bad = make_extraction(quotes={"carrier": "TESTLINE CASUALTY"})
    client = FakeClient([respond(bad)] * (SETTINGS.max_retries + 1))
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert (outcome.status, outcome.escalation) == ("ok", None)
    assert [f.category for f in outcome.findings] == [Category.GROUNDING]


def test_a_document_discrepancy_survives_one_retry_with_the_documents_values_unchanged():
    document = DOCUMENT + "Base $ 1,000.00\nFee $ 100.00\n"
    components = [
        PremiumComponent(name="Base", amount=Decimal("1000.00"), quote="Base $ 1,000.00"),
        PremiumComponent(name="Fee", amount=Decimal("100.00"), quote="Fee $ 100.00"),
    ]
    stated = make_extraction(premium_components=components)
    client = FakeClient([respond(stated), respond(stated)])

    outcome = run_extraction(client, document, SETTINGS)

    assert len(client.requests) == 2  # one retry, not max_retries
    assert outcome.status == "ok"
    [finding] = outcome.findings
    assert (finding.code, finding.document_discrepancy) == ("components_sum_mismatch", True)
    assert outcome.extraction is not None
    assert outcome.extraction.record.premium_amount == Decimal("1200.00")
    assert outcome.history[0][0].document_discrepancy is False


def test_a_retry_that_changes_a_stated_number_is_caught_by_grounding():
    document = DOCUMENT + "Base $ 1,000.00\nFee $ 100.00\n"
    components = [
        PremiumComponent(name="Base", amount=Decimal("1000.00"), quote="Base $ 1,000.00"),
        PremiumComponent(name="Fee", amount=Decimal("100.00"), quote="Fee $ 100.00"),
    ]
    stated = make_extraction(premium_components=components)
    fudged = make_extraction(premium_components=components, premium_amount=Decimal("1100.00"))
    client = FakeClient([respond(stated)] + [respond(fudged)] * SETTINGS.max_retries)

    outcome = run_extraction(client, document, SETTINGS)

    assert outcome.status == "ok"
    assert ("grounding", "number_not_in_quote") in {
        (f.category.value, f.code) for f in outcome.findings
    }


def test_a_refusal_is_a_model_refusal_escalation_and_is_never_parsed():
    client = FakeClient([respond(make_extraction(), stop_reason="refusal")])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert (outcome.status, outcome.escalation, outcome.extraction) == (
        "escalated",
        Escalation.MODEL_REFUSAL,
        None,
    )


def test_truncated_output_is_a_truncated_escalation():
    client = FakeClient([respond(None, stop_reason="max_tokens")])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert outcome.escalation is Escalation.TRUNCATED
    assert outcome.failure == "max_tokens"


def test_every_call_is_accounted_for():
    client = FakeClient([respond(make_extraction(effective_date="x")), respond(make_extraction())])
    outcome = run_extraction(client, DOCUMENT, SETTINGS)
    assert [c.role for c in outcome.calls] == ["extractor", "extractor"]
