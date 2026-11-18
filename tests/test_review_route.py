"""Independent review and deterministic routing (§7)."""

from policy_extractor.fields import FIELD_DEFINITIONS
from policy_extractor.models import ALL_FIELDS, PolicyType
from policy_extractor.review import review_request
from policy_extractor.route import Escalation, Outcome, escalate, route
from policy_extractor.validate import Category, Finding
from tests.fakes import DOCUMENT, make_extraction, make_review, settings

SETTINGS = settings()


def test_reviewer_sees_the_field_definitions_the_document_and_the_values_only():
    extraction = make_extraction(confidence=0.37, quotes={"carrier": "QUOTE-MARKER"})
    request = review_request(DOCUMENT, extraction, SETTINGS)

    for definition in FIELD_DEFINITIONS.values():
        assert definition in request.system
    [message] = request.messages
    assert DOCUMENT in message.content
    assert '"carrier": "TESTLINE MUTUAL INSURANCE"' in message.content
    # No extractor confidence, quotes, statuses or reasoning reach the reviewer.
    for leaked in ("0.37", "confidence", "QUOTE-MARKER", "evidence", "status"):
        assert leaked not in message.content


def test_extractor_and_reviewer_use_the_configured_different_models():
    from policy_extractor.extract import extraction_request

    extractor = extraction_request(DOCUMENT, SETTINGS)
    reviewer = review_request(DOCUMENT, make_extraction(), SETTINGS)
    assert (extractor.model, extractor.effort) == ("claude-sonnet-5-5", "medium")
    assert (reviewer.model, reviewer.effort) == ("claude-opus-5-5", "high")
    assert extractor.model != reviewer.model


def test_all_clear_is_auto_approved():
    decision = route(make_extraction(), [], make_review(), SETTINGS)
    assert (decision.outcome, decision.reasons) == (Outcome.AUTO_APPROVE, [])


def test_high_confidence_does_not_override_a_reviewer_disagreement():
    review = make_review({"coverage_limits": 'Document says "$100,000 / $300,000".'})
    decision = route(make_extraction(confidence=1.0), [], review, SETTINGS)
    assert decision.outcome is Outcome.HUMAN_REVIEW
    assert decision.reasons == ["reviewer_disagreement:coverage_limits"]
    assert decision.reviewer_reasons == {"coverage_limits": 'Document says "$100,000 / $300,000".'}


def test_routing_reasons_list_every_signal_that_fired():
    extraction = make_extraction(
        confidence={"premium_amount": 0.6},
        policy_type=PolicyType.OTHER,
        policy_type_detail="Fine Art Floater",
        exclusions_complete=False,
    )
    findings = [
        Finding(
            field="premium_amount",
            category=Category.CONSISTENCY,
            code="components_sum_mismatch",
            message="m",
            document_discrepancy=True,
        ),
        Finding(
            field="carrier", category=Category.GROUNDING, code="quote_not_in_document", message="m"
        ),
    ]
    review = make_review({"deductibles": "r"}, omit=["exclusions"])
    decision = route(extraction, findings, review, SETTINGS)
    assert decision.outcome is Outcome.HUMAN_REVIEW
    assert decision.reasons == [
        "low_confidence:premium_amount",
        "reviewer_disagreement:deductibles",
        "reviewer_no_verdict:exclusions",
        "document_discrepancy_consistency:components_sum_mismatch:premium_amount",
        "unresolved_grounding:quote_not_in_document:carrier",
        "policy_type_other",
        "exclusions_partial",
    ]


def test_a_missing_review_never_approves():
    decision = route(make_extraction(), [], None, SETTINGS)
    assert decision.reasons == ["reviewer_unavailable"]


def test_confidence_is_checked_for_every_field():
    for field in ALL_FIELDS:
        decision = route(make_extraction(confidence={field: 0.5}), [], make_review(), SETTINGS)
        assert decision.reasons == [f"low_confidence:{field}"]


def test_escalation_keeps_its_category_and_reasons():
    finding = Finding(
        field="premium_amount",
        category=Category.MISSING_SOURCE,
        code="premium_amount_absent",
        message="m",
    )
    decision = escalate(Escalation.MISSING_SOURCE, [finding])
    assert (decision.outcome, decision.escalation) == (Outcome.ESCALATED, Escalation.MISSING_SOURCE)
    assert decision.reasons == ["missing_source:premium_amount_absent:premium_amount"]
