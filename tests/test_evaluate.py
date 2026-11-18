"""Evaluation rules (§9): field comparison, outcome scoring, calibration and reviewer value."""

from decimal import Decimal
from typing import Any

from policy_extractor.derive import derive
from policy_extractor.evaluate import (
    GoldLabel,
    compare_field,
    compare_list,
    evaluate,
    render_report,
    same_name,
)
from policy_extractor.models import (
    CoverageLimitValue,
    LimitBasis,
    PolicyValues,
)
from policy_extractor.pipeline import DocumentResult
from policy_extractor.route import Decision, Escalation, Outcome
from tests.fakes import make_extraction, make_review, settings

SETTINGS = settings()
TOL = SETTINGS.money_tolerance


def gold_values(**overrides: Any) -> PolicyValues:
    return make_extraction().values().model_copy(update=overrides)


def result(
    doc_id: str,
    outcome: Outcome,
    *,
    escalation: Escalation | None = None,
    confidence: float | dict[str, float] = 0.99,
    disagree: dict[str, str] | None = None,
    **record: Any,
) -> DocumentResult:
    extraction = make_extraction(confidence=confidence, **record)
    return DocumentResult(
        doc_id=doc_id,
        policy_type="auto",
        decision=Decision(outcome=outcome, escalation=escalation, reasons=[]),
        extraction=extraction,
        derived=derive(extraction.values()),
        findings=[],
        finding_history=[],
        review=make_review(disagree),
        calls=[],
    )


def label(
    doc_id: str, outcome: str, escalation: Escalation | None = None, **values: Any
) -> GoldLabel:
    return GoldLabel(
        doc_id=doc_id,
        expected_outcome=outcome,  # type: ignore[arg-type]
        expected_escalation=escalation,
        why="test",
        values=gold_values(**values),
    )


def test_names_match_leniently_but_not_loosely():
    assert same_name("ANYWHERE INSURANCE CO.", "Anywhere Insurance Co.")
    assert same_name("Comprehensive", "Comprehensive deductible")
    assert same_name("Coverage A - Dwelling", "Coverage A (Dwelling)")
    assert same_name("Summit Excess Insurance Company", "Summit Excess Ins Co., Hartford CT")
    assert not same_name("Collision", "Comprehensive")
    assert not same_name("Property Damage", "Bodily Injury")


def test_lists_match_one_to_one_in_any_order_with_money_tolerance_and_free_notes():
    expected = [
        CoverageLimitValue(
            coverage="Bodily Injury", amount=Decimal(100000), basis=LimitBasis.PER_PERSON, note=None
        ),
        CoverageLimitValue(
            coverage="Bodily Injury",
            amount=Decimal(300000),
            basis=LimitBasis.PER_ACCIDENT,
            note=None,
        ),
    ]
    actual = [
        CoverageLimitValue(
            coverage="Bodily Injury Liability",
            amount=Decimal("300000.00"),
            basis=LimitBasis.PER_ACCIDENT,
            note="split",
        ),
        CoverageLimitValue(
            coverage="Bodily Injury Liability",
            amount=Decimal("100000"),
            basis=LimitBasis.PER_PERSON,
            note=None,
        ),
    ]
    assert compare_list("coverage_limits", expected, actual, TOL)
    wrong_basis = [actual[0].model_copy(update={"basis": LimitBasis.AGGREGATE}), actual[1]]
    assert not compare_list("coverage_limits", expected, wrong_basis, TOL)
    assert not compare_list("coverage_limits", expected, actual[:1], TOL)


def test_no_list_and_an_empty_list_are_the_same():
    assert compare_list("endorsements", None, [], TOL)
    assert compare_list("exclusions", [], None, TOL)


def test_scalar_rules():
    gold = gold_values()
    assert compare_field("policy_number", gold, gold_values(policy_number="tp-100"), TOL)
    assert compare_field(
        "premium_amount", gold, gold_values(premium_amount=Decimal("1200.004")), TOL
    )
    assert not compare_field(
        "premium_amount", gold, gold_values(premium_amount=Decimal("1200.10")), TOL
    )
    assert compare_field("carrier", gold, gold_values(carrier="Testline Mutual Insurance"), TOL)
    assert not compare_field("effective_date", gold, gold_values(effective_date="2026-01-02"), TOL)
    assert not compare_field("policy_form", gold, gold_values(policy_form="HO-3"), TOL)


def test_a_wrong_field_that_was_auto_approved_is_reported():
    results = [
        result("A", Outcome.AUTO_APPROVE),
        result("B", Outcome.SPOT_CHECK, premium_amount=Decimal("1250.00")),
        result("C", Outcome.HUMAN_REVIEW, disagree={"carrier": "x"}, carrier="Wrong Co"),
    ]
    gold = {doc: label(doc, "auto_approve") for doc in ("A", "B")} | {
        "C": label("C", "human_review")
    }
    evaluation = evaluate(results, gold, SETTINGS)
    assert evaluation.auto_approved == 2  # a spot check is still an automatic approval
    assert evaluation.auto_approved_with_errors == ["B"]
    assert evaluation.outcome_correct == 3
    assert evaluation.field_accuracy["premium_amount"] == 2 / 3


def test_reviewer_value_counts_catches_misses_and_false_disputes():
    results = [
        result("A", Outcome.HUMAN_REVIEW, carrier="Wrong Co", disagree={"carrier": "x"}),
        result("B", Outcome.AUTO_APPROVE, policyholder="Wrong Person"),
        result("C", Outcome.HUMAN_REVIEW, disagree={"premium_amount": "x"}),
    ]
    gold = {doc: label(doc, "human_review") for doc in ("A", "B", "C")}
    reviewer = evaluate(results, gold, SETTINGS).reviewer
    assert (reviewer["caught"], reviewer["missed"], reviewer["false_dispute"]) == (1, 1, 1)


def test_calibration_is_sliced_and_flags_confident_wrong_cells():
    results = [
        result(
            doc,
            Outcome.AUTO_APPROVE,
            confidence={"premium_amount": 0.95},
            premium_amount=Decimal("9.00"),
        )
        for doc in ("A", "B")
    ]
    gold = {doc: label(doc, "auto_approve") for doc in ("A", "B")}
    evaluation = evaluate(results, gold, SETTINGS)
    [cell] = [c for c in evaluation.calibration if c.field == "premium_amount"]
    assert (cell.policy_type, cell.n, cell.accuracy, cell.flagged) == ("auto", 2, 0.0, True)
    assert round(cell.brier, 4) == round(0.95**2, 4)
    assert evaluation.overall_accuracy > 0.9  # the aggregate hides the bad cell


def test_escalations_are_scored_by_category():
    escalated = result("E", Outcome.ESCALATED, escalation=Escalation.MISSING_SOURCE)
    gold = {"E": label("E", "escalated", Escalation.MISSING_SOURCE)}
    evaluation = evaluate([escalated], gold, SETTINGS)
    assert (evaluation.escalation_correct, evaluation.escalations_expected) == (1, 1)
    report = render_report(evaluation, SETTINGS)
    assert "Escalation category as expected: 1/1" in report
    assert "| E | escalated (missing_source) | escalated (missing_source) | - |" in report
