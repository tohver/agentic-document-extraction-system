"""Names are matched by whole words, and scoring cannot be flattered by loose matching, missing
gold labels or fields that are empty by construction."""

from decimal import Decimal

import pytest

from policy_extractor.derive import percentage_deductibles, primary_limit
from policy_extractor.evaluate import compare_field, evaluate, render_report
from policy_extractor.models import PolicyType
from policy_extractor.route import Outcome
from tests.fakes import settings
from tests.test_derive import ded, lim, values
from tests.test_evaluate import gold_values, label, result

SETTINGS = settings()
TOL = SETTINGS.money_tolerance
HOME_LIMITS = [
    lim("Coverage Additional Living Expense", 40000),
    lim("Coverage A - Dwelling", 385000),
]


def test_coverage_a_is_matched_by_word_not_by_letter_prefix():
    home = values(PolicyType.HOME, coverage_limits=HOME_LIMITS)
    assert primary_limit(home) == (Decimal(385000), "home: Coverage A (dwelling)")
    windstorm = ded("Wind/Hail", None, percent=Decimal(2), percent_of="Coverage A")
    [computed] = percentage_deductibles(
        values(PolicyType.HOME, coverage_limits=HOME_LIMITS, deductibles=[windstorm])
    )
    assert (computed.base_coverage, computed.amount) == (
        "Coverage A - Dwelling",
        Decimal("7700.00"),
    )


def test_parties_are_compared_word_for_word():
    gold = gold_values(
        policyholder="Gregory & Patricia Lindholm",
        carrier="Summit Excess & Umbrella Insurance Company",
    )
    assert compare_field(
        "policyholder", gold, gold_values(policyholder="GREGORY & PATRICIA LINDHOLM"), TOL
    )
    assert not compare_field(
        "policyholder", gold, gold_values(policyholder="Gregory Lindholm"), TOL
    )
    assert compare_field(
        "carrier", gold, gold_values(carrier="Summit Excess & Umbrella Ins Co."), TOL
    )
    assert not compare_field("carrier", gold, gold_values(carrier="Insurance Company"), TOL)
    assert not compare_field(
        "carrier", gold, gold_values(carrier="Summit Excess & Umbrella Insurance Brokers"), TOL
    )


def test_the_policy_form_is_an_identifier():
    gold = gold_values(policy_form="HO-3")
    assert compare_field("policy_form", gold, gold_values(policy_form="HO 3"), TOL)
    assert not compare_field("policy_form", gold, gold_values(policy_form="HO-3 Special Form"), TOL)


def test_a_result_without_a_gold_label_is_an_error_not_a_skip():
    results = [result("A", Outcome.AUTO_APPROVE), result("B", Outcome.AUTO_APPROVE)]
    with pytest.raises(ValueError, match="no gold label for: B"):
        evaluate(results, {"A": label("A", "auto_approve")}, SETTINGS)


def test_accuracy_is_also_reported_on_fields_the_document_states():
    wrong_premium = result("A", Outcome.HUMAN_REVIEW, premium_amount=Decimal("1250.00"))
    evaluation = evaluate([wrong_premium], {"A": label("A", "human_review")}, SETTINGS)

    assert evaluation.field_accuracy_stated["premium_amount"] == 0.0
    assert evaluation.field_accuracy_stated["underlying_requirements"] is None  # empty in gold
    assert evaluation.field_count_stated["underlying_requirements"] == 0
    # 1 wrong of 17 fields overall, but 1 wrong of the 13 the document states.
    assert evaluation.overall_accuracy == pytest.approx(16 / 17)
    assert evaluation.overall_accuracy_stated == pytest.approx(12 / 13)
    report = render_report(evaluation, SETTINGS)
    assert "on fields the document states: 92.3%" in report
    assert "| premium_amount | 0.0% | 0.0% | 1 |" in report
    assert "| underlying_requirements | 100.0% | - | 0 |" in report
