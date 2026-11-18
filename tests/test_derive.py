"""Every §5 derivation rule, including the ambiguous cases that must return None."""

from decimal import Decimal
from typing import Any

from policy_extractor.derive import derive, primary_deductible, primary_limit, term_months
from policy_extractor.models import (
    CoverageLimitValue,
    DeductibleValue,
    LimitBasis,
    PolicyType,
    PolicyValues,
)


def lim(coverage: str, amount: int, basis: LimitBasis = LimitBasis.OTHER) -> CoverageLimitValue:
    return CoverageLimitValue(coverage=coverage, amount=Decimal(amount), basis=basis, note=None)


def ded(
    applies_to: str, amount: int | None, *, primary: bool = False, **extra: Any
) -> DeductibleValue:
    values: dict[str, Any] = {
        "applies_to": applies_to,
        "amount": None if amount is None else Decimal(amount),
        "percent": None,
        "percent_of": None,
        "labelled_primary": primary,
    }
    return DeductibleValue(**(values | extra))


def values(policy_type: PolicyType | None, **fields: Any) -> PolicyValues:
    empty = {name: None for name in PolicyValues.model_fields}
    return PolicyValues(**(empty | {"policy_type": policy_type} | fields))


AUTO_LIMITS = [
    lim("Bodily Injury Liability", 100000, LimitBasis.PER_PERSON),
    lim("Bodily Injury Liability", 300000, LimitBasis.PER_ACCIDENT),
    lim("Uninsured Motorist BI", 500000, LimitBasis.PER_ACCIDENT),
    lim("Property Damage", 50000, LimitBasis.PER_ACCIDENT),
]


def test_auto_primary_limit_is_bodily_injury_per_accident_not_uninsured_motorist():
    assert primary_limit(values(PolicyType.AUTO, coverage_limits=AUTO_LIMITS)) == (
        Decimal(300000),
        "auto: bodily injury per accident",
    )


def test_home_primary_limit_is_coverage_a_however_it_is_written():
    for name in ("Coverage A - Dwelling", "A - Dwelling", "Coverage A (Dwelling)", "A Dwelling"):
        limits = [lim(name, 385000), lim("Coverage B - Other Structures", 38500)]
        assert primary_limit(values(PolicyType.HOME, coverage_limits=limits))[0] == 385000


def test_umbrella_primary_limit_is_per_occurrence():
    limits = [
        lim("Each occurrence", 2000000, LimitBasis.PER_OCCURRENCE),
        lim("Aggregate", 4000000, LimitBasis.AGGREGATE),
    ]
    assert primary_limit(values(PolicyType.UMBRELLA, coverage_limits=limits))[0] == 2000000


def test_other_uses_scheduled_total_then_a_single_aggregate_else_none():
    scheduled = [lim("Total scheduled value", 1427500, LimitBasis.SCHEDULED_TOTAL)]
    assert primary_limit(values(PolicyType.OTHER, coverage_limits=scheduled))[0] == 1427500
    one_aggregate = [lim("Aggregate", 900000, LimitBasis.AGGREGATE), lim("Hull", 96000)]
    assert primary_limit(values(PolicyType.OTHER, coverage_limits=one_aggregate))[0] == 900000
    assert primary_limit(values(PolicyType.OTHER, coverage_limits=[lim("Hull", 96000)]))[0] is None


def test_primary_limit_is_none_when_the_rule_is_ambiguous():
    limits = [
        lim("Coverage A - Dwelling", 385000),
        lim("Coverage A - Dwelling (extended)", 400000),
    ]
    limit, rule = primary_limit(values(PolicyType.HOME, coverage_limits=limits))
    assert (limit, rule) == (None, "home: Coverage A (dwelling): ambiguous")


def test_primary_deductible_prefers_the_labelled_one():
    deductibles = [
        ded("Comprehensive", 500),
        ded("Collision", 1000),
        ded("Primary", 500, primary=True),
    ]
    assert primary_deductible(values(PolicyType.AUTO, deductibles=deductibles)) == (
        Decimal(500),
        "labelled primary",
    )


def test_primary_deductible_falls_back_to_the_only_one_or_the_common_amount():
    assert primary_deductible(values(PolicyType.HOME, deductibles=[ded("AOP", 1000)]))[0] == 1000
    equal = [ded("Comprehensive", 500), ded("Collision", 500)]
    assert primary_deductible(values(PolicyType.AUTO, deductibles=equal)) == (
        Decimal(500),
        "all deductibles equal",
    )


def test_different_unlabelled_deductibles_are_ambiguous_not_guessed():
    deductibles = [ded("Collision", 750), ded("Comprehensive", 250)]
    assert primary_deductible(values(PolicyType.AUTO, deductibles=deductibles)) == (
        None,
        "ambiguous: different deductibles, none labelled",
    )


def test_percentage_deductibles_do_not_compete_for_primary():
    deductibles = [
        ded("All other perils", 2500),
        ded("Named storm", None, percent=Decimal(5), percent_of="Coverage A"),
    ]
    assert primary_deductible(values(PolicyType.HOME, deductibles=deductibles))[0] == 2500


def test_umbrella_primary_deductible_is_the_self_insured_retention():
    deductibles = [ded("Self-insured retention", 250)]
    assert primary_deductible(values(PolicyType.UMBRELLA, deductibles=deductibles)) == (
        Decimal(250),
        "umbrella: self-insured retention",
    )


def test_percentage_deductible_amount_is_computed_from_the_named_coverage():
    derived = derive(
        values(
            PolicyType.HOME,
            coverage_limits=[lim("A - Dwelling", 385000)],
            deductibles=[ded("Wind/Hail", None, percent=Decimal(2), percent_of="Coverage A")],
        )
    )
    [computed] = derived.percentage_deductibles
    assert (computed.base_coverage, computed.amount) == ("A - Dwelling", Decimal("7700.00"))


def test_term_months():
    assert term_months("2025-03-15", "2026-03-15") == 12
    assert term_months("2025-03-15", "2025-09-14") == 5
    assert term_months("2025-07-01", "2025-07-01") == 0
    assert term_months(None, "2026-01-01") is None
    assert term_months("01/01/2025", "2026-01-01") is None
