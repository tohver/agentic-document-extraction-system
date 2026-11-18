"""Grounding: a value must be readable from its own quote, and a retry may not drift."""

from datetime import date
from decimal import Decimal

from policy_extractor.validate import Category, changed_on_retry, dates_in, validate
from tests.fakes import DOCUMENT, default_limits, make_extraction, settings

SETTINGS = settings()


def codes(findings) -> set[tuple[str, str, str]]:
    return {(f.category.value, f.code, f.field) for f in findings}


def test_dates_are_read_as_whole_dates_in_every_notation():
    assert dates_in("March 1, 2025 through March 1, 2026") == {date(2025, 3, 1), date(2026, 3, 1)}
    assert dates_in("Mar. 1st 2025") == {date(2025, 3, 1)}
    assert dates_in("Sept 30, 2025") == {date(2025, 9, 30)}
    assert dates_in("1 March 2025") == {date(2025, 3, 1)}
    assert dates_in("Policy Period: 01/20/2025 to 01-20-2026") == {
        date(2025, 1, 20),
        date(2026, 1, 20),
    }
    assert dates_in("Generated 2025-03-15") == {date(2025, 3, 15)}


def test_scattered_numbers_and_ocr_noise_are_not_a_date():
    assert dates_in("Summary of coverage 2025, 1 page") == set()
    assert dates_in("This policy may be renewed 3 times in 2025") == set()
    assert dates_in("INV DATE  Mar ll, 2O26") == set()


def test_a_swapped_day_and_month_is_not_grounded():
    period = "Policy Period: 12/01/2026 to 12/01/2027"
    document = DOCUMENT.replace("Policy Period: 01/01/2026 to 01/01/2027", period)
    quotes = {"effective_date": period, "expiration_date": period}
    right = make_extraction(
        effective_date="2026-12-01", expiration_date="2027-12-01", quotes=quotes
    )
    swapped = make_extraction(
        effective_date="2026-01-12", expiration_date="2027-12-01", quotes=quotes
    )
    assert validate(right, document, SETTINGS) == []
    assert codes(validate(swapped, document, SETTINGS)) == {
        ("grounding", "date_not_in_quote", "effective_date")
    }


def test_a_carrier_must_be_read_from_its_own_quote():
    document = DOCUMENT + "Agent: Pinewood Insurance Services\n"
    agent = make_extraction(carrier="Pinewood Insurance Services")  # quote: the header
    assert ("grounding", "value_not_in_quote", "carrier") in codes(
        validate(agent, document, SETTINGS)
    )


def test_a_policyholder_missing_a_joint_insured_is_not_grounded():
    document = DOCUMENT.replace("Jane Q. Tester", "Jane & John Tester")
    quotes = {"policyholder": "Named Insured: Jane & John Tester"}
    whole = make_extraction(policyholder="Jane & John Tester", quotes=quotes)
    partial = make_extraction(policyholder="Jane Tester", quotes=quotes)
    assert validate(whole, document, SETTINGS) == []
    assert codes(validate(partial, document, SETTINGS)) == {
        ("grounding", "value_not_in_quote", "policyholder")
    }


def test_identity_values_tolerate_case_abbreviations_and_a_spaced_policy_number():
    document = DOCUMENT.replace("TESTLINE MUTUAL INSURANCE", "TESTLINE MUTUAL INS CO.").replace(
        "Policy Number: TP-100", "Policy Number: TP - 100"
    )
    extraction = make_extraction(
        carrier="Testline Mutual Insurance Company",
        quotes={"carrier": "TESTLINE MUTUAL INS CO.", "policy_number": "Policy Number: TP - 100"},
    )
    assert validate(extraction, document, SETTINGS) == []


def test_a_change_no_finding_asked_for_is_drift_but_reordering_is_not():
    first = make_extraction()
    reordered = make_extraction(coverage_limits=list(reversed(default_limits())))
    assert changed_on_retry(first, reordered, set()) == []

    fudged = make_extraction(premium_amount=Decimal("1100.00"))
    [finding] = changed_on_retry(first, fudged, set())
    assert (finding.category, finding.code, finding.field) == (
        Category.GROUNDING,
        "changed_on_retry",
        "premium_amount",
    )
    assert finding.observed == "1200.00 -> 1100.00"
    assert changed_on_retry(first, fudged, {"premium_amount"}) == []
