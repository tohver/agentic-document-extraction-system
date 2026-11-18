"""Every §6 check, both passing (the clean fake) and failing (one targeted change each)."""

from decimal import Decimal

import pytest

from policy_extractor.models import DocumentKind, FieldStatus, LimitBasis, PolicyType
from policy_extractor.validate import Category, numbers_in, validate
from tests.fakes import (
    DOCUMENT,
    deductible,
    default_limits,
    limit,
    make_extraction,
    settings,
)

SETTINGS = settings()


def codes(findings) -> set[tuple[str, str, str]]:
    return {(f.category.value, f.code, f.field) for f in findings}


def test_a_clean_grounded_extraction_has_no_findings():
    assert validate(make_extraction(), DOCUMENT, SETTINGS) == []


def test_not_a_policy_is_the_only_finding():
    extraction = make_extraction(document_kind=DocumentKind.NOT_A_POLICY, premium_amount=None)
    [finding] = validate(extraction, DOCUMENT, SETTINGS)
    assert (finding.category, finding.code) == (Category.NOT_A_POLICY, "not_a_policy")


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"premium_amount": None}, "premium_amount"),
        ({"policy_number": ""}, "policy_number"),
        ({"carrier": ""}, "carrier"),
        ({"policyholder": ""}, "policyholder"),
        ({"policy_type": None}, "policy_type"),
        ({"effective_date": ""}, "effective_date"),
        ({"expiration_date": ""}, "expiration_date"),
        ({"coverage_limits": []}, "coverage_limits"),
    ],
)
def test_an_absent_required_field_is_missing_source(override, field):
    findings = validate(make_extraction(**override), DOCUMENT, SETTINGS)
    assert ("missing_source", f"{field}_absent", field) in codes(findings)


def test_a_field_referenced_to_a_missing_attachment_is_missing_source():
    extraction = make_extraction(
        endorsements=[],
        statuses={"endorsements": FieldStatus.REFERENCED_NOT_ATTACHED},
    )
    findings = validate(extraction, DOCUMENT, SETTINGS)
    assert ("missing_source", "referenced_not_attached", "endorsements") in codes(findings)


def test_an_optional_field_that_is_not_stated_is_fine():
    assert validate(make_extraction(endorsements=[]), DOCUMENT, SETTINGS) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"premium_amount": Decimal("-1200.00")}, ("format", "negative", "premium_amount")),
        (
            {"deductibles": [deductible(amount=Decimal("-500"))]},
            ("format", "negative", "deductibles"),
        ),
        (
            {"deductibles": [deductible(amount=None, percent=Decimal("150"), percent_of="A")]},
            ("format", "percent_range", "deductibles"),
        ),
        (
            {"deductibles": [deductible(amount=None, percent=Decimal("2"), percent_of="")]},
            ("format", "percent_of_missing", "deductibles"),
        ),
        ({"policy_type": PolicyType.OTHER}, ("format", "detail_missing", "policy_type_detail")),
        ({"effective_date": "01/01/2026"}, ("format", "date_not_iso", "effective_date")),
        ({"expiration_date": "2027-02-30"}, ("format", "date_not_iso", "expiration_date")),
        ({"policy_number": "TP - 100"}, ("format", "whitespace", "policy_number")),
    ],
)
def test_format_findings(override, expected):
    assert expected in codes(validate(make_extraction(**override), DOCUMENT, SETTINGS))


def test_status_must_match_the_value():
    stated_but_empty = make_extraction(policy_form="", statuses={"policy_form": FieldStatus.STATED})
    value_but_not_stated = make_extraction(statuses={"carrier": FieldStatus.NOT_STATED})
    assert ("format", "status_value_mismatch", "policy_form") in codes(
        validate(stated_but_empty, DOCUMENT, SETTINGS)
    )
    assert ("format", "status_value_mismatch", "carrier") in codes(
        validate(value_but_not_stated, DOCUMENT, SETTINGS)
    )


def test_a_stated_value_needs_a_quote():
    extraction = make_extraction(quotes={"carrier": ""})
    assert ("format", "quote_missing", "carrier") in codes(validate(extraction, DOCUMENT, SETTINGS))


def test_every_field_needs_exactly_one_evidence_entry():
    extraction = make_extraction()
    extraction.evidence.pop()
    assert ("format", "evidence_count", "exclusions_complete") in codes(
        validate(extraction, DOCUMENT, SETTINGS)
    )


def test_premium_components_that_do_not_add_up_are_a_consistency_finding():
    from policy_extractor.models import PremiumComponent

    components = [
        PremiumComponent(name="Base", amount=Decimal("1000.00"), quote="$ 1,200.00"),
        PremiumComponent(name="Fee", amount=Decimal("100.00"), quote="$ 1,200.00"),
    ]
    findings = validate(make_extraction(premium_components=components), DOCUMENT, SETTINGS)
    [finding] = [f for f in findings if f.code == "components_sum_mismatch"]
    assert finding.category is Category.CONSISTENCY
    assert finding.observed == "premium=1200.00 components=1100.00"


def test_expiration_must_be_after_effective():
    extraction = make_extraction(
        expiration_date="2026-01-01",
        quotes={"expiration_date": "Policy Period: 01/01/2026 to 01/01/2027"},
    )
    assert ("consistency", "term_not_positive", "expiration_date") in codes(
        validate(extraction, DOCUMENT, SETTINGS)
    )


def test_per_person_cannot_exceed_per_accident():
    bodily = "Bodily Injury: $100,000 / $300,000"
    limits = [
        limit("Bodily Injury", "300000", LimitBasis.PER_PERSON, bodily),
        limit("Bodily Injury", "100000", LimitBasis.PER_ACCIDENT, bodily),
    ]
    assert ("consistency", "per_person_exceeds_per_accident", "coverage_limits") in codes(
        validate(make_extraction(coverage_limits=limits), DOCUMENT, SETTINGS)
    )


def test_primary_deductible_cannot_exceed_primary_limit():
    document = DOCUMENT + "Deductible (Primary): $500,000\n"
    extraction = make_extraction(
        deductibles=[deductible(amount=Decimal("500000"), quote="Deductible (Primary): $500,000")]
    )
    assert ("consistency", "deductible_exceeds_limit", "deductibles") in codes(
        validate(extraction, document, SETTINGS)
    )


def test_a_stated_percentage_deductible_amount_must_match_its_computation():
    document = DOCUMENT + "Coverage A - Dwelling $ 385,000\nWind/Hail 2% of Cov A (= $7,800)\n"
    limits = [
        *default_limits(),
        limit(
            "Coverage A - Dwelling", "385000", LimitBasis.OTHER, "Coverage A - Dwelling $ 385,000"
        ),
    ]
    wind = deductible(
        applies_to="Wind/Hail",
        amount=Decimal("7800"),
        percent=Decimal("2"),
        percent_of="Coverage A",
        labelled_primary=False,
        quote="Wind/Hail 2% of Cov A (= $7,800)",
    )
    extraction = make_extraction(coverage_limits=limits, deductibles=[deductible(), wind])
    [finding] = [
        f
        for f in validate(extraction, document, SETTINGS)
        if f.code == "percentage_amount_mismatch"
    ]
    assert finding.observed == "Wind/Hail: stated=7800 computed=7700.00"


def test_a_quote_that_is_not_in_the_document_is_a_grounding_finding():
    extraction = make_extraction(quotes={"carrier": "TESTLINE CASUALTY"})
    assert ("grounding", "quote_not_in_document", "carrier") in codes(
        validate(extraction, DOCUMENT, SETTINGS)
    )


def test_quotes_match_despite_whitespace_case_and_dot_leaders():
    document = DOCUMENT.replace(
        "Annual Premium: $ 1,200.00", "ANNUAL   PREMIUM ......... $ 1,200.00"
    )
    extraction = make_extraction(quotes={"premium_amount": "Annual Premium $ 1,200.00"})
    assert validate(extraction, document, SETTINGS) == []


def test_a_changed_number_is_caught_against_its_own_quote():
    """The guard that stops a retry from 'fixing' a number the document states."""
    findings = validate(make_extraction(premium_amount=Decimal("1250.00")), DOCUMENT, SETTINGS)
    assert ("grounding", "number_not_in_quote", "premium_amount") in codes(findings)


def test_a_list_item_quote_and_number_are_checked():
    limits = [
        limit("Property Damage", "75000", LimitBasis.PER_ACCIDENT, "Property Damage: $50,000")
    ]
    findings = validate(make_extraction(coverage_limits=limits), DOCUMENT, SETTINGS)
    assert ("grounding", "number_not_in_quote", "coverage_limits") in codes(findings)


def test_dates_must_be_readable_from_their_quote_in_numbers_or_words():
    wrong = make_extraction(effective_date="2026-02-01")
    assert ("grounding", "date_not_in_quote", "effective_date") in codes(
        validate(wrong, DOCUMENT, SETTINGS)
    )
    document = DOCUMENT + "Renewed for the period January 1, 2026 through January 1, 2027.\n"
    worded = make_extraction(quotes={"effective_date": "January 1, 2026"})
    assert validate(worded, document, SETTINGS) == []


def test_exclusion_text_must_be_in_the_document():
    extraction = make_extraction(exclusions=["Racing", "War"])
    assert ("grounding", "text_not_in_document", "exclusions") in codes(
        validate(extraction, DOCUMENT, SETTINGS)
    )


def test_numbers_are_read_with_thousands_and_million_suffixes():
    assert Decimal("1000000") in numbers_in("UMB-09 Motorist Excess - $1M")
    assert Decimal("250000") in numbers_in("sublimit $250k")
    assert numbers_in("$ 1,847.62") == {Decimal("1847.62")}


def test_in_thousands_limits_are_grounded():
    document = DOCUMENT + "Uninsured Motorist ..... 50/100 (in thousands)\n"
    quote = "Uninsured Motorist ..... 50/100 (in thousands)"
    limits = [
        *default_limits(),
        limit("Uninsured Motorist", "50000", LimitBasis.PER_PERSON, quote),
        limit("Uninsured Motorist", "100000", LimitBasis.PER_ACCIDENT, quote),
    ]
    assert validate(make_extraction(coverage_limits=limits), document, SETTINGS) == []
