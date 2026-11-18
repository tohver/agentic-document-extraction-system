"""Derived fields (CLAUDE.md §5): computed in Python from stated values, never asked of the model.

"Which limit is the primary one?" and "which deductible?" were the questions the prototype asked
the model, and both the model and the expected values answered them inconsistently. Here they are
explicit rules over the full lists, and an ambiguous case returns None instead of a guess.
"""

import re
from datetime import date
from decimal import Decimal
from typing import Final

from pydantic import BaseModel

from policy_extractor.models import (
    CoverageLimitValue,
    DeductibleValue,
    LimitBasis,
    Money,
    PolicyType,
    PolicyValues,
)

CENT = Decimal("0.01")


class PercentageDeductible(BaseModel):
    applies_to: str
    percent: Money
    base_coverage: str
    amount: Money


class Derived(BaseModel):
    primary_limit: Money | None
    primary_limit_rule: str
    primary_deductible: Money | None
    primary_deductible_rule: str
    percentage_deductibles: list[PercentageDeductible]
    term_months: int | None


def derive(values: PolicyValues) -> Derived:
    limit, limit_rule = primary_limit(values)
    deductible, deductible_rule = primary_deductible(values)
    return Derived(
        primary_limit=limit,
        primary_limit_rule=limit_rule,
        primary_deductible=deductible,
        primary_deductible_rule=deductible_rule,
        percentage_deductibles=percentage_deductibles(values),
        term_months=term_months(values.effective_date, values.expiration_date),
    )


def normalise(text: str) -> str:
    """Lower-case words only, so 'Coverage A - Dwelling ....' matches 'coverage a dwelling'."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    expanded = ["coverage" if word == "cov" else word for word in words]
    return " ".join(expanded)


# Abbreviations in company names, so "Summit Ins Co." and "Summit Insurance Company" are one name.
ABBREVIATIONS: Final = {
    "ins": "insurance",
    "co": "company",
    "mut": "mutual",
    "corp": "corporation",
    "inc": "incorporated",
}


def name_key(text: str) -> str:
    """Comparable form of a name as written: case, punctuation and abbreviations do not count,
    but every word does ("Gregory Lindholm" is not "Gregory & Patricia Lindholm")."""
    return " ".join(ABBREVIATIONS.get(word, word) for word in normalise(text).split())


def starts_with_words(text: str, prefix: str) -> bool:
    """Word-wise prefix: 'coverage a dwelling' starts with 'coverage a', 'coverage additional'
    does not."""
    words, wanted = text.split(), prefix.split()
    return bool(wanted) and words[: len(wanted)] == wanted


def _is_bodily_injury(limit: CoverageLimitValue) -> bool:
    name = normalise(limit.coverage)
    words = set(name.split())
    uninsured = "uninsured" in name or "underinsured" in name or bool(words & {"um", "uim"})
    return ("bodily injury" in name or "bi" in words) and not uninsured


def _is_dwelling(limit: CoverageLimitValue) -> bool:
    name = normalise(limit.coverage)
    return "dwelling" in name.split() or starts_with_words(name, "coverage a") or name == "a"


def primary_limit(values: PolicyValues) -> tuple[Decimal | None, str]:
    limits = [limit for limit in values.coverage_limits or [] if limit.amount is not None]
    match values.policy_type:
        case PolicyType.AUTO:
            candidates = [
                limit
                for limit in limits
                if _is_bodily_injury(limit) and limit.basis is LimitBasis.PER_ACCIDENT
            ]
            rule = "auto: bodily injury per accident"
        case PolicyType.HOME:
            candidates = [limit for limit in limits if _is_dwelling(limit)]
            rule = "home: Coverage A (dwelling)"
        case PolicyType.UMBRELLA:
            candidates = [limit for limit in limits if limit.basis is LimitBasis.PER_OCCURRENCE]
            rule = "umbrella: per occurrence"
        case PolicyType.OTHER:
            candidates = [limit for limit in limits if limit.basis is LimitBasis.SCHEDULED_TOTAL]
            rule = "other: scheduled total"
            if not candidates:
                aggregates = [limit for limit in limits if limit.basis is LimitBasis.AGGREGATE]
                candidates = aggregates if len(aggregates) == 1 else []
                rule = "other: single aggregate"
        case None:
            return None, "no policy type"
    amounts = {limit.amount for limit in candidates}
    if len(amounts) != 1:
        return None, f"{rule}: {'none found' if not amounts else 'ambiguous'}"
    return amounts.pop(), rule


def _is_retention(deductible: DeductibleValue) -> bool:
    name = normalise(deductible.applies_to)
    return "retention" in name or "sir" in name.split() or "retained" in name


def primary_deductible(values: PolicyValues) -> tuple[Decimal | None, str]:
    deductibles = values.deductibles or []
    if values.policy_type is PolicyType.UMBRELLA:
        retained = {d.amount for d in deductibles if _is_retention(d) and d.amount is not None}
        if len(retained) == 1:
            return retained.pop(), "umbrella: self-insured retention"
        return None, "umbrella: self-insured retention not found"
    flat = [d for d in deductibles if d.amount is not None and d.percent is None]
    labelled = {d.amount for d in flat if d.labelled_primary}
    if len(labelled) == 1:
        return labelled.pop(), "labelled primary"
    if len(labelled) > 1:
        return None, "ambiguous: several labelled primary"
    if len(flat) == 1:
        return flat[0].amount, "only deductible"
    amounts = {d.amount for d in flat}
    if len(amounts) == 1:
        return amounts.pop(), "all deductibles equal"
    return None, "none stated" if not flat else "ambiguous: different deductibles, none labelled"


def percentage_deductibles(values: PolicyValues) -> list[PercentageDeductible]:
    """`percent` of the named coverage's limit, e.g. 2% of Coverage A 385,000 = 7,700."""
    results: list[PercentageDeductible] = []
    for deductible in values.deductibles or []:
        if deductible.percent is None or not deductible.percent_of:
            continue
        base = _find_limit(values.coverage_limits or [], deductible.percent_of)
        if base is None or base.amount is None:
            continue
        results.append(
            PercentageDeductible(
                applies_to=deductible.applies_to,
                percent=deductible.percent,
                base_coverage=base.coverage,
                amount=(deductible.percent / 100 * base.amount).quantize(CENT),
            )
        )
    return results


def _find_limit(limits: list[CoverageLimitValue], name: str) -> CoverageLimitValue | None:
    """Find the coverage a percentage refers to: 'Coverage A' also matches 'A - Dwelling'."""
    wanted = normalise(name)
    letter = re.fullmatch(r"coverage ([a-f])", wanted)
    for limit in limits:
        have = normalise(limit.coverage)
        if starts_with_words(have, wanted) or starts_with_words(wanted, have):
            return limit
        if letter and starts_with_words(have, letter[1]):
            return limit
    return None


def parse_iso(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def term_months(effective: str | None, expiration: str | None) -> int | None:
    start, end = parse_iso(effective), parse_iso(expiration)
    if start is None or end is None:
        return None
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return months - 1 if end.day < start.day else months
