"""Deterministic validation (CLAUDE.md §6). Pure Python: no model calls, no I/O.

Each finding has a category, and the category decides what happens next (see `retry.py`):
`not_a_policy` and `missing_source` escalate at once, `format` and `grounding` are retried,
`consistency` is retried once and then treated as the document's own discrepancy.
"""

import re
from collections.abc import Iterator, Set
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Final

from pydantic import BaseModel

from policy_extractor.config import Settings
from policy_extractor.derive import derive, name_key, normalise, parse_iso
from policy_extractor.models import (
    ALL_FIELDS,
    LIST_FIELDS,
    REQUIRED_LISTS,
    REQUIRED_SCALARS,
    DocumentKind,
    FieldStatus,
    LimitBasis,
    PolicyExtraction,
    PolicyType,
)


class Category(StrEnum):
    NOT_A_POLICY = "not_a_policy"
    MISSING_SOURCE = "missing_source"
    FORMAT = "format"
    CONSISTENCY = "consistency"
    GROUNDING = "grounding"


class Finding(BaseModel):
    field: str
    category: Category
    code: str
    message: str
    observed: str | None = None
    # Set by retry.py when a consistency finding survives a retry with the same values.
    document_discrepancy: bool = False


# Fields whose stated value needs no quote: the classification and a flag derived from absence.
UNQUOTED_FIELDS: Final = frozenset({"document_kind", "exclusions_complete"})
ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER: Final = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*((?:million|m|k)(?![a-z]))?", re.IGNORECASE)
MULTIPLIERS: Final = {"": 1, "k": 1_000, "m": 1_000_000, "million": 1_000_000}
MONTH_NAMES: Final = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
MONTHS_BY_NAME: Final = {
    **{name: number for number, name in enumerate(MONTH_NAMES, start=1)},
    **{name[:3]: number for number, name in enumerate(MONTH_NAMES, start=1)},
    "sept": 9,
}
# Date notations in the documents. Numeric dates are read month first: the carriers are US.
ISO_IN_TEXT: Final = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
NUMERIC_DATE: Final = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
MONTH_FIRST: Final = re.compile(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
DAY_FIRST: Final = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?,?\s+(\d{4})\b")
# Fields whose value must be readable from its own quote, not merely have a quote somewhere in
# the document: otherwise the agent's name could pass as the carrier with the header as "proof".
IDENTITY_FIELDS: Final = ("policy_number", "carrier", "policyholder", "policy_form")
CHANGED_ON_RETRY: Final = "changed_on_retry"


def validate(extraction: PolicyExtraction, document: str, settings: Settings) -> list[Finding]:
    record = extraction.record
    if record.document_kind is DocumentKind.NOT_A_POLICY:
        return [
            Finding(
                field="document_kind",
                category=Category.NOT_A_POLICY,
                code="not_a_policy",
                message="The document is not a policy declarations page.",
            )
        ]
    findings = [
        *_missing_source(extraction),
        *_format(extraction),
        *_consistency(extraction, settings),
        *_grounding(extraction, document),
    ]
    return findings


# ---------- missing_source ----------


def _missing_source(extraction: PolicyExtraction) -> Iterator[Finding]:
    values = extraction.values()
    for name in REQUIRED_SCALARS:
        if getattr(values, name) is None:
            yield Finding(
                field=name,
                category=Category.MISSING_SOURCE,
                code=f"{name}_absent",
                message=f"Required field {name} has no value: the document does not state it.",
            )
    for name in REQUIRED_LISTS:
        if not getattr(values, name):
            yield Finding(
                field=name,
                category=Category.MISSING_SOURCE,
                code=f"{name}_absent",
                message=f"Required list {name} is empty: the document states no limits.",
            )
    for evidence in extraction.evidence:
        if evidence.status is FieldStatus.REFERENCED_NOT_ATTACHED:
            yield Finding(
                field=evidence.field.value,
                category=Category.MISSING_SOURCE,
                code="referenced_not_attached",
                message=(
                    f"{evidence.field.value} is referenced to an attachment that is not part of "
                    "the document."
                ),
                observed=evidence.quote or None,
            )


# ---------- format ----------


def _format(extraction: PolicyExtraction) -> Iterator[Finding]:
    yield from _evidence_shape(extraction)
    yield from _status_matches_value(extraction)
    yield from _amount_ranges(extraction)
    record = extraction.record
    if record.policy_type is PolicyType.OTHER and not record.policy_type_detail.strip():
        yield _fmt("policy_type_detail", "detail_missing", "policy_type is other but no detail.")
    for name in ("effective_date", "expiration_date"):
        value: str = getattr(record, name)
        if value and (not ISO_DATE.fullmatch(value) or parse_iso(value) is None):
            yield _fmt(name, "date_not_iso", "Dates must be valid ISO YYYY-MM-DD.", value)
    if re.search(r"\s", record.policy_number):
        yield _fmt(
            "policy_number",
            "whitespace",
            "Remove spaces inside the policy number.",
            record.policy_number,
        )


def _fmt(field: str, code: str, message: str, observed: str | None = None) -> Finding:
    return Finding(
        field=field, category=Category.FORMAT, code=code, message=message, observed=observed
    )


def _evidence_shape(extraction: PolicyExtraction) -> Iterator[Finding]:
    counts = {name: 0 for name in ALL_FIELDS}
    for evidence in extraction.evidence:
        counts[evidence.field.value] += 1
        if not 0.0 <= evidence.confidence <= 1.0:
            yield _fmt(
                evidence.field.value,
                "confidence_range",
                "Confidence must be within 0-1.",
                str(evidence.confidence),
            )
    for name, count in counts.items():
        if count != 1:
            yield _fmt(name, "evidence_count", f"Give exactly one evidence entry, not {count}.")


def _status_matches_value(extraction: PolicyExtraction) -> Iterator[Finding]:
    record = extraction.record
    for evidence in extraction.evidence:
        name = evidence.field.value
        value = getattr(record, name)
        stated = evidence.status is FieldStatus.STATED
        if name in LIST_FIELDS:
            has_value = bool(value)
            if has_value and not stated:
                yield _fmt(name, "status_value_mismatch", "Items given but status is not stated.")
            for index, item in enumerate(value):
                if not isinstance(item, str) and not item.quote.strip():
                    yield _fmt(name, "quote_missing", f"Item {index} has no quote.")
            continue
        has_value = value is not None and value != ""
        if name == "exclusions_complete":
            continue
        if stated and not has_value:
            yield _fmt(name, "status_value_mismatch", "Status is stated but there is no value.")
        elif has_value and not stated:
            yield _fmt(
                name,
                "status_value_mismatch",
                f"A value is given but status is {evidence.status.value}.",
                str(value),
            )
        elif stated and name not in UNQUOTED_FIELDS and not evidence.quote.strip():
            yield _fmt(name, "quote_missing", "A stated value needs a verbatim quote.")


def _amount_ranges(extraction: PolicyExtraction) -> Iterator[Finding]:
    record = extraction.record
    if record.premium_amount is not None and record.premium_amount < 0:
        yield _fmt(
            "premium_amount", "negative", "Premium cannot be negative.", str(record.premium_amount)
        )
    for deductible in record.deductibles:
        if deductible.amount is not None and deductible.amount < 0:
            yield _fmt(
                "deductibles", "negative", "Deductible cannot be negative.", deductible.applies_to
            )
        if deductible.percent is not None:
            if not Decimal(0) < deductible.percent <= Decimal(100):
                yield _fmt(
                    "deductibles",
                    "percent_range",
                    "Percent must be in (0, 100].",
                    str(deductible.percent),
                )
            if not deductible.percent_of.strip():
                yield _fmt(
                    "deductibles",
                    "percent_of_missing",
                    "A percentage deductible needs percent_of.",
                    deductible.applies_to,
                )
    for limit in record.coverage_limits:
        if limit.amount is not None and limit.amount < 0:
            yield _fmt("coverage_limits", "negative", "Limit cannot be negative.", limit.coverage)
    for endorsement in record.endorsements:
        if endorsement.limit_amount is not None and endorsement.limit_amount < 0:
            yield _fmt(
                "endorsements",
                "negative",
                "Endorsement limit cannot be negative.",
                endorsement.name,
            )


# ---------- consistency ----------


def _consistency(extraction: PolicyExtraction, settings: Settings) -> Iterator[Finding]:
    record = extraction.record
    tolerance = settings.money_tolerance
    if record.premium_components and record.premium_amount is not None:
        total = sum((c.amount for c in record.premium_components), Decimal(0))
        if abs(total - record.premium_amount) > tolerance:
            yield _cons(
                "premium_amount",
                "components_sum_mismatch",
                f"Premium components sum to {total}, the stated premium is "
                f"{record.premium_amount}.",
                f"premium={record.premium_amount} components={total}",
            )
    start, end = parse_iso(record.effective_date), parse_iso(record.expiration_date)
    if start is not None and end is not None and end <= start:
        yield _cons(
            "expiration_date",
            "term_not_positive",
            "Expiration is not after the effective date.",
            f"{record.effective_date} -> {record.expiration_date}",
        )
    yield from _split_limits(extraction)
    values = extraction.values()
    derived = derive(values)
    if (
        derived.primary_deductible is not None
        and derived.primary_limit is not None
        and derived.primary_deductible > derived.primary_limit
    ):
        yield _cons(
            "deductibles",
            "deductible_exceeds_limit",
            "The primary deductible is larger than the primary limit.",
            f"deductible={derived.primary_deductible} limit={derived.primary_limit}",
        )
    stated = {
        d.applies_to: d.amount
        for d in values.deductibles or []
        if d.percent is not None and d.amount is not None
    }
    for computed in derived.percentage_deductibles:
        amount = stated.get(computed.applies_to)
        if amount is not None and abs(amount - computed.amount) > tolerance:
            yield _cons(
                "deductibles",
                "percentage_amount_mismatch",
                f"{computed.percent}% of {computed.base_coverage} is {computed.amount}, but "
                f"{amount} is stated.",
                f"{computed.applies_to}: stated={amount} computed={computed.amount}",
            )


def _cons(field: str, code: str, message: str, observed: str) -> Finding:
    return Finding(
        field=field, category=Category.CONSISTENCY, code=code, message=message, observed=observed
    )


def _split_limits(extraction: PolicyExtraction) -> Iterator[Finding]:
    by_coverage: dict[str, dict[LimitBasis, Decimal]] = {}
    for limit in extraction.record.coverage_limits:
        if limit.amount is not None:
            by_coverage.setdefault(normalise(limit.coverage), {})[limit.basis] = limit.amount
    for coverage, amounts in by_coverage.items():
        person = amounts.get(LimitBasis.PER_PERSON)
        accident = amounts.get(LimitBasis.PER_ACCIDENT)
        if person is not None and accident is not None and person > accident:
            yield _cons(
                "coverage_limits",
                "per_person_exceeds_per_accident",
                "A per-person limit cannot exceed the per-accident limit of the same coverage.",
                f"{coverage}: per_person={person} per_accident={accident}",
            )


# ---------- grounding ----------


def flatten(text: str) -> str:
    """Whitespace- and dot-leader-insensitive, case-insensitive form for quote matching."""
    without_leaders = re.sub(r"\.{2,}", " ", text)
    return " ".join(without_leaders.split()).casefold()


def numbers_in(text: str) -> set[Decimal]:
    """Every number in the text, with "$1M" and "250k" also read as their full amounts."""
    found: set[Decimal] = set()
    for digits, suffix in NUMBER.findall(text):
        number = Decimal(digits.replace(",", ""))
        found.add(number)
        found.add(number * MULTIPLIERS[suffix.casefold()])
    return found


def _grounding(extraction: PolicyExtraction, document: str) -> Iterator[Finding]:
    flat_document = flatten(document)
    record = extraction.record

    def grounded(quote: str) -> bool:
        return flatten(quote) in flat_document

    for evidence in extraction.evidence:
        name = evidence.field.value
        if evidence.status is not FieldStatus.STATED or not evidence.quote.strip():
            continue
        if name in LIST_FIELDS:
            continue  # list items are checked one by one below
        if not grounded(evidence.quote):
            yield _ground(
                name, "quote_not_in_document", "The quote is not in the document.", evidence.quote
            )
            continue
        if name == "premium_amount" and record.premium_amount is not None:
            yield from _number_in_quote(name, record.premium_amount, evidence.quote)
        if name in ("effective_date", "expiration_date"):
            yield from _date_in_quote(name, getattr(record, name), evidence.quote)
        if name in IDENTITY_FIELDS:
            yield from _value_in_quote(name, getattr(record, name), evidence.quote)

    items: list[tuple[str, str, list[Decimal]]] = []
    items += [("premium_components", c.quote, [abs(c.amount)]) for c in record.premium_components]
    items += [
        ("deductibles", d.quote, [v for v in (d.amount, d.percent) if v is not None])
        for d in record.deductibles
    ]
    items += [
        ("coverage_limits", c.quote, [] if c.amount is None else [c.amount])
        for c in record.coverage_limits
    ]
    items += [("underlying_requirements", u.quote, []) for u in record.underlying_requirements]
    items += [
        ("endorsements", e.quote, [] if e.limit_amount is None else [e.limit_amount])
        for e in record.endorsements
    ]
    for name, quote, numbers in items:
        if not quote.strip():
            continue  # reported as a format finding
        if not grounded(quote):
            yield _ground(name, "quote_not_in_document", "The quote is not in the document.", quote)
            continue
        for number in numbers:
            yield from _number_in_quote(name, number, quote)
    for exclusion in record.exclusions:
        if not grounded(exclusion):
            yield _ground(
                "exclusions",
                "text_not_in_document",
                "The exclusion text is not in the document.",
                exclusion,
            )


def _ground(field: str, code: str, message: str, observed: str) -> Finding:
    return Finding(
        field=field, category=Category.GROUNDING, code=code, message=message, observed=observed
    )


def _number_in_quote(field: str, value: Decimal, quote: str) -> Iterator[Finding]:
    """The value must be readable from its own quote: the guard against changed numbers."""
    found = numbers_in(quote)
    lowered = quote.casefold()
    if value in found:
        return
    if "thousand" in lowered and value / 1000 in found:
        return
    if value == 0 and any(word in lowered for word in ("nil", "zero", "none")):
        return
    yield _ground(
        field,
        "number_not_in_quote",
        f"{value} cannot be read from its quote.",
        f"{value} <- {quote!r}",
    )


def dates_in(text: str) -> set[date]:
    """Every date written in the text: 2025-03-15, 03/15/2025, 03-15-2025, March 15, 2025,
    Mar. 15th 2025 or 15 March 2025. Character noise ("2O25") makes a date unreadable."""
    lowered = text.casefold()
    candidates: list[tuple[int, int, int]] = []
    candidates += [(int(y), int(m), int(d)) for y, m, d in ISO_IN_TEXT.findall(lowered)]
    candidates += [(int(y), int(m), int(d)) for m, d, y in NUMERIC_DATE.findall(lowered)]
    for name, day, year in MONTH_FIRST.findall(lowered):
        if name in MONTHS_BY_NAME:
            candidates.append((int(year), MONTHS_BY_NAME[name], int(day)))
    for day, name, year in DAY_FIRST.findall(lowered):
        if name in MONTHS_BY_NAME:
            candidates.append((int(year), MONTHS_BY_NAME[name], int(day)))
    found: set[date] = set()
    for year, month, day in candidates:
        try:
            found.add(date(year, month, day))
        except ValueError:
            continue
    return found


def _date_in_quote(field: str, value: str, quote: str) -> Iterator[Finding]:
    """The date must be written in the quote as a whole date, not just as scattered numbers:
    otherwise a swapped day and month (01/12 for 12/01) would pass."""
    parsed = parse_iso(value)
    if parsed is None:
        return  # reported as a format finding
    if parsed not in dates_in(quote):
        yield _ground(
            field,
            "date_not_in_quote",
            f"{value} cannot be read from its quote.",
            f"{value} <- {quote!r}",
        )


def _alphanumeric(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def _value_in_quote(field: str, value: str, quote: str) -> Iterator[Finding]:
    """The value must be what the quote says, word for word (case, punctuation and common
    abbreviations aside); a policy number may lose its spaces and dashes."""
    if not value.strip():
        return  # reported as a format finding when the status says stated
    if field == "policy_number":
        readable = _alphanumeric(value) in _alphanumeric(quote)
    else:
        readable = f" {name_key(value)} " in f" {name_key(quote)} "
    if not readable:
        yield _ground(
            field,
            "value_not_in_quote",
            f"{field} {value!r} cannot be read from its quote.",
            f"{value} <- {quote!r}",
        )


# ---------- changes between retries ----------


def _comparable(value: object) -> object:
    """Lists compare as multisets: reordering items is not a change of what the document says."""
    if isinstance(value, list):
        return sorted(
            item.model_dump_json() if isinstance(item, BaseModel) else str(item) for item in value
        )
    return value


def _short(value: object) -> str:
    text = value.model_dump_json() if isinstance(value, BaseModel) else str(_comparable(value))
    return text if len(text) <= 160 else text[:157] + "..."


def changed_on_retry(
    baseline: PolicyExtraction, current: PolicyExtraction, allowed: Set[str]
) -> list[Finding]:
    """Fields a retry changed although no format or grounding finding asked about them.

    The first reading is the baseline. A retry may fix what a finding named; anything else it
    changes, including a dropped list item that makes a premium add up, is a grounding failure:
    "never let a retry fix the numbers so the check passes" (CLAUDE.md §6).
    """
    before, after = baseline.values(), current.values()
    findings: list[Finding] = []
    for name in ALL_FIELDS:
        if name in allowed:
            continue
        old, new = getattr(before, name), getattr(after, name)
        if _comparable(old) != _comparable(new):
            findings.append(
                _ground(
                    name,
                    CHANGED_ON_RETRY,
                    f"{name} changed on retry although no finding asked for it. Keep the value "
                    "as the document states it.",
                    f"{_short(old)} -> {_short(new)}",
                )
            )
    return findings
