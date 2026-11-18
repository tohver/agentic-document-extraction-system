"""Hand-built fakes shared by the offline tests. Nothing here touches the network."""

from collections.abc import Iterable
from dataclasses import replace
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from policy_extractor.config import Settings
from policy_extractor.llm import LLMRequest, LLMResponse, Usage
from policy_extractor.models import (
    ALL_FIELDS,
    LIST_FIELDS,
    CoverageLimit,
    Deductible,
    DocumentKind,
    Endorsement,
    ExtractionPatch,
    FieldEvidence,
    FieldName,
    FieldStatus,
    FieldVerdict,
    LimitBasis,
    PolicyExtraction,
    PolicyRecord,
    PolicyType,
    ReviewOutput,
    Verdict,
)

DOCUMENT = """\
TESTLINE MUTUAL INSURANCE
Personal Auto Renewal Declarations
Policy Number: TP-100
Named Insured: Jane Q. Tester
Policy Period: 01/01/2026 to 01/01/2027
Annual Premium: $ 1,200.00
Bodily Injury: $100,000 / $300,000
Property Damage: $50,000 per accident
Deductible (Primary): $500
Endorsements: Roadside Assistance (TP-RA)
Exclusions:
 - Racing
 - Intentional acts
"""

QUOTES: dict[str, str] = {
    "document_kind": "Personal Auto Renewal Declarations",
    "policy_number": "Policy Number: TP-100",
    "carrier": "TESTLINE MUTUAL INSURANCE",
    "policyholder": "Named Insured: Jane Q. Tester",
    "policy_type": "Personal Auto",
    "effective_date": "Policy Period: 01/01/2026 to 01/01/2027",
    "expiration_date": "Policy Period: 01/01/2026 to 01/01/2027",
    "premium_amount": "Annual Premium: $ 1,200.00",
}


def settings(**overrides: Any) -> Settings:
    """Settings that ignore the developer's environment and `.env`."""
    return Settings(_env_file=None, **overrides)


def deductible(**overrides: Any) -> Deductible:
    values: dict[str, Any] = {
        "applies_to": "Primary",
        "amount": Decimal("500"),
        "percent": None,
        "percent_of": "",
        "labelled_primary": True,
        "quote": "Deductible (Primary): $500",
    }
    return Deductible(**(values | overrides))


def limit(coverage: str, amount: str | None, basis: LimitBasis, quote: str) -> CoverageLimit:
    return CoverageLimit(
        coverage=coverage,
        amount=None if amount is None else Decimal(amount),
        basis=basis,
        note="",
        quote=quote,
    )


def default_limits() -> list[CoverageLimit]:
    bodily = "Bodily Injury: $100,000 / $300,000"
    return [
        limit("Bodily Injury", "100000", LimitBasis.PER_PERSON, bodily),
        limit("Bodily Injury", "300000", LimitBasis.PER_ACCIDENT, bodily),
        limit(
            "Property Damage",
            "50000",
            LimitBasis.PER_ACCIDENT,
            "Property Damage: $50,000 per accident",
        ),
    ]


def make_record(**overrides: Any) -> PolicyRecord:
    values: dict[str, Any] = {
        "document_kind": DocumentKind.POLICY_DECLARATIONS,
        "policy_number": "TP-100",
        "carrier": "TESTLINE MUTUAL INSURANCE",
        "policyholder": "Jane Q. Tester",
        "policy_type": PolicyType.AUTO,
        "policy_type_detail": "",
        "policy_form": "",
        "effective_date": "2026-01-01",
        "expiration_date": "2027-01-01",
        "premium_amount": Decimal("1200.00"),
        "premium_components": [],
        "deductibles": [deductible()],
        "coverage_limits": default_limits(),
        "underlying_requirements": [],
        "endorsements": [
            Endorsement(
                name="Roadside Assistance",
                form_code="TP-RA",
                limit_amount=None,
                limit_text="",
                quote="Roadside Assistance (TP-RA)",
            )
        ],
        "exclusions": ["Racing", "Intentional acts"],
        "exclusions_complete": True,
    }
    return PolicyRecord(**(values | overrides))


def _has_value(record: PolicyRecord, name: str) -> bool:
    value = getattr(record, name)
    if name in LIST_FIELDS:
        return bool(value)
    return value is not None and value != ""


def make_extraction(
    *,
    confidence: float | dict[str, float] = 0.99,
    quotes: dict[str, str] | None = None,
    statuses: dict[str, FieldStatus] | None = None,
    **record_overrides: Any,
) -> PolicyExtraction:
    """A clean, grounded extraction of DOCUMENT. A field emptied by an override is not_stated."""
    record = make_record(**record_overrides)
    merged_quotes = QUOTES | (quotes or {})
    evidence = []
    for name in ALL_FIELDS:
        always_stated = name in ("exclusions_complete", "document_kind")
        has_value = _has_value(record, name) or always_stated
        default_status = FieldStatus.STATED if has_value else FieldStatus.NOT_STATED
        status = (statuses or {}).get(name, default_status)
        evidence.append(
            FieldEvidence(
                field=FieldName(name),
                status=status,
                quote=merged_quotes.get(name, "") if status is FieldStatus.STATED else "",
                confidence=confidence.get(name, 0.99)
                if isinstance(confidence, dict)
                else confidence,
            )
        )
    return PolicyExtraction(record=record, evidence=evidence)


def make_review(disagree: dict[str, str] | None = None, omit: Iterable[str] = ()) -> ReviewOutput:
    disagree = disagree or {}
    return ReviewOutput(
        verdicts=[
            FieldVerdict(
                field=FieldName(name),
                verdict=Verdict.DISAGREE if name in disagree else Verdict.AGREE,
                reason=disagree.get(name, ""),
            )
            for name in ALL_FIELDS
            if name not in set(omit)
        ]
    )


def respond(
    parsed: BaseModel | None,
    stop_reason: str = "end_turn",
    model: str = "fake-model",
    usage: Usage | None = None,
) -> LLMResponse[Any]:
    return LLMResponse(
        parsed=parsed,
        stop_reason=stop_reason,
        model=model,
        request_id="req_fake",
        usage=usage or Usage(input_tokens=1000, output_tokens=200),
        latency_s=0.5,
    )


class FakeClient:
    """Returns scripted responses in order and records every request it receives."""

    def __init__(self, responses: Iterable[LLMResponse[Any]]) -> None:
        self._responses = iter(responses)
        self.requests: list[LLMRequest] = []

    def parse(self, request: LLMRequest, output_type: type[BaseModel]) -> LLMResponse[Any]:
        self.requests.append(request)
        try:
            response = next(self._responses)
        except StopIteration:
            raise AssertionError(
                f"unexpected extra {request.role} call #{len(self.requests)}"
            ) from None
        if output_type is ExtractionPatch and isinstance(response.parsed, PolicyExtraction):
            # A scripted whole extraction stands for "the retry now reads the document like this".
            return replace(response, parsed=full_patch(response.parsed))
        return response


def full_patch(extraction: PolicyExtraction) -> ExtractionPatch:
    """A retry answer that restates every field: applied, it yields `extraction` again."""
    return ExtractionPatch(
        changed_fields=[FieldName(name) for name in ALL_FIELDS],
        record=extraction.record,
        evidence=list(extraction.evidence),
    )


# What a model puts in the fields a patch does not change. `exclusions_complete` is deliberately
# the opposite of the fake extraction's value, so a test notices if filler is ever applied.
PATCH_FILLER: dict[str, Any] = {
    **{name: [] for name in LIST_FIELDS},
    **{name: "" for name in ALL_FIELDS if name not in LIST_FIELDS},
    "document_kind": DocumentKind.POLICY_DECLARATIONS,
    "policy_type": None,
    "premium_amount": None,
    "exclusions_complete": False,
}


def patch(*, evidence: Iterable[FieldEvidence] = (), **changes: Any) -> ExtractionPatch:
    """A retry answer that changes only `changes` (and replaces the given evidence entries)."""
    return ExtractionPatch(
        changed_fields=[FieldName(name) for name in changes],
        record=PolicyRecord(**(PATCH_FILLER | changes)),
        evidence=list(evidence),
    )
