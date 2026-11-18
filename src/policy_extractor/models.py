"""Domain models and the structured-output contract (CLAUDE.md §4).

The extractor returns a flat `PolicyRecord` plus one `FieldEvidence` per field (status, verbatim
quote, confidence). This shape is deliberate: wrapping every field in its own object with nullable
members made the API reject the schema ("compiled grammar is too large"). So on the wire:
- absent text is "", absent lists are [], and `FieldEvidence.status` says why (`not_stated`,
  `referenced_not_attached`, `illegible`); only `policy_type` and `premium_amount` are nullable;
- money and percentages are decimal *strings*, so no float rounding creeps in;
- dates are ISO strings (`YYYY-MM-DD`), so a malformed date becomes a finding, not a schema error;
- range and format rules live in `validate.py`, never in the schema.

`PolicyExtraction.values()` turns the wire shape into `PolicyValues`, where "the document does not
say" is `None` again. Gold labels and the reviewer use `PolicyValues`.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, WithJsonSchema

Money = Annotated[
    Decimal,
    WithJsonSchema({"type": "string", "description": 'Decimal number as a string, e.g. "1234.50"'}),
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentKind(StrEnum):
    POLICY_DECLARATIONS = "policy_declarations"
    NOT_A_POLICY = "not_a_policy"


class PolicyType(StrEnum):
    AUTO = "auto"
    HOME = "home"
    UMBRELLA = "umbrella"
    OTHER = "other"


class LimitBasis(StrEnum):
    PER_PERSON = "per_person"
    PER_ACCIDENT = "per_accident"
    PER_OCCURRENCE = "per_occurrence"
    AGGREGATE = "aggregate"
    SCHEDULED_TOTAL = "scheduled_total"
    OTHER = "other"


class FieldStatus(StrEnum):
    """Why a field has or lacks a value. Only `stated` may carry a value."""

    STATED = "stated"
    NOT_STATED = "not_stated"
    REFERENCED_NOT_ATTACHED = "referenced_not_attached"
    ILLEGIBLE = "illegible"


# ---------- List items: value models (gold, reviewer) and quoted wire models (extractor) ----------


class PremiumComponentValue(Strict):
    name: str
    amount: Money


class PremiumComponent(PremiumComponentValue):
    quote: str


class DeductibleValue(Strict):
    applies_to: str
    amount: Money | None
    percent: Money | None
    percent_of: str | None
    labelled_primary: bool


class Deductible(Strict):
    applies_to: str
    amount: Money | None
    percent: Money | None
    percent_of: str
    labelled_primary: bool
    quote: str


class CoverageLimitValue(Strict):
    coverage: str
    amount: Money | None
    basis: LimitBasis
    note: str | None


class CoverageLimit(Strict):
    coverage: str
    amount: Money | None
    basis: LimitBasis
    note: str
    quote: str


class UnderlyingRequirementValue(Strict):
    coverage: str
    limits_text: str


class UnderlyingRequirement(UnderlyingRequirementValue):
    quote: str


class EndorsementValue(Strict):
    name: str
    form_code: str | None
    limit_amount: Money | None
    limit_text: str | None


class Endorsement(Strict):
    name: str
    form_code: str
    limit_amount: Money | None
    limit_text: str
    quote: str


# ---------- Field names ----------


class FieldName(StrEnum):
    """Every field of the §4.2 contract, in contract order."""

    DOCUMENT_KIND = "document_kind"
    POLICY_NUMBER = "policy_number"
    CARRIER = "carrier"
    POLICYHOLDER = "policyholder"
    POLICY_TYPE = "policy_type"
    POLICY_TYPE_DETAIL = "policy_type_detail"
    POLICY_FORM = "policy_form"
    EFFECTIVE_DATE = "effective_date"
    EXPIRATION_DATE = "expiration_date"
    PREMIUM_AMOUNT = "premium_amount"
    PREMIUM_COMPONENTS = "premium_components"
    DEDUCTIBLES = "deductibles"
    COVERAGE_LIMITS = "coverage_limits"
    UNDERLYING_REQUIREMENTS = "underlying_requirements"
    ENDORSEMENTS = "endorsements"
    EXCLUSIONS = "exclusions"
    EXCLUSIONS_COMPLETE = "exclusions_complete"


ALL_FIELDS: Final[tuple[str, ...]] = tuple(f.value for f in FieldName)
LIST_FIELDS: Final[tuple[str, ...]] = (
    "premium_components",
    "deductibles",
    "coverage_limits",
    "underlying_requirements",
    "endorsements",
    "exclusions",
)
SCALAR_FIELDS: Final[tuple[str, ...]] = tuple(f for f in ALL_FIELDS if f not in LIST_FIELDS)
TEXT_FIELDS: Final[tuple[str, ...]] = (
    "policy_number",
    "carrier",
    "policyholder",
    "policy_type_detail",
    "policy_form",
    "effective_date",
    "expiration_date",
)
REQUIRED_SCALARS: Final[tuple[str, ...]] = (
    "document_kind",
    "policy_number",
    "carrier",
    "policyholder",
    "policy_type",
    "effective_date",
    "expiration_date",
    "premium_amount",
)
REQUIRED_LISTS: Final[tuple[str, ...]] = ("coverage_limits",)


# ---------- The extraction (wire shape) ----------


class PolicyRecord(Strict):
    """The values as the extractor returns them: "" and [] stand for "no value"."""

    document_kind: DocumentKind
    policy_number: str
    carrier: str
    policyholder: str
    policy_type: PolicyType | None
    policy_type_detail: str
    policy_form: str
    effective_date: str
    expiration_date: str
    premium_amount: Money | None
    premium_components: list[PremiumComponent]
    deductibles: list[Deductible]
    coverage_limits: list[CoverageLimit]
    underlying_requirements: list[UnderlyingRequirement]
    endorsements: list[Endorsement]
    exclusions: list[str]
    exclusions_complete: bool


class FieldEvidence(Strict):
    """The extractor's claim about one field: why it has (or lacks) a value, where it read it,
    and how sure it is. For list fields `quote` may be "", because every item has its own."""

    field: FieldName
    status: FieldStatus
    quote: str
    confidence: float


class PolicyExtraction(Strict):
    """What the extractor returns: the record plus one evidence entry per field."""

    record: PolicyRecord
    evidence: list[FieldEvidence]

    def evidence_for(self, field: str) -> FieldEvidence | None:
        return next((e for e in self.evidence if e.field.value == field), None)

    def status(self, field: str) -> FieldStatus | None:
        evidence = self.evidence_for(field)
        return None if evidence is None else evidence.status

    def values(self) -> PolicyValues:
        """Stated values only: a field whose status is not `stated` is None, "" becomes None."""
        dumped = self.record.model_dump()
        data: dict[str, object] = {}
        for name in ALL_FIELDS:
            value = dumped[name]
            if self.status(name) not in (FieldStatus.STATED, None):
                data[name] = None
            elif name in TEXT_FIELDS:
                data[name] = value or None
            elif name in LIST_FIELDS and name != "exclusions":
                data[name] = [_strip_item(item) for item in value]
            else:
                data[name] = value
        return PolicyValues.model_validate(data)


# ---------- Retry patch (wire shape of a retry answer) ----------


class ExtractionPatch(Strict):
    """A retry answer: only what changes. Re-sending the whole record cost as many output tokens
    as the first extraction, and gave the model a chance to drift on fields nobody asked about.

    `record` has the same shape as a first answer, because a schema that makes every field
    nullable is rejected by the API ("compiled grammar is too large"). Only the fields listed in
    `changed_fields` are read from it; the model fills the others with empty values.
    """

    changed_fields: list[FieldName]
    record: PolicyRecord
    evidence: list[FieldEvidence]


def apply_patch(previous: PolicyExtraction, patch: ExtractionPatch) -> PolicyExtraction:
    """The previous extraction with the fields in `changed_fields` taken from the patch.

    Evidence entries in the patch replace the previous entry for the same field, so a retry can
    fix a quote without changing the value.
    """
    record = previous.record.model_dump()
    new_values = patch.record.model_dump()
    for field in patch.changed_fields:
        record[field.value] = new_values[field.value]
    evidence = {entry.field: entry for entry in previous.evidence}
    for entry in patch.evidence:
        evidence[entry.field] = entry
    return PolicyExtraction(
        record=PolicyRecord.model_validate(record), evidence=list(evidence.values())
    )


def _strip_item(item: dict[str, object]) -> dict[str, object]:
    return {key: (None if val == "" else val) for key, val in item.items() if key != "quote"}


class PolicyValues(Strict):
    """Stated values only. Used for gold labels and as the reviewer's view of the extraction."""

    document_kind: DocumentKind | None
    policy_number: str | None
    carrier: str | None
    policyholder: str | None
    policy_type: PolicyType | None
    policy_type_detail: str | None
    policy_form: str | None
    effective_date: str | None
    expiration_date: str | None
    premium_amount: Money | None
    premium_components: list[PremiumComponentValue] | None
    deductibles: list[DeductibleValue] | None
    coverage_limits: list[CoverageLimitValue] | None
    underlying_requirements: list[UnderlyingRequirementValue] | None
    endorsements: list[EndorsementValue] | None
    exclusions: list[str] | None
    exclusions_complete: bool | None


# ---------- Review ----------


class Verdict(StrEnum):
    AGREE = "agree"
    DISAGREE = "disagree"
    NOT_VERIFIABLE = "not_verifiable"


class FieldVerdict(Strict):
    field: FieldName
    verdict: Verdict
    reason: str


class ReviewOutput(Strict):
    """Reviewer output: one verdict per §4.2 field. `reason` quotes the document on disagreement."""

    verdicts: list[FieldVerdict]

    def verdict_for(self, field: str) -> FieldVerdict | None:
        return next((v for v in self.verdicts if v.field.value == field), None)
