"""Evaluation against gold labels (CLAUDE.md §9). Pure functions over finished results.

The headline numbers are not "accuracy" alone: the question is whether the routing was right.
A wrong field that was auto-approved is the costly failure, so it is counted separately, and
calibration is sliced by policy type x field so a weak slice cannot hide behind a good average.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel

from policy_extractor.config import Settings
from policy_extractor.derive import Derived, derive, name_key, normalise
from policy_extractor.models import ALL_FIELDS, PolicyValues, Strict, Verdict
from policy_extractor.pipeline import DocumentResult
from policy_extractor.route import Escalation, Outcome

ExpectedOutcome = Literal["auto_approve", "human_review", "escalated"]


class GoldLabel(Strict):
    """What a careful person would extract from one document, and how it should be routed."""

    doc_id: str
    expected_outcome: ExpectedOutcome
    expected_escalation: Escalation | None = None
    why: str
    values: PolicyValues


# ---------- Field comparison: every rule lives here ----------

# Name-like fields of each list item: matched leniently, because "Comprehensive" and
# "Comprehensive deductible" name the same thing. Every other item field must match exactly
# (money within tolerance), except free-text notes, whose wording is not an extraction error.
LIST_NAME_FIELDS: dict[str, str] = {
    "premium_components": "name",
    "deductibles": "applies_to",
    "coverage_limits": "coverage",
    "underlying_requirements": "coverage",
    "endorsements": "name",
}
FREE_TEXT_ITEM_FIELDS = frozenset({"note", "limits_text", "limit_text", "percent_of"})
# Parties are compared word for word (after case, punctuation and abbreviations): dropping a
# joint insured, the word "Trust" or the company's distinctive name is exactly the error to catch.
PARTY_FIELDS = frozenset({"carrier", "policyholder"})
# A product description, where a qualifier more or less is not an extraction error.
LENIENT_TEXT_FIELDS = frozenset({"policy_type_detail"})


def _text(value: object) -> str:
    return normalise(str(value)) if value not in (None, "") else ""


def same_party(expected: str | None, actual: str | None) -> bool:
    """Both absent, or the same name once case, punctuation and abbreviations are ignored."""
    if expected is None or actual is None:
        return expected is None and actual is None
    return name_key(expected) == name_key(actual)


def same_name(expected: object, actual: object) -> bool:
    """For list-item names and exclusions, where "Comprehensive" and "Comprehensive deductible"
    name the same thing: equal after normalising and expanding abbreviations, one contained in
    the other, or most words shared. Never used for the policy's parties."""
    a, b = name_key(_text(expected)), name_key(_text(actual))
    if a == b or (a and b and (a in b or b in a)):
        return True
    words_a, words_b = set(a.split()), set(b.split())
    if not words_a or not words_b:
        return False
    return len(words_a & words_b) / len(words_a | words_b) >= 0.5


def _same(expected: object, actual: object, tolerance: Decimal) -> bool:
    if isinstance(expected, Decimal) or isinstance(actual, Decimal):
        if expected is None or actual is None:
            return expected is actual
        return abs(Decimal(str(expected)) - Decimal(str(actual))) <= tolerance
    if isinstance(expected, str) or isinstance(actual, str):
        return _text(expected) == _text(actual)
    return expected == actual


def _items_match(
    name_field: str, want: Mapping[str, Any], have: Mapping[str, Any], tol: Decimal
) -> bool:
    if not same_name(want[name_field], have[name_field]):
        return False
    return all(
        _same(value, have[field], tol)
        for field, value in want.items()
        if field != name_field and field not in FREE_TEXT_ITEM_FIELDS
    )


def compare_list(
    name: str, expected: list[Any] | None, actual: list[Any] | None, tol: Decimal
) -> bool:
    """Same items, matched one to one; order does not matter. No list and [] are the same."""
    expected, actual = expected or [], actual or []
    if len(expected) != len(actual):
        return False
    if name == "exclusions":
        remaining = list(actual)
        for text in expected:
            match = next((a for a in remaining if same_name(text, a)), None)
            if match is None:
                return False
            remaining.remove(match)
        return True
    name_field = LIST_NAME_FIELDS[name]
    unmatched = [item.model_dump() for item in actual]
    for item in (i.model_dump() for i in expected):
        match = next((h for h in unmatched if _items_match(name_field, item, h, tol)), None)
        if match is None:
            return False
        unmatched.remove(match)
    return True


def compare_field(name: str, expected: PolicyValues, actual: PolicyValues, tol: Decimal) -> bool:
    want, have = getattr(expected, name), getattr(actual, name)
    if name in LIST_NAME_FIELDS or name == "exclusions":
        return compare_list(name, want, have, tol)
    if name in ("policy_number", "policy_form"):
        # Identifiers: "HO-3" and "HO 3" are one code, "HO-3" and "HO-5" are not.
        return _text(want).replace(" ", "") == _text(have).replace(" ", "")
    if name in PARTY_FIELDS:
        return same_party(want, have)
    if name in LENIENT_TEXT_FIELDS:
        return (want is None) == (have is None) and (want is None or same_name(want, have))
    return _same(want, have, tol)


def compare_derived(expected: Derived, actual: Derived, tol: Decimal) -> dict[str, bool]:
    return {
        "primary_limit": _same(expected.primary_limit, actual.primary_limit, tol),
        "primary_deductible": _same(expected.primary_deductible, actual.primary_deductible, tol),
        "term_months": expected.term_months == actual.term_months,
        "percentage_deductibles": sorted(p.amount for p in expected.percentage_deductibles)
        == sorted(p.amount for p in actual.percentage_deductibles),
    }


# ---------- Scoring ----------


@dataclass
class FieldScore:
    doc_id: str
    policy_type: str
    field: str
    correct: bool
    confidence: float | None
    verdict: Verdict | None
    # Whether gold says the document states a value. Fields that are empty by construction
    # (underlying insurance on an auto policy) are easy to get right and inflate accuracy.
    gold_stated: bool


class CalibrationCell(BaseModel):
    policy_type: str
    field: str
    n: int
    mean_confidence: float
    accuracy: float
    brier: float
    flagged: bool


class Evaluation(BaseModel):
    documents: int
    scored_documents: int
    field_accuracy: dict[str, float]
    field_accuracy_stated: dict[str, float | None]
    field_count_stated: dict[str, int]
    derived_accuracy: dict[str, float]
    outcome_matrix: dict[str, dict[str, int]]
    outcome_correct: int
    escalation_correct: int
    escalations_expected: int
    auto_approved: int
    auto_approved_with_errors: list[str]
    calibration: list[CalibrationCell]
    overall_brier: float
    overall_mean_confidence: float
    overall_accuracy: float
    overall_accuracy_stated: float
    reviewer: dict[str, int]
    cost_usd: Decimal
    input_tokens: int
    output_tokens: int
    per_document: list[dict[str, Any]]


def _final_outcome(result: DocumentResult) -> str:
    outcome = result.decision.outcome
    return Outcome.AUTO_APPROVE.value if outcome is Outcome.SPOT_CHECK else outcome.value


def evaluate(
    results: Sequence[DocumentResult], gold: Mapping[str, GoldLabel], settings: Settings
) -> Evaluation:
    """Score every result against its gold label. A result without a label is an error: skipping
    it would quietly shrink the evaluation set while the report still counts it."""
    missing = sorted(r.doc_id for r in results if r.doc_id not in gold)
    if missing:
        raise ValueError(f"no gold label for: {', '.join(missing)}")
    tol = settings.money_tolerance
    scores: list[FieldScore] = []
    derived_hits: dict[str, list[bool]] = defaultdict(list)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_document: list[dict[str, Any]] = []
    outcome_correct = escalation_correct = escalations_expected = auto_approved = 0
    wrong_auto: list[str] = []

    for result in results:
        label = gold[result.doc_id]
        actual = _final_outcome(result)
        matrix[label.expected_outcome][actual] += 1
        outcome_correct += actual == label.expected_outcome
        if label.expected_outcome == "escalated":
            escalations_expected += 1
            escalation_correct += result.decision.escalation == label.expected_escalation
        wrong_fields: list[str] = []
        if (
            result.extraction is not None
            and label.expected_escalation is not Escalation.NOT_A_POLICY
        ):
            values = result.extraction.values()
            policy_type = (label.values.policy_type or "unknown").__str__()
            for field in ALL_FIELDS:
                correct = compare_field(field, label.values, values, tol)
                evidence = result.extraction.evidence_for(field)
                verdict = result.review.verdict_for(field) if result.review else None
                scores.append(
                    FieldScore(
                        doc_id=result.doc_id,
                        policy_type=policy_type,
                        field=field,
                        correct=correct,
                        confidence=None if evidence is None else evidence.confidence,
                        verdict=None if verdict is None else verdict.verdict,
                        gold_stated=getattr(label.values, field) not in (None, "", []),
                    )
                )
                if not correct:
                    wrong_fields.append(field)
            for name, hit in compare_derived(derive(label.values), derive(values), tol).items():
                derived_hits[name].append(hit)
        if actual == Outcome.AUTO_APPROVE.value:
            auto_approved += 1
            if wrong_fields:
                wrong_auto.append(result.doc_id)
        per_document.append(
            {
                "doc_id": result.doc_id,
                "expected": label.expected_outcome
                + (f" ({label.expected_escalation.value})" if label.expected_escalation else ""),
                "actual": actual
                + (f" ({result.decision.escalation.value})" if result.decision.escalation else ""),
                "wrong_fields": wrong_fields,
            }
        )

    calls = [c for r in results for c in r.calls]
    calibration, brier, mean_conf, accuracy = _calibration(scores, settings.calibration_gap)
    stated = [s for s in scores if s.gold_stated]
    return Evaluation(
        documents=len(results),
        scored_documents=len({s.doc_id for s in scores}),
        field_accuracy={
            field: _rate([s.correct for s in scores if s.field == field]) for field in ALL_FIELDS
        },
        field_accuracy_stated={
            field: _rate(hits)
            if (hits := [s.correct for s in stated if s.field == field])
            else None
            for field in ALL_FIELDS
        },
        field_count_stated={field: sum(s.field == field for s in stated) for field in ALL_FIELDS},
        overall_accuracy_stated=_rate([s.correct for s in stated]),
        derived_accuracy={name: _rate(hits) for name, hits in derived_hits.items()},
        outcome_matrix={k: dict(v) for k, v in matrix.items()},
        outcome_correct=outcome_correct,
        escalation_correct=escalation_correct,
        escalations_expected=escalations_expected,
        auto_approved=auto_approved,
        auto_approved_with_errors=wrong_auto,
        calibration=calibration,
        overall_brier=brier,
        overall_mean_confidence=mean_conf,
        overall_accuracy=accuracy,
        reviewer=_reviewer_value(scores),
        cost_usd=sum((c.cost_usd for c in calls if c.cost_usd is not None), Decimal(0)),
        input_tokens=sum(c.usage.input_tokens for c in calls),
        output_tokens=sum(c.usage.output_tokens for c in calls),
        per_document=per_document,
    )


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _calibration(
    scores: Sequence[FieldScore], gap: float
) -> tuple[list[CalibrationCell], float, float, float]:
    rated = [s for s in scores if s.confidence is not None]
    cells: dict[tuple[str, str], list[FieldScore]] = defaultdict(list)
    for score in rated:
        cells[(score.policy_type, score.field)].append(score)
    out: list[CalibrationCell] = []
    for (policy_type, field), group in sorted(cells.items()):
        confidence = [float(s.confidence or 0.0) for s in group]
        outcomes = [1.0 if s.correct else 0.0 for s in group]
        mean_conf = sum(confidence) / len(group)
        accuracy = sum(outcomes) / len(group)
        out.append(
            CalibrationCell(
                policy_type=policy_type,
                field=field,
                n=len(group),
                mean_confidence=mean_conf,
                accuracy=accuracy,
                brier=sum((c - o) ** 2 for c, o in zip(confidence, outcomes, strict=True))
                / len(group),
                flagged=mean_conf - accuracy >= gap,
            )
        )
    if not rated:
        return out, 0.0, 0.0, 0.0
    brier = sum(((s.confidence or 0.0) - s.correct) ** 2 for s in rated) / len(rated)
    mean_conf = sum(s.confidence or 0.0 for s in rated) / len(rated)
    return out, brier, mean_conf, _rate([s.correct for s in rated])


def _reviewer_value(scores: Sequence[FieldScore]) -> dict[str, int]:
    counts = {
        "caught": 0,
        "missed": 0,
        "false_dispute": 0,
        "confirmed": 0,
        "not_verifiable": 0,
    }
    for score in scores:
        if score.verdict is None:
            continue
        if score.verdict is Verdict.NOT_VERIFIABLE:
            counts["not_verifiable"] += 1
        elif score.verdict is Verdict.DISAGREE:
            counts["false_dispute" if score.correct else "caught"] += 1
        else:
            counts["confirmed" if score.correct else "missed"] += 1
    return counts


# ---------- Report ----------


def render_report(evaluation: Evaluation, settings: Settings) -> str:
    e = evaluation
    per_doc_cost = e.cost_usd / e.documents if e.documents else Decimal(0)
    lines = [
        "# Evaluation report",
        "",
        f"- Documents: {e.documents} ({e.scored_documents} with an extraction scored field by "
        "field)",
        f"- Routing outcome as expected: {e.outcome_correct}/{e.documents}",
        f"- Escalation category as expected: {e.escalation_correct}/{e.escalations_expected}",
        f"- Auto-approved: {e.auto_approved}; auto-approved with a wrong field: "
        f"{len(e.auto_approved_with_errors)}"
        + (f" ({', '.join(e.auto_approved_with_errors)})" if e.auto_approved_with_errors else ""),
        f"- Field accuracy overall: {e.overall_accuracy:.1%}; on fields the document states: "
        f"{e.overall_accuracy_stated:.1%}; mean confidence "
        f"{e.overall_mean_confidence:.2f}; Brier {e.overall_brier:.3f}",
        f"- Cost: ${e.cost_usd:.4f} (${per_doc_cost:.4f} per document); tokens: "
        f"{e.input_tokens} input, {e.output_tokens} output",
        "",
        "## Routing: expected vs actual",
        "",
        "| Expected \\ Actual | auto_approve | human_review | escalated |",
        "|---|---|---|---|",
        *(
            f"| {expected} | "
            + " | ".join(
                str(e.outcome_matrix.get(expected, {}).get(a, 0))
                for a in ("auto_approve", "human_review", "escalated")
            )
            + " |"
            for expected in ("auto_approve", "human_review", "escalated")
        ),
        "",
        "## Field accuracy",
        "",
        '"Where stated" counts only documents whose gold says the field has a value; fields '
        "that are empty by construction are easy to get right and inflate the plain figure.",
        "",
        "| Field | Accuracy | Where stated | n stated |",
        "|---|---|---|---|",
        *(
            f"| {field} | {acc:.1%} | "
            f"{'-' if (hit := e.field_accuracy_stated[field]) is None else f'{hit:.1%}'} | "
            f"{e.field_count_stated[field]} |"
            for field, acc in e.field_accuracy.items()
        ),
        "",
        "## Derived fields (computed in Python from extracted vs gold values)",
        "",
        "| Derived field | Accuracy |",
        "|---|---|",
        *(f"| {name} | {acc:.1%} |" for name, acc in e.derived_accuracy.items()),
        "",
        "## Reviewer value",
        "",
        "| Extractor field was | Reviewer agreed | Reviewer disagreed |",
        "|---|---|---|",
        f"| correct | {e.reviewer['confirmed']} | {e.reviewer['false_dispute']} (false dispute) |",
        f"| wrong | {e.reviewer['missed']} (missed) | {e.reviewer['caught']} (caught) |",
        "",
        f"Not verifiable: {e.reviewer['not_verifiable']}.",
        "",
        f"## Calibration by policy type x field (flag: confidence exceeds accuracy by "
        f"{settings.calibration_gap:.2f} or more)",
        "",
        "| Policy type | Field | n | Mean confidence | Accuracy | Brier | Flag |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {c.policy_type} | {c.field} | {c.n} | {c.mean_confidence:.2f} | "
            f"{c.accuracy:.2f} | {c.brier:.3f} | {'FLAG' if c.flagged else ''} |"
            for c in e.calibration
            if c.flagged or c.accuracy < 1.0
        ),
        "",
        "Cells with every field correct are omitted from this table.",
        "",
        "## Per document",
        "",
        "| Document | Expected | Actual | Wrong fields |",
        "|---|---|---|---|",
        *(
            f"| {d['doc_id']} | {d['expected']} | {d['actual']} | "
            f"{', '.join(d['wrong_fields']) or '-'} |"
            for d in e.per_document
        ),
        "",
    ]
    return "\n".join(lines)
