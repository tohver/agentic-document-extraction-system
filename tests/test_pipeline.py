"""Whole-run behaviour (§3, §8): isolation of failures and the output files."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from policy_extractor.cli import finish_run, run_documents
from policy_extractor.llm import LLMCallError, LLMRequest, LLMResponse
from policy_extractor.models import PolicyType
from policy_extractor.route import Escalation, Outcome
from policy_extractor.runlog import JsonlRunLog, RunWriter, read_records
from tests.fakes import DOCUMENT, make_extraction, make_review, respond, settings

SETTINGS = settings(spot_check_share=0.0)


class ScriptedClient:
    """Answers by document: a document whose text contains a marker gets that behaviour."""

    def parse(self, request: LLMRequest, output_type: type[BaseModel]) -> LLMResponse[Any]:
        text = request.messages[0].content
        if "API-DOWN" in text:
            raise LLMCallError("rate_limited", "429 after SDK retries")
        if request.role == "reviewer":
            return respond(make_review())
        if "NO-PREMIUM" in text:
            return respond(make_extraction(premium_amount=None))
        if "REFUSE" in text:
            return respond(None, stop_reason="refusal")
        if "FINE-ART" in text:
            return respond(
                make_extraction(
                    policy_type=PolicyType.OTHER,
                    policy_type_detail="Personal Auto",
                    quotes={"policy_type_detail": "Personal Auto"},
                )
            )
        return respond(make_extraction())


DOCUMENTS = {
    "D1": DOCUMENT,
    "D2": DOCUMENT + "API-DOWN",
    "D3": DOCUMENT + "NO-PREMIUM",
    "D4": DOCUMENT + "REFUSE",
    "D5": DOCUMENT + "FINE-ART",
    "D6": DOCUMENT,
}


def run(tmp_path: Path) -> list[Any]:
    writer = RunWriter(tmp_path)
    log = JsonlRunLog(tmp_path / "events.jsonl")
    results = run_documents(ScriptedClient(), DOCUMENTS, SETTINGS, log, writer)
    return finish_run(results, SETTINGS, log, writer)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_one_failing_document_does_not_stop_the_run(tmp_path):
    results = run(tmp_path)
    outcomes = {r.doc_id: (r.decision.outcome, r.decision.escalation) for r in results}
    assert outcomes == {
        "D1": (Outcome.AUTO_APPROVE, None),
        "D2": (Outcome.ESCALATED, Escalation.API_ERROR),
        "D3": (Outcome.ESCALATED, Escalation.MISSING_SOURCE),
        "D4": (Outcome.ESCALATED, Escalation.MODEL_REFUSAL),
        "D5": (Outcome.HUMAN_REVIEW, None),
        "D6": (Outcome.AUTO_APPROVE, None),
    }


def test_every_input_appears_in_exactly_one_decision_file(tmp_path):
    run(tmp_path)
    decided = [line["doc_id"] for line in lines(tmp_path / "decisions.jsonl")]
    escalated = [line["doc_id"] for line in lines(tmp_path / "escalations.jsonl")]
    assert sorted(decided + escalated) == sorted(DOCUMENTS)
    assert not set(decided) & set(escalated)
    categories = {
        line["doc_id"]: line["category"] for line in lines(tmp_path / "escalations.jsonl")
    }
    assert categories == {"D2": "api_error", "D3": "missing_source", "D4": "model_refusal"}


def test_records_keep_the_extraction_and_derived_values_and_round_trip(tmp_path):
    results = run(tmp_path)
    records = read_records(tmp_path / "records.jsonl")
    assert [r.doc_id for r in records] == list(DOCUMENTS)
    assert records == results
    d1 = next(r for r in records if r.doc_id == "D1")
    assert d1.extraction is not None and d1.derived is not None
    assert (d1.derived.primary_limit, d1.derived.primary_deductible) == (300000, 500)
    d3 = next(r for r in records if r.doc_id == "D3")
    assert d3.extraction is not None  # escalated documents keep what was extracted


def test_events_and_summaries_are_written_without_document_text(tmp_path):
    run(tmp_path)
    events = lines(tmp_path / "events.jsonl")
    kinds = {e["event"] for e in events}
    assert {"document_started", "llm_call", "validation", "decision", "run_summary"} <= kinds
    [api_down] = [
        e for e in events if e["event"] == "extraction_finished" and e["escalation"] == "api_error"
    ]
    assert api_down["failure"] == "api_error:rate_limited"
    assert all("Jane Q. Tester" not in json.dumps(e) for e in events)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["documents"] == 6
    assert summary["escalations"]["missing_source"] == 1
    assert summary["finding_codes"]["missing_source:premium_amount_absent"] == 1
    assert "| D5 | other | human_review |" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_spot_checks_are_drawn_per_policy_type_from_auto_approved_records(tmp_path):
    writer = RunWriter(tmp_path)
    log = JsonlRunLog(tmp_path / "events.jsonl")
    sampling = settings(spot_check_share=0.1)
    results = run_documents(ScriptedClient(), DOCUMENTS, sampling, log, writer)
    final = finish_run(results, sampling, log, writer)
    spot = [r.doc_id for r in final if r.decision.outcome is Outcome.SPOT_CHECK]
    assert len(spot) == 1 and spot[0] in {"D1", "D6"}
    assert {line["outcome"] for line in lines(tmp_path / "decisions.jsonl")} >= {"spot_check"}
