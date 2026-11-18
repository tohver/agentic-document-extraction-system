from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import pytest

from policy_extractor.batch import BatchItem, PrefilledClient, run_batch, run_pipeline_batched
from policy_extractor.extract import extraction_request
from policy_extractor.llm import STOP_UNPARSEABLE, LLMCallError, request_key
from policy_extractor.models import PolicyExtraction
from policy_extractor.route import Escalation, Outcome
from policy_extractor.runlog import NullRunLog
from tests.fakes import DOCUMENT, FakeClient, make_extraction, make_review, respond, settings

SETTINGS = settings()
LOG = NullRunLog()
USAGE = SimpleNamespace(
    input_tokens=800,
    output_tokens=120,
    cache_read_input_tokens=None,
    cache_creation_input_tokens=600,
)


def succeeded(custom_id: str, text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="claude-sonnet-5-5",
        id=f"msg_{custom_id}",
        usage=USAGE,
    )
    return SimpleNamespace(
        custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message)
    )


def not_succeeded(custom_id: str, kind: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=kind))


class FakeBatches:
    """Stands in for `client.messages.batches`: answers each batch from a scripted function."""

    def __init__(self, answer: Any, polls_until_ended: int = 2) -> None:
        self._answer = answer
        self._polls_until_ended = polls_until_ended
        self.created: list[list[Any]] = []
        self.retrieved = 0

    def create(self, requests: Any) -> SimpleNamespace:
        self.created.append(list(requests))
        return self._batch("in_progress")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.retrieved += 1
        done = self.retrieved % self._polls_until_ended == 0
        return self._batch("ended" if done else "in_progress")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        # Deliberately reversed: results arrive in any order.
        return [self._answer(request) for request in reversed(self.created[-1])]

    def _batch(self, status: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=f"batch_{len(self.created)}", processing_status=status, request_counts=None
        )


def make_sdk(batches: FakeBatches) -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, SimpleNamespace(messages=SimpleNamespace(batches=batches)))


def items(*doc_ids: str) -> list[BatchItem]:
    return [
        BatchItem(doc_id, extraction_request(f"{DOCUMENT}{doc_id}", SETTINGS), PolicyExtraction)
        for doc_id in doc_ids
    ]


def test_batch_requests_carry_effort_schema_and_cached_system_prompt():
    batches = FakeBatches(lambda r: succeeded(r["custom_id"], make_extraction().model_dump_json()))
    sleeps: list[float] = []
    run_batch(make_sdk(batches), items("A"), LOG, poll_interval_s=5, sleep=sleeps.append)

    [[request]] = batches.created
    params = request["params"]
    assert request["custom_id"] == "A"
    assert params["model"] == SETTINGS.extractor_model
    assert params["output_config"]["effort"] == SETTINGS.extractor_effort
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "record" in params["output_config"]["format"]["schema"]["properties"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert not {"temperature", "thinking", "tool_choice", "fallbacks", "betas"} & params.keys()
    assert sleeps == [5, 5]


def test_results_are_matched_by_custom_id_not_by_position():
    extractions = {
        doc_id: make_extraction(policy_number=f"{doc_id}-1", quotes={"policy_number": "TP-100"})
        for doc_id in ("A", "B", "C")
    }
    batches = FakeBatches(
        lambda r: succeeded(r["custom_id"], extractions[r["custom_id"]].model_dump_json())
    )
    batch_items = items("A", "B", "C")
    responses = run_batch(make_sdk(batches), batch_items, LOG, sleep=lambda _: None)

    for item in batch_items:
        response = responses[request_key(item.request, item.output_type)]
        assert response.parsed == extractions[item.custom_id]
        assert response.batch is True
        assert response.request_id == f"msg_{item.custom_id}"
        assert response.usage.cache_creation_input_tokens == 600


def test_failed_batch_entries_are_failed_responses_not_records():
    answers = {
        "A": succeeded("A", make_extraction().model_dump_json(), stop_reason="refusal"),
        "B": succeeded("B", '{"record": '),
        "C": not_succeeded("C", "errored"),
        "D": not_succeeded("D", "expired"),
    }
    batches = FakeBatches(lambda r: answers[r["custom_id"]])
    batch_items = items("A", "B", "C", "D")
    responses = run_batch(make_sdk(batches), batch_items, LOG, sleep=lambda _: None)

    by_id = {
        item.custom_id: responses[request_key(item.request, item.output_type)]
        for item in batch_items
    }
    assert {doc_id: r.stop_reason for doc_id, r in by_id.items()} == {
        "A": "refusal",
        "B": STOP_UNPARSEABLE,
        "C": "batch_errored",
        "D": "batch_expired",
    }
    assert all(r.parsed is None and r.failed for r in by_id.values())


def test_custom_ids_must_be_unique_and_valid():
    sdk = make_sdk(FakeBatches(lambda r: None))
    with pytest.raises(ValueError, match="custom_ids"):
        run_batch(sdk, items("A", "A"), LOG)
    with pytest.raises(ValueError, match="custom_ids"):
        run_batch(sdk, items("has space"), LOG)
    assert run_batch(sdk, [], LOG) == {}


def test_an_endpoint_without_the_batches_api_is_a_typed_error():
    class NoBatches:
        def create(self, requests: Any) -> Any:
            request = httpx2.Request("POST", "https://proxy.invalid/v1/messages/batches")
            raise anthropic.BadRequestError(
                "Invalid Service", response=httpx2.Response(400, request=request), body=None
            )

    sdk = cast(anthropic.Anthropic, SimpleNamespace(messages=SimpleNamespace(batches=NoBatches())))
    with pytest.raises(LLMCallError) as caught:
        run_batch(sdk, items("A"), LOG)
    assert caught.value.kind == "batch_api_status_400"


def test_prefilled_client_falls_back_to_the_live_client_for_unknown_requests():
    known, unknown = items("A", "B")
    prefilled = {request_key(known.request, PolicyExtraction): respond(make_extraction())}
    live = FakeClient([respond(make_extraction(), model="live")])
    client = PrefilledClient(prefilled, live)

    assert client.parse(known.request, PolicyExtraction).model == "fake-model"
    assert client.parse(unknown.request, PolicyExtraction).model == "live"
    assert live.requests == [unknown.request]


def test_batched_pipeline_reviews_only_what_survived_extraction_and_retries_live():
    documents = {"OK": DOCUMENT, "ESC": DOCUMENT + " no premium", "FIX": DOCUMENT + " bad date"}
    first_pass = {
        "extract-OK": make_extraction(),
        "extract-ESC": make_extraction(premium_amount=None),
        "extract-FIX": make_extraction(effective_date="01/01/2026"),
    }

    def answer(request: Any) -> SimpleNamespace:
        custom_id = request["custom_id"]
        if custom_id.startswith("extract-"):
            return succeeded(custom_id, first_pass[custom_id].model_dump_json())
        return succeeded(custom_id, make_review().model_dump_json())

    batches = FakeBatches(answer)
    live = FakeClient([respond(make_extraction(), model="claude-sonnet-5-5")])

    results = run_pipeline_batched(
        make_sdk(batches), live, documents, SETTINGS, LOG, sleep=lambda _: None
    )

    extraction_batch, review_batch = batches.created
    assert [r["custom_id"] for r in extraction_batch] == [
        "extract-OK",
        "extract-ESC",
        "extract-FIX",
    ]
    assert [r["custom_id"] for r in review_batch] == ["review-OK", "review-FIX"]
    assert review_batch[0]["params"]["model"] == SETTINGS.reviewer_model
    assert review_batch[0]["params"]["output_config"]["effort"] == SETTINGS.reviewer_effort
    # Only the retry of FIX went out as a live request.
    assert [(r.role, len(r.messages)) for r in live.requests] == [("extractor", 3)]

    by_id = {r.doc_id: r for r in results}
    assert {doc_id: r.decision.outcome for doc_id, r in by_id.items()} == {
        "OK": Outcome.AUTO_APPROVE,
        "ESC": Outcome.ESCALATED,
        "FIX": Outcome.AUTO_APPROVE,
    }
    assert by_id["ESC"].decision.escalation is Escalation.MISSING_SOURCE
    assert [c.batch for c in by_id["OK"].calls] == [True, True]
    assert [c.batch for c in by_id["FIX"].calls] == [True, False, True]
    assert [c.role for c in by_id["ESC"].calls] == ["extractor"]


def test_a_failed_batch_item_is_that_documents_batch_item_failed_escalation():
    def answer(request: Any) -> SimpleNamespace:
        if request["custom_id"] == "extract-BAD":
            return not_succeeded("extract-BAD", "expired")
        if request["custom_id"].startswith("extract-"):
            return succeeded(request["custom_id"], make_extraction().model_dump_json())
        return succeeded(request["custom_id"], make_review().model_dump_json())

    results = run_pipeline_batched(
        make_sdk(FakeBatches(answer)),
        FakeClient([]),
        {"GOOD": DOCUMENT, "BAD": DOCUMENT + " "},
        SETTINGS,
        LOG,
        sleep=lambda _: None,
    )
    by_id = {r.doc_id: r.decision for r in results}
    assert by_id["GOOD"].outcome is Outcome.AUTO_APPROVE
    assert by_id["BAD"].escalation is Escalation.BATCH_ITEM_FAILED
