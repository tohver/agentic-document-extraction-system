"""Message Batches path for non-urgent bulk runs (half price, asynchronous).

A run is two batches: every first extraction, then every review (CLAUDE.md §10). Batch results
are fed to the ordinary pipeline through `PrefilledClient`, so validation, retry and routing are
exactly the code the synchronous path uses; the rare retry goes out as a normal request. Results
arrive in any order, so everything is keyed by `custom_id`. Server-side refusal fallback is not
available on the Batches API and is not sent.
"""

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from policy_extractor.config import Settings
from policy_extractor.extract import extraction_request
from policy_extractor.llm import (
    FAILED_STOP_REASONS,
    STOP_UNPARSEABLE,
    LLMCallError,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Usage,
    log_call,
    message_params,
    request_key,
    system_blocks,
    usage_from_sdk,
)
from policy_extractor.models import PolicyExtraction, ReviewOutput
from policy_extractor.pipeline import (
    DocumentResult,
    api_error_result,
    extraction_phase,
    review_phase,
)
from policy_extractor.retry import ExtractionOutcome
from policy_extractor.review import review_request
from policy_extractor.runlog import RunLog

CUSTOM_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class BatchItem:
    custom_id: str
    request: LLMRequest
    output_type: type[BaseModel]


class PrefilledClient:
    """Serves responses fetched by a batch; anything else goes to the inner (live) client."""

    def __init__(self, prefilled: Mapping[str, LLMResponse[Any]], inner: LLMClient) -> None:
        self._prefilled = prefilled
        self._inner = inner

    def parse[T: BaseModel](self, request: LLMRequest, output_type: type[T]) -> LLMResponse[T]:
        response = self._prefilled.get(request_key(request, output_type))
        if response is None:
            return self._inner.parse(request, output_type)
        return response


def run_batch(
    sdk: anthropic.Anthropic,
    items: Sequence[BatchItem],
    log: RunLog,
    *,
    poll_interval_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, LLMResponse[Any]]:
    """Submit one batch, wait for it, and return responses keyed by request hash."""
    if not items:
        return {}
    by_id = {item.custom_id: item for item in items}
    if len(by_id) != len(items) or not all(CUSTOM_ID.fullmatch(key) for key in by_id):
        raise ValueError("custom_ids must be unique and match [A-Za-z0-9_-]{1,64}")
    started = time.perf_counter()
    try:
        return _run(sdk, items, by_id, log, started, poll_interval_s, sleep)
    except anthropic.APIStatusError as exc:
        # For example a proxy that does not offer the Batches API at all.
        raise LLMCallError(f"batch_api_status_{exc.status_code}", exc.message) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMCallError("batch_connection", str(exc)) from exc


def _run(
    sdk: anthropic.Anthropic,
    items: Sequence[BatchItem],
    by_id: Mapping[str, BatchItem],
    log: RunLog,
    started: float,
    poll_interval_s: float,
    sleep: Callable[[float], None],
) -> dict[str, LLMResponse[Any]]:
    batch = sdk.messages.batches.create(
        requests=[
            Request(
                custom_id=item.custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=item.request.model,
                    max_tokens=item.request.max_tokens,
                    system=system_blocks(item.request),
                    messages=message_params(item.request),
                    output_config={
                        "effort": item.request.effort,
                        "format": {
                            "type": "json_schema",
                            "schema": anthropic.transform_schema(item.output_type),
                        },
                    },
                ),
            )
            for item in items
        ]
    )
    log.event("batch_submitted", batch_id=batch.id, requests=len(items))
    while batch.processing_status != "ended":
        sleep(poll_interval_s)
        batch = sdk.messages.batches.retrieve(batch.id)
    elapsed = time.perf_counter() - started
    log.event("batch_ended", batch_id=batch.id, seconds=elapsed, counts=batch.request_counts)

    responses: dict[str, LLMResponse[Any]] = {}
    for entry in sdk.messages.batches.results(batch.id):
        item = by_id[entry.custom_id]
        result = entry.result
        if result.type == "succeeded":
            message = result.message
            text = next((block.text for block in message.content if block.type == "text"), None)
            response = _parse_message(
                item.output_type,
                text,
                message.stop_reason,
                message.model,
                message.id,
                usage_from_sdk(message.usage),
                elapsed,
            )
        else:
            response = LLMResponse(
                parsed=None,
                stop_reason=f"batch_{result.type}",
                model=item.request.model,
                request_id=None,
                usage=Usage(),
                latency_s=elapsed,
                batch=True,
            )
        log_call(item.request, response)
        responses[request_key(item.request, item.output_type)] = response
    return responses


def _parse_message(
    output_type: type[BaseModel],
    text: str | None,
    stop_reason: str | None,
    model: str,
    message_id: str,
    usage: Usage,
    latency_s: float,
) -> LLMResponse[Any]:
    """Check `stop_reason` first; only a complete, unrefused answer is parsed."""
    parsed: BaseModel | None = None
    if stop_reason not in FAILED_STOP_REASONS and text is not None:
        try:
            parsed = output_type.model_validate_json(text)
        except ValidationError:
            stop_reason = STOP_UNPARSEABLE
    return LLMResponse(
        parsed=parsed,
        stop_reason=stop_reason,
        model=model,
        request_id=message_id,
        usage=usage,
        latency_s=latency_s,
        batch=True,
    )


def run_pipeline_batched(
    sdk: anthropic.Anthropic,
    live: LLMClient,
    documents: Mapping[str, str],
    settings: Settings,
    log: RunLog,
    *,
    poll_interval_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DocumentResult]:
    """Run the whole pipeline with first extractions and reviews going through two batches.

    A batch item that errors or expires becomes that document's `batch_item_failed` escalation;
    retries and anything not prefilled go out as ordinary requests through `live`.
    """
    extraction_items = [
        BatchItem(f"extract-{doc_id}", extraction_request(text, settings), PolicyExtraction)
        for doc_id, text in documents.items()
    ]
    extractions = PrefilledClient(
        run_batch(sdk, extraction_items, log, poll_interval_s=poll_interval_s, sleep=sleep), live
    )
    outcomes: dict[str, ExtractionOutcome] = {}
    failed: dict[str, DocumentResult] = {}
    for doc_id, text in documents.items():
        try:
            outcomes[doc_id] = extraction_phase(extractions, doc_id, text, settings, log)
        except LLMCallError as error:
            failed[doc_id] = api_error_result(doc_id, error)
    review_items = [
        BatchItem(
            f"review-{doc_id}",
            review_request(documents[doc_id], outcome.extraction, settings),
            ReviewOutput,
        )
        for doc_id, outcome in outcomes.items()
        if outcome.status == "ok" and outcome.extraction is not None
    ]
    reviews = PrefilledClient(
        run_batch(sdk, review_items, log, poll_interval_s=poll_interval_s, sleep=sleep), live
    )
    results: list[DocumentResult] = []
    for doc_id, text in documents.items():
        if doc_id in failed:
            results.append(failed[doc_id])
            continue
        try:
            results.append(review_phase(reviews, doc_id, text, outcomes[doc_id], settings, log))
        except LLMCallError as error:
            results.append(api_error_result(doc_id, error))
    return results
