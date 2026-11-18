from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, ValidationError

from policy_extractor.llm import (
    FALLBACK_BETA,
    STOP_UNPARSEABLE,
    AnthropicClient,
    ChatMessage,
    LLMCallError,
    LLMRequest,
    MissingRecordingError,
    RecordingClient,
    request_key,
)
from policy_extractor.models import PolicyExtraction, ReviewOutput
from tests.fakes import FakeClient, make_extraction, respond

FORBIDDEN_PARAMS = {"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools"}


def make_request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "role": "extractor",
        "model": "claude-sonnet-5-5",
        "effort": "medium",
        "system": "You extract.",
        "messages": (ChatMessage("user", "<document>x</document>"),),
        "max_tokens": 16000,
    }
    return LLMRequest(**(values | overrides))


def sdk_message(parsed: Any, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model="claude-sonnet-5-5",
        _request_id="req_123",
        usage=SimpleNamespace(
            input_tokens=900,
            output_tokens=150,
            cache_read_input_tokens=700,
            cache_creation_input_tokens=None,
        ),
    )


class FakeMessages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make_client(result: Any, *, refusal_fallback: bool) -> tuple[AnthropicClient, Any, Any]:
    stable, beta = FakeMessages(result), FakeMessages(result)
    sdk = SimpleNamespace(messages=stable, beta=SimpleNamespace(messages=beta))
    client = AnthropicClient(cast(anthropic.Anthropic, sdk), refusal_fallback=refusal_fallback)
    return client, stable, beta


def test_request_goes_through_parse_with_explicit_effort_and_cached_system_prompt():
    extraction = make_extraction()
    client, stable, beta = make_client(sdk_message(extraction), refusal_fallback=False)

    response = client.parse(make_request(), PolicyExtraction)

    assert beta.calls == []
    [call] = stable.calls
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["output_format"] is PolicyExtraction
    assert call["system"] == [
        {"type": "text", "text": "You extract.", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"] == [{"role": "user", "content": "<document>x</document>"}]
    assert not FORBIDDEN_PARAMS & call.keys()
    assert response.parsed == extraction
    assert (response.request_id, response.stop_reason) == ("req_123", "end_turn")
    assert response.usage.model_dump() == {
        "input_tokens": 900,
        "output_tokens": 150,
        "cache_read_input_tokens": 700,
        "cache_creation_input_tokens": 0,
    }


def test_refusal_fallback_flag_uses_the_beta_endpoint_with_default_fallbacks():
    client, stable, beta = make_client(sdk_message(make_extraction()), refusal_fallback=True)
    client.parse(make_request(), PolicyExtraction)

    assert stable.calls == []
    [call] = beta.calls
    assert (call["betas"], call["fallbacks"]) == ([FALLBACK_BETA], "default")
    assert call["output_config"] == {"effort": "medium"}
    assert not FORBIDDEN_PARAMS & call.keys()


def test_multi_turn_retry_messages_keep_their_roles():
    client, stable, _ = make_client(sdk_message(make_extraction()), refusal_fallback=False)
    messages = (
        ChatMessage("user", "doc"),
        ChatMessage("assistant", "{}"),
        ChatMessage("user", "findings"),
    )
    client.parse(make_request(messages=messages), PolicyExtraction)
    assert [m["role"] for m in stable.calls[0]["messages"]] == ["user", "assistant", "user"]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_stop_reason_is_checked_before_the_parsed_output_is_used(stop_reason):
    message = sdk_message(make_extraction(), stop_reason=stop_reason)
    client, _, _ = make_client(message, refusal_fallback=True)
    response = client.parse(make_request(), PolicyExtraction)
    assert (response.parsed, response.stop_reason, response.failed) == (None, stop_reason, True)


def test_output_the_sdk_cannot_parse_is_a_failed_response_not_a_crash():
    class Tiny(BaseModel):
        value: int

    with pytest.raises(ValidationError) as caught:
        Tiny.model_validate_json('{"value": ')
    client, _, _ = make_client(caught.value, refusal_fallback=False)
    response = client.parse(make_request(), PolicyExtraction)
    assert (response.parsed, response.stop_reason, response.failed) == (
        None,
        STOP_UNPARSEABLE,
        True,
    )


def api_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.invalid/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (api_error(anthropic.AuthenticationError, 401), "authentication"),
        (api_error(anthropic.PermissionDeniedError, 403), "permission_denied"),
        (api_error(anthropic.NotFoundError, 404), "not_found"),
        (api_error(anthropic.BadRequestError, 400), "bad_request"),
        (api_error(anthropic.RateLimitError, 429), "rate_limited"),
        (api_error(anthropic.InternalServerError, 500), "api_status_500"),
    ],
)
def test_typed_sdk_errors_become_llm_call_errors(error, kind):
    client, _, _ = make_client(error, refusal_fallback=False)
    with pytest.raises(LLMCallError) as caught:
        client.parse(make_request(), PolicyExtraction)
    assert caught.value.kind == kind


def test_request_key_changes_with_everything_that_shapes_the_answer():
    base = request_key(make_request(), PolicyExtraction)
    assert base == request_key(make_request(), PolicyExtraction)
    variants = [
        (make_request(model="claude-opus-5-5"), PolicyExtraction),
        (make_request(effort="high"), PolicyExtraction),
        (make_request(system="You extract carefully."), PolicyExtraction),
        (make_request(messages=(ChatMessage("user", "other"),)), PolicyExtraction),
        (make_request(), ReviewOutput),
    ]
    keys = {request_key(request, output) for request, output in variants}
    assert len(keys) == len(variants)
    assert base not in keys
    # Not part of what the model sees, so not part of the key.
    assert request_key(make_request(max_tokens=1), PolicyExtraction) == base


def test_recording_client_records_then_replays_without_a_client(tmp_path):
    extraction = make_extraction()
    inner = FakeClient([respond(extraction, model="claude-sonnet-5-5")])
    recorded = RecordingClient(tmp_path, "record", inner).parse(make_request(), PolicyExtraction)

    replayed = RecordingClient(tmp_path, "replay").parse(make_request(), PolicyExtraction)

    assert len(inner.requests) == 1
    assert replayed == recorded
    assert replayed.parsed == extraction
    assert [p.name for p in tmp_path.iterdir()] == [
        f"{request_key(make_request(), PolicyExtraction)}.json"
    ]


def test_recording_client_replays_failed_responses_too(tmp_path):
    inner = FakeClient([respond(None, stop_reason="refusal")])
    RecordingClient(tmp_path, "record", inner).parse(make_request(), PolicyExtraction)
    replayed = RecordingClient(tmp_path, "replay").parse(make_request(), PolicyExtraction)
    assert (replayed.parsed, replayed.stop_reason) == (None, "refusal")


def test_missing_recording_is_an_error_not_a_network_call(tmp_path):
    inner = FakeClient([respond(make_extraction())])
    RecordingClient(tmp_path, "record", inner).parse(make_request(), PolicyExtraction)
    with pytest.raises(MissingRecordingError):
        RecordingClient(tmp_path, "replay").parse(
            make_request(system="A changed prompt."), PolicyExtraction
        )


def test_record_mode_needs_an_inner_client(tmp_path):
    with pytest.raises(ValueError, match="inner client"):
        RecordingClient(tmp_path, "record")
