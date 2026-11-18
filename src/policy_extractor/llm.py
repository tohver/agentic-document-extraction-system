"""The only module that talks to Claude for single requests.

Business logic depends on the narrow `LLMClient` protocol, never on the SDK, so every branch can
be tested offline. `RecordingClient` keys recordings on a hash of the request, which means editing
a prompt invalidates its recording on purpose: a stale fixture can never pass silently.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol, Self

import anthropic
from anthropic.types import MessageParam, TextBlockParam
from anthropic.types.beta import BetaMessageParam
from pydantic import BaseModel, ValidationError

from policy_extractor.config import Effort, Settings

logger = logging.getLogger(__name__)

Role = Literal["extractor", "reviewer"]
RecordingMode = Literal["replay", "record"]

FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
# `messages.parse` validates the text before we can look at `stop_reason`, so truncated or refused
# output surfaces as a ValidationError. We report it under this synthetic stop reason.
STOP_UNPARSEABLE: Final = "unparseable_output"
FAILED_STOP_REASONS: Final = frozenset({"refusal", "max_tokens", STOP_UNPARSEABLE})


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    role: Role
    model: str
    effort: Effort
    system: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse[T]:
    """`parsed` is None whenever `stop_reason` is in FAILED_STOP_REASONS."""

    parsed: T | None
    stop_reason: str | None
    model: str
    request_id: str | None
    usage: Usage
    latency_s: float
    batch: bool = False

    @property
    def failed(self) -> bool:
        return self.parsed is None or self.stop_reason in FAILED_STOP_REASONS


class LLMCallError(Exception):
    """An infrastructure failure (auth, bad request, rate limit, network), not an outcome."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class MissingRecordingError(Exception):
    """Replay mode was asked for a request that was never recorded."""


class LLMClient(Protocol):
    def parse[T: BaseModel](self, request: LLMRequest, output_type: type[T]) -> LLMResponse[T]: ...


def system_blocks(request: LLMRequest) -> list[TextBlockParam]:
    """The static system prompt with a cache breakpoint, before any per-document content."""
    return [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]


def message_params(request: LLMRequest) -> list[MessageParam]:
    params: list[MessageParam] = []
    for message in request.messages:
        if message.role == "user":
            params.append({"role": "user", "content": message.content})
        else:
            params.append({"role": "assistant", "content": message.content})
    return params


def _beta_message_params(request: LLMRequest) -> list[BetaMessageParam]:
    params: list[BetaMessageParam] = []
    for message in request.messages:
        if message.role == "user":
            params.append({"role": "user", "content": message.content})
        else:
            params.append({"role": "assistant", "content": message.content})
    return params


def request_key(request: LLMRequest, output_type: type[BaseModel]) -> str:
    """Hash of everything that determines the model's answer."""
    payload = {
        "model": request.model,
        "effort": request.effort,
        "system": request.system,
        "messages": [[m.role, m.content] for m in request.messages],
        "schema": output_type.model_json_schema(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def log_call(request: LLMRequest, response: LLMResponse[Any]) -> None:
    logger.info(
        "llm_call role=%s model=%s request_id=%s stop_reason=%s input=%d output=%d "
        "cache_read=%d cache_write=%d latency_s=%.2f batch=%s",
        request.role,
        response.model,
        response.request_id,
        response.stop_reason,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.cache_read_input_tokens,
        response.usage.cache_creation_input_tokens,
        response.latency_s,
        response.batch,
    )


class AnthropicClient:
    """Adapter over the official SDK. The SDK handles 429/5xx retries itself."""

    def __init__(self, sdk: anthropic.Anthropic, *, refusal_fallback: bool) -> None:
        self._sdk = sdk
        self._refusal_fallback = refusal_fallback

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        headers = {"X-Api-Key": key} if settings.send_x_api_key_header and key else None
        sdk = anthropic.Anthropic(
            api_key=key,
            base_url=settings.anthropic_base_url,
            max_retries=settings.sdk_max_retries,
            default_headers=headers,
        )
        return cls(sdk, refusal_fallback=settings.refusal_fallback)

    @property
    def sdk(self) -> anthropic.Anthropic:
        return self._sdk

    def parse[T: BaseModel](self, request: LLMRequest, output_type: type[T]) -> LLMResponse[T]:
        started = time.perf_counter()
        try:
            response = self._call(request, output_type)
        except ValidationError:
            response = LLMResponse(
                parsed=None,
                stop_reason=STOP_UNPARSEABLE,
                model=request.model,
                request_id=None,
                usage=Usage(),
                latency_s=0.0,
            )
        except anthropic.AuthenticationError as exc:
            raise LLMCallError("authentication", exc.message) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMCallError("permission_denied", exc.message) from exc
        except anthropic.NotFoundError as exc:
            raise LLMCallError("not_found", exc.message) from exc
        except anthropic.BadRequestError as exc:
            raise LLMCallError("bad_request", exc.message) from exc
        except anthropic.RateLimitError as exc:
            raise LLMCallError("rate_limited", exc.message) from exc
        except anthropic.APIStatusError as exc:
            raise LLMCallError(f"api_status_{exc.status_code}", exc.message) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMCallError("connection", str(exc)) from exc
        response = LLMResponse(
            parsed=response.parsed,
            stop_reason=response.stop_reason,
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
            latency_s=time.perf_counter() - started,
        )
        log_call(request, response)
        return response

    def _call[T: BaseModel](self, request: LLMRequest, output_type: type[T]) -> LLMResponse[T]:
        if self._refusal_fallback:
            beta_message = self._sdk.beta.messages.parse(
                model=request.model,
                max_tokens=request.max_tokens,
                system=[
                    {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
                ],
                messages=_beta_message_params(request),
                output_config={"effort": request.effort},
                output_format=output_type,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            return _to_response(
                beta_message.parsed_output,
                beta_message.stop_reason,
                beta_message.model,
                beta_message._request_id,
                beta_message.usage,
            )
        message = self._sdk.messages.parse(
            model=request.model,
            max_tokens=request.max_tokens,
            system=system_blocks(request),
            messages=message_params(request),
            output_config={"effort": request.effort},
            output_format=output_type,
        )
        return _to_response(
            message.parsed_output,
            message.stop_reason,
            message.model,
            message._request_id,
            message.usage,
        )


class _SdkUsage(Protocol):
    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cache_read_input_tokens(self) -> int | None: ...
    @property
    def cache_creation_input_tokens(self) -> int | None: ...


def usage_from_sdk(usage: _SdkUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )


def _to_response[T](
    parsed: T | None,
    stop_reason: str | None,
    model: str,
    request_id: str | None,
    usage: _SdkUsage,
) -> LLMResponse[T]:
    """Check `stop_reason` before trusting the parse: refusals and truncation are not records."""
    usable = stop_reason not in FAILED_STOP_REASONS
    return LLMResponse(
        parsed=parsed if usable else None,
        stop_reason=stop_reason,
        model=model,
        request_id=request_id,
        usage=usage_from_sdk(usage),
        latency_s=0.0,
    )


class Recording(BaseModel):
    """One recorded response, stored as `<request hash>.json`."""

    role: Role
    request_model: str
    effort: Effort
    parsed: dict[str, Any] | None
    stop_reason: str | None
    model: str
    request_id: str | None
    usage: Usage
    latency_s: float
    batch: bool = False


class RecordingClient:
    """Record real responses once, then replay them offline.

    In replay mode a missing recording raises: tests must never fall through to the network.
    """

    def __init__(
        self, directory: Path, mode: RecordingMode, inner: LLMClient | None = None
    ) -> None:
        if mode == "record" and inner is None:
            raise ValueError("record mode needs an inner client to call")
        self._directory = directory
        self._mode = mode
        self._inner = inner

    def parse[T: BaseModel](self, request: LLMRequest, output_type: type[T]) -> LLMResponse[T]:
        key = request_key(request, output_type)
        path = self._directory / f"{key}.json"
        if self._mode == "replay":
            if not path.exists():
                raise MissingRecordingError(
                    f"no recording for {request.role} request {key[:12]} in {self._directory}; "
                    "re-record with --mode record"
                )
            recording = Recording.model_validate_json(path.read_text(encoding="utf-8"))
            return LLMResponse(
                parsed=None
                if recording.parsed is None
                else output_type.model_validate(recording.parsed),
                stop_reason=recording.stop_reason,
                model=recording.model,
                request_id=recording.request_id,
                usage=recording.usage,
                latency_s=recording.latency_s,
                batch=recording.batch,
            )
        assert self._inner is not None
        response = self._inner.parse(request, output_type)
        recording = Recording(
            role=request.role,
            request_model=request.model,
            effort=request.effort,
            parsed=None if response.parsed is None else response.parsed.model_dump(mode="json"),
            stop_reason=response.stop_reason,
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
            latency_s=response.latency_s,
            batch=response.batch,
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        path.write_text(recording.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return response
