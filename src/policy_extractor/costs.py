"""Per-call accounting: what each request used and what it cost."""

from decimal import Decimal
from typing import Any, Final

from pydantic import BaseModel

from policy_extractor.config import ModelPrice, Settings
from policy_extractor.llm import LLMRequest, LLMResponse, Role, Usage

CACHE_WRITE_MULTIPLIER: Final = Decimal("1.25")
BATCH_DISCOUNT: Final = Decimal("0.5")
PER_MILLION: Final = Decimal(1_000_000)


class CallRecord(BaseModel):
    role: Role
    model: str
    request_id: str | None
    stop_reason: str | None
    usage: Usage
    latency_s: float
    batch: bool
    # None when the serving model has no configured price; never silently counted as free.
    cost_usd: Decimal | None


def call_cost(price: ModelPrice, usage: Usage, *, batch: bool) -> Decimal:
    cost = (
        usage.input_tokens * price.input
        + usage.cache_creation_input_tokens * price.input * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * price.cache_read
        + usage.output_tokens * price.output
    ) / PER_MILLION
    return cost * BATCH_DISCOUNT if batch else cost


def record_call(request: LLMRequest, response: LLMResponse[Any], settings: Settings) -> CallRecord:
    # A refusal fallback can be served by a different model than the one requested.
    price = settings.prices.get(response.model) or settings.prices.get(request.model)
    return CallRecord(
        role=request.role,
        model=response.model,
        request_id=response.request_id,
        stop_reason=response.stop_reason,
        usage=response.usage,
        latency_s=response.latency_s,
        batch=response.batch,
        cost_usd=None if price is None else call_cost(price, response.usage, batch=response.batch),
    )
