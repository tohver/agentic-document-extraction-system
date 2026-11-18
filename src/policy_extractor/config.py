"""Settings: every tunable the pipeline reads lives here, so runs are reproducible from config."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ModelPrice(BaseModel):
    """USD per million tokens. Cache writes cost 1.25x input; batch runs cost half."""

    input: Decimal
    output: Decimal
    cache_read: Decimal


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-sonnet-5-5": ModelPrice(
        input=Decimal("2.00"), output=Decimal("10.00"), cache_read=Decimal("0.10")
    ),
    "claude-opus-5-5": ModelPrice(
        input=Decimal("4.00"), output=Decimal("20.00"), cache_read=Decimal("0.20")
    ),
}


class Settings(BaseSettings):
    """Pipeline configuration. Override any field with `POLICY_<FIELD>` or in `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="POLICY_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Credentials and endpoint: never hard-coded, never logged.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, validation_alias="ANTHROPIC_BASE_URL")
    # SDK 2.x sends the key as `Authorization: Bearer`; some proxies still read `X-Api-Key`.
    send_x_api_key_header: bool = False

    # The reviewer is a different, stronger model so the review is independent.
    extractor_model: str = "claude-sonnet-5-5"
    extractor_effort: Effort = "medium"
    reviewer_model: str = "claude-opus-5-5"
    reviewer_effort: Effort = "high"
    max_tokens: int = 16000
    sdk_max_retries: int = 2
    refusal_fallback: bool = True

    # Validation and retry.
    max_retries: int = 2
    money_tolerance: Decimal = Decimal("0.01")

    # Routing and sampling.
    confidence_threshold: float = 0.90
    spot_check_share: float = 0.10
    sampling_seed: int = 0

    # Evaluation.
    calibration_gap: float = 0.15

    prices: dict[str, ModelPrice] = DEFAULT_PRICES
