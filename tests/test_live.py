"""Live smoke tests: the real API, through the configured endpoint (for example a proxy).

Skipped without a key, and excluded from the default run (`-m 'not live'`). Run with:

    uv run pytest -m live
"""

import pytest

from policy_extractor.config import Settings
from policy_extractor.extract import extract, extraction_request
from policy_extractor.llm import AnthropicClient
from policy_extractor.review import review, review_request
from tests.fakes import DOCUMENT

pytestmark = pytest.mark.live

SETTINGS = Settings()
needs_key = pytest.mark.skipif(
    SETTINGS.anthropic_api_key is None, reason="ANTHROPIC_API_KEY is not set"
)


@needs_key
def test_extractor_model_returns_the_policy_schema():
    client = AnthropicClient.from_settings(SETTINGS)
    response = extract(client, extraction_request(DOCUMENT, SETTINGS))
    assert not response.failed, response.stop_reason
    assert response.parsed is not None
    assert response.parsed.record.policy_number == "TP-100"
    assert response.usage.input_tokens > 0


@needs_key
def test_reviewer_model_returns_a_verdict_per_field():
    client = AnthropicClient.from_settings(SETTINGS)
    extracted = extract(client, extraction_request(DOCUMENT, SETTINGS)).parsed
    assert extracted is not None
    response = review(client, review_request(DOCUMENT, extracted, SETTINGS))
    assert not response.failed, response.stop_reason
    assert response.parsed is not None
    assert len(response.parsed.verdicts) == len(extracted.evidence)
