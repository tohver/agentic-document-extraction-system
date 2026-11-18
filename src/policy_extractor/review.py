"""Independent review (CLAUDE.md §7).

The reviewer is a different, stronger model. It sees the document, the same field definitions
the extractor used (in its system prompt) and the proposed values, and nothing else: no extractor
prompt, reasoning, confidence, quotes or retry history, so it cannot anchor on the extractor's view.
"""

from policy_extractor.config import Settings
from policy_extractor.llm import ChatMessage, LLMClient, LLMRequest, LLMResponse
from policy_extractor.models import PolicyExtraction, ReviewOutput
from policy_extractor.prompts import REVIEW_TEMPLATE, REVIEWER_SYSTEM


def review_request(
    document_text: str, extraction: PolicyExtraction, settings: Settings
) -> LLMRequest:
    content = REVIEW_TEMPLATE.format(
        document=document_text, extraction=extraction.values().model_dump_json(indent=2)
    )
    return LLMRequest(
        role="reviewer",
        model=settings.reviewer_model,
        effort=settings.reviewer_effort,
        system=REVIEWER_SYSTEM,
        messages=(ChatMessage("user", content),),
        max_tokens=settings.max_tokens,
    )


def review(client: LLMClient, request: LLMRequest) -> LLMResponse[ReviewOutput]:
    return client.parse(request, ReviewOutput)
