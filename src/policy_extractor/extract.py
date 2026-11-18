"""Build extraction requests. One request is one model call; the loop lives in `retry.py`."""

from collections.abc import Sequence

from policy_extractor.config import Settings
from policy_extractor.llm import ChatMessage, LLMClient, LLMRequest, LLMResponse
from policy_extractor.models import ExtractionPatch, PolicyExtraction
from policy_extractor.prompts import DOCUMENT_TEMPLATE, EXTRACTOR_SYSTEM, RETRY_TEMPLATE
from policy_extractor.validate import Finding


def extraction_request(document_text: str, settings: Settings) -> LLMRequest:
    return LLMRequest(
        role="extractor",
        model=settings.extractor_model,
        effort=settings.extractor_effort,
        system=EXTRACTOR_SYSTEM,
        messages=(ChatMessage("user", DOCUMENT_TEMPLATE.format(document=document_text)),),
        max_tokens=settings.max_tokens,
    )


def retry_request(
    document_text: str,
    previous: PolicyExtraction,
    findings: Sequence[Finding],
    settings: Settings,
) -> LLMRequest:
    """The original request plus the previous output (as JSON) and its findings, nothing else.
    The answer is an `ExtractionPatch`, not a whole new record.

    Appending turns instead of rewriting the first message keeps the cached prefix intact.
    """
    first = extraction_request(document_text, settings)
    return LLMRequest(
        role=first.role,
        model=first.model,
        effort=first.effort,
        system=first.system,
        messages=(
            *first.messages,
            ChatMessage("assistant", previous.model_dump_json()),
            ChatMessage("user", RETRY_TEMPLATE.format(findings=format_findings(findings))),
        ),
        max_tokens=first.max_tokens,
    )


def format_findings(findings: Sequence[Finding]) -> str:
    return "\n".join(
        f"- [{f.category.value}/{f.code}] {f.field}: {f.message}"
        + (f" Observed: {f.observed}" if f.observed is not None else "")
        for f in findings
    )


def extract(client: LLMClient, request: LLMRequest) -> LLMResponse[PolicyExtraction]:
    return client.parse(request, PolicyExtraction)


def extract_patch(client: LLMClient, request: LLMRequest) -> LLMResponse[ExtractionPatch]:
    """A retry: the answer is a patch over the previous output, see `models.apply_patch`."""
    return client.parse(request, ExtractionPatch)
