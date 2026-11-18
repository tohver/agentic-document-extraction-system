"""What an unsure or unavailable reviewer does to routing: never an automatic approval."""

from typing import Any

from pydantic import BaseModel

from policy_extractor.llm import LLMCallError, LLMRequest, LLMResponse
from policy_extractor.models import FieldVerdict, Verdict
from policy_extractor.pipeline import process_document
from policy_extractor.route import Outcome, route
from policy_extractor.runlog import NullRunLog
from tests.fakes import DOCUMENT, make_extraction, make_review, respond, settings

SETTINGS = settings()


def test_a_field_the_reviewer_could_not_verify_is_not_approved():
    reason = "The premium line is cut off in the scan."
    review = make_review()
    review.verdicts = [
        FieldVerdict(field=v.field, verdict=Verdict.NOT_VERIFIABLE, reason=reason)
        if v.field.value == "premium_amount"
        else v
        for v in review.verdicts
    ]
    decision = route(make_extraction(), [], review, SETTINGS)
    assert (decision.outcome, decision.reasons) == (
        Outcome.HUMAN_REVIEW,
        ["reviewer_not_verifiable:premium_amount"],
    )
    assert decision.reviewer_reasons == {"premium_amount": reason}


class ReviewerDown:
    def parse(self, request: LLMRequest, output_type: type[BaseModel]) -> LLMResponse[Any]:
        if request.role == "reviewer":
            raise LLMCallError("rate_limited", "429 after SDK retries")
        return respond(make_extraction())


def test_a_failed_review_call_keeps_the_extraction_and_sends_it_to_a_person():
    result = process_document(ReviewerDown(), "D1", DOCUMENT, SETTINGS, NullRunLog())
    assert (result.decision.outcome, result.decision.escalation) == (Outcome.HUMAN_REVIEW, None)
    assert result.decision.reasons == ["reviewer_unavailable", "reviewer_api_error:rate_limited"]
    assert result.extraction == make_extraction()
    assert result.derived is not None
    assert [call.role for call in result.calls] == ["extractor"]
