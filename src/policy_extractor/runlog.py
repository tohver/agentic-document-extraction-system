"""Machine-readable run trace (CLAUDE.md §8). Document text is never written here.

- `events.jsonl`: one event per step and per API call, appended as it happens.
- `records.jsonl`: the full result of every document (extraction, derived fields, findings,
  review, calls), appended as each document finishes.
- `decisions.jsonl` / `escalations.jsonl`: every document lands in exactly one of them. They are
  appended as documents finish and rewritten once at the end, after spot-check sampling.
- `summary.json` and `summary.md`: written at the end.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic_core import to_jsonable_python

if TYPE_CHECKING:
    from policy_extractor.pipeline import DocumentResult


class RunLog(Protocol):
    def event(self, kind: str, **fields: object) -> None: ...


class NullRunLog:
    def event(self, kind: str, **fields: object) -> None:
        return None


class JsonlRunLog:
    """Appends events to a JSONL file, one line per event, flushed immediately."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def event(self, kind: str, **fields: object) -> None:
        line = {"ts": datetime.now(UTC).isoformat(), "event": kind, **to_jsonable_python(fields)}
        _append(self._path, line)


def _append(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable_python(payload), ensure_ascii=False) + "\n")


def decision_line(result: DocumentResult) -> dict[str, object]:
    return {
        "doc_id": result.doc_id,
        "policy_type": result.policy_type,
        "outcome": result.decision.outcome,
        "reasons": result.decision.reasons,
        "reviewer_reasons": result.decision.reviewer_reasons,
    }


def escalation_line(result: DocumentResult) -> dict[str, object]:
    return {
        "doc_id": result.doc_id,
        "category": result.decision.escalation,
        "reasons": result.decision.reasons,
        "finding_history": result.finding_history,
    }


class RunWriter:
    """Writes the per-document output files of one run directory."""

    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        for name in ("records.jsonl", "decisions.jsonl", "escalations.jsonl", "events.jsonl"):
            (out / name).write_text("", encoding="utf-8")

    def document(self, result: DocumentResult) -> None:
        _append(self.out / "records.jsonl", result.model_dump(mode="json"))
        if result.decision.escalation is None:
            _append(self.out / "decisions.jsonl", decision_line(result))
        else:
            _append(self.out / "escalations.jsonl", escalation_line(result))

    def finish(self, results: Sequence[DocumentResult], summary: object, text: str) -> None:
        """Rewrite the decision files with final outcomes (after sampling) and the summaries."""
        decisions = [r for r in results if r.decision.escalation is None]
        escalations = [r for r in results if r.decision.escalation is not None]
        _rewrite(self.out / "decisions.jsonl", [decision_line(r) for r in decisions])
        _rewrite(self.out / "escalations.jsonl", [escalation_line(r) for r in escalations])
        _rewrite(self.out / "records.jsonl", [r.model_dump(mode="json") for r in results])
        (self.out / "summary.json").write_text(
            json.dumps(to_jsonable_python(summary), indent=2) + "\n", encoding="utf-8"
        )
        (self.out / "summary.md").write_text(text, encoding="utf-8")


def _rewrite(path: Path, lines: Sequence[object]) -> None:
    path.write_text(
        "".join(json.dumps(to_jsonable_python(line), ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )


def read_records(path: Path) -> list[DocumentResult]:
    from policy_extractor.pipeline import DocumentResult

    lines = path.read_text(encoding="utf-8").splitlines()
    return [DocumentResult.model_validate_json(line) for line in lines if line.strip()]
