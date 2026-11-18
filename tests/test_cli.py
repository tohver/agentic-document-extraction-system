"""The CLI end to end on recorded responses: the whole dataset, offline, no key."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from policy_extractor.cli import app

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ROOT / "data" / "policies"
GOLD = ROOT / "data" / "gold"
FIXTURES = ROOT / "fixtures" / "recorded"
runner = CliRunner()


@pytest.fixture(scope="module")
def replayed_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("run")
    result = runner.invoke(
        app,
        ["run", str(POLICIES), "--out", str(out), "--mode", "replay", "--fixtures", str(FIXTURES)],
    )
    assert result.exit_code == 0, result.output
    return out


def test_run_replays_the_whole_dataset_offline(replayed_run: Path):
    for name in (
        "records.jsonl",
        "decisions.jsonl",
        "escalations.jsonl",
        "events.jsonl",
        "summary.json",
        "summary.md",
    ):
        assert (replayed_run / name).exists(), name
    summary = json.loads((replayed_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["documents"] == len(list(POLICIES.glob("*.txt")))


def test_eval_scores_the_replayed_run_against_gold(replayed_run: Path):
    result = runner.invoke(app, ["eval", str(replayed_run), "--gold", str(GOLD)])
    assert result.exit_code == 0, result.output
    report = (replayed_run / "eval_report.md").read_text(encoding="utf-8")
    assert "# Evaluation report" in report
    # The property the pipeline exists for: nothing wrong was approved without a person.
    assert "auto-approved with a wrong field: 0" in report
    assert "Escalation category as expected: 5/5" in report


def test_extract_prints_the_outcome_of_one_document():
    result = runner.invoke(
        app,
        [
            "extract",
            str(POLICIES / "POL-2025-015.txt"),
            "--mode",
            "replay",
            "--fixtures",
            str(FIXTURES),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "POL-2025-015: escalated (missing_source) after 1 call(s)" in result.output
    assert "premium_amount_absent" in result.output


def test_replay_without_a_recording_reports_it_instead_of_calling_the_network(tmp_path: Path):
    document = tmp_path / "NEW-001.txt"
    document.write_text("A document nobody recorded.", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "run",
            str(document),
            "--out",
            str(tmp_path / "out"),
            "--mode",
            "replay",
            "--fixtures",
            str(tmp_path / "empty"),
        ],
    )
    assert result.exit_code != 0
    assert "no recording" in str(result.exception)
