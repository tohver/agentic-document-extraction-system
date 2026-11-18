"""Command line: `run`, `extract`, `eval`, `batch`. All side effects of a run start here."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from policy_extractor.batch import run_pipeline_batched
from policy_extractor.config import Settings
from policy_extractor.dataset import load_documents, load_gold
from policy_extractor.evaluate import evaluate, render_report
from policy_extractor.llm import AnthropicClient, LLMCallError, LLMClient, RecordingClient
from policy_extractor.pipeline import (
    DocumentResult,
    apply_spot_checks,
    extraction_phase,
    process_document,
    render_summary,
    summarise,
)
from policy_extractor.runlog import JsonlRunLog, NullRunLog, RunLog, RunWriter, read_records

app = typer.Typer(no_args_is_help=True, help="Insurance policy extraction pipeline.")


class Mode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


InputArg = Annotated[Path, typer.Argument(exists=True, help="A .txt document or a directory.")]
OutOpt = Annotated[Path, typer.Option("--out", help="Directory for run outputs.")]
ModeOpt = Annotated[
    Mode,
    typer.Option(help="live: call the API. record: call and save responses. replay: offline."),
]
FixturesOpt = Annotated[Path, typer.Option(help="Directory of recorded responses.")]

DEFAULT_OUT = Path("runs/latest")
DEFAULT_FIXTURES = Path("fixtures/recorded")


@app.callback()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # The SDK's HTTP client logs every request line; the pipeline logs its own structured line.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def make_client(settings: Settings, mode: Mode, fixtures: Path) -> LLMClient:
    if mode is Mode.REPLAY:
        return RecordingClient(fixtures, "replay")
    live = AnthropicClient.from_settings(settings)
    return RecordingClient(fixtures, "record", live) if mode is Mode.RECORD else live


def run_documents(
    client: LLMClient,
    documents: dict[str, str],
    settings: Settings,
    log: RunLog,
    writer: RunWriter,
) -> list[DocumentResult]:
    """Process every document in isolation, writing each result as soon as it is known."""
    results: list[DocumentResult] = []
    for doc_id, text in documents.items():
        result = process_document(client, doc_id, text, settings, log)
        writer.document(result)
        results.append(result)
        typer.echo(f"{doc_id}: {result.decision.outcome.value}", err=True)
    return results


def finish_run(
    results: list[DocumentResult], settings: Settings, log: RunLog, writer: RunWriter
) -> list[DocumentResult]:
    final = apply_spot_checks(results, settings, log)
    summary = summarise(final)
    log.event("run_summary", summary=summary)
    text = render_summary(summary, final)
    writer.finish(final, summary, text)
    typer.echo(text)
    return final


@app.command()
def run(
    input_path: InputArg,
    out: OutOpt = DEFAULT_OUT,
    mode: ModeOpt = Mode.LIVE,
    fixtures: FixturesOpt = DEFAULT_FIXTURES,
) -> None:
    """Run the full pipeline and write records, decisions, escalations, events and a summary."""
    settings = Settings()
    writer = RunWriter(out)
    log = JsonlRunLog(out / "events.jsonl")
    client = make_client(settings, mode, fixtures)
    results = run_documents(client, load_documents(input_path), settings, log, writer)
    finish_run(results, settings, log, writer)


@app.command()
def extract(
    input_path: InputArg,
    mode: ModeOpt = Mode.LIVE,
    fixtures: FixturesOpt = DEFAULT_FIXTURES,
) -> None:
    """Extract and validate documents (with retry or escalation) and print the outcome."""
    settings = Settings()
    client = make_client(settings, mode, fixtures)
    for doc_id, text in load_documents(input_path).items():
        try:
            outcome = extraction_phase(client, doc_id, text, settings, NullRunLog())
        except LLMCallError as error:
            typer.echo(f"{doc_id}: api_error: {error}", err=True)
            continue
        status = outcome.status + (f" ({outcome.escalation.value})" if outcome.escalation else "")
        typer.echo(f"{doc_id}: {status} after {len(outcome.calls)} call(s)")
        if outcome.extraction:
            typer.echo(outcome.extraction.values().model_dump_json(indent=2))
        for finding in outcome.findings:
            typer.echo(f"finding: {finding.model_dump_json()}")


@app.command("eval")
def evaluate_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, help="A run directory from `run`.")],
    gold: Annotated[Path, typer.Option(exists=True, help="Directory of gold labels.")] = Path(
        "data/gold"
    ),
) -> None:
    """Score a finished run against gold labels and write eval_report.md into the run directory."""
    settings = Settings()
    results = read_records(run_dir / "records.jsonl")
    report = render_report(evaluate(results, load_gold(gold), settings), settings)
    (run_dir / "eval_report.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@app.command()
def batch(
    input_path: InputArg,
    out: OutOpt = Path("runs/batch"),
    poll_interval: Annotated[float, typer.Option(help="Seconds between status checks.")] = 30.0,
) -> None:
    """Run the pipeline through the Message Batches API (half price, not for urgent work)."""
    settings = Settings()
    writer = RunWriter(out)
    log = JsonlRunLog(out / "events.jsonl")
    # Refusal fallback is not available on the Batches API; retries go out as ordinary requests.
    live = AnthropicClient.from_settings(settings)
    try:
        results = run_pipeline_batched(
            live.sdk, live, load_documents(input_path), settings, log, poll_interval_s=poll_interval
        )
    except LLMCallError as error:
        log.event("batch_error", error=error.kind, message=str(error))
        typer.echo(f"batch run failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for result in results:
        writer.document(result)
    finish_run(results, settings, log, writer)
