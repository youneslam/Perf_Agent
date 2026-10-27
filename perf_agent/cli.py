"""Command-line interface (spec 02). The only module that configures logging or prints."""

from __future__ import annotations

import logging
import signal
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import find_dotenv, load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.table import Table

from perf_agent.adapters.java import JavaAdapter
from perf_agent.agent.deps import Deps
from perf_agent.agent.graph import build_graph, prepare_resume, run_config
from perf_agent.agent.nodes import BaselineError
from perf_agent.agent.state import LoopState
from perf_agent.agents.coder import Coder
from perf_agent.agents.critic import Critic
from perf_agent.agents.planner import Planner
from perf_agent.config.scenario import Scenario, ScenarioError, load_scenario
from perf_agent.llm.dry_run import DryRunChatModel
from perf_agent.llm.models import ModelRegistry, default_factory
from perf_agent.models import RunSummary
from perf_agent.storage.noop_repository import NoopRunRepository
from perf_agent.versions.manager import VersionError, VersionManager

logger = logging.getLogger(__name__)

# Exit codes: 0 = success, 1 = the run failed, 2 = bad scenario or workspace, 130 = aborted.
EXIT_RUN_FAILED = 1
EXIT_INVALID_SCENARIO = 2
EXIT_INVALID_WORKSPACE = 2
EXIT_ABORTED = 130

# The `perf-agent` command; each @app.command() below is a sub-command.
app = typer.Typer(
    name="perf-agent",
    help="Iteratively optimize the performance of one Java component with LLMs.",
    no_args_is_help=True,
)


# Runs before every command: set up logging and load the .env file.
@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
) -> None:
    """Configure logging and load `.env` before any command runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Look for .env from the folder the command is run in, not from where the code is installed.
    load_dotenv(find_dotenv(usecwd=True))


# `perf-agent validate SCENARIO`: check a scenario file before running it.
@app.command()
def validate(
    scenario_path: Annotated[
        Path, typer.Argument(metavar="SCENARIO", help="Path to the scenario YAML file.")
    ],
) -> None:
    """Check a scenario file and its base version folder, then print a short summary."""
    try:
        scenario = load_scenario(scenario_path)
    except ScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INVALID_SCENARIO) from exc

    problem = _check_base_version(scenario)
    if problem:
        typer.echo(f"invalid scenario {scenario_path}:\n  - {problem}", err=True)
        raise typer.Exit(EXIT_INVALID_SCENARIO)

    objective = scenario.objective
    typer.echo(f"Scenario '{scenario.name}' is valid.")
    typer.echo(f"  component : {scenario.component.class_name} ({scenario.component.source_file})")
    typer.echo(f"  base      : {scenario.base_dir}")
    typer.echo(f"  combos    : {len(scenario.combos())}")
    typer.echo(
        f"  objective : +{objective.target_gain_pct:g} % total, "
        f"accept a trial at +{objective.accept_threshold_pct:g} %"
    )


# `perf-agent versions WORKSPACE`: list the version folders and leftover trials.
@app.command()
def versions(
    workspace: Annotated[
        Path, typer.Argument(metavar="WORKSPACE", help="Folder containing graph_V1/, graph_V2/, …")
    ],
    prefix: Annotated[str, typer.Option("--prefix", help="Version name prefix.")] = "graph",
) -> None:
    """List version folders in numeric order, and leftover *_trial folders."""
    manager = VersionManager(workspace, prefix)
    try:
        names = manager.list_versions()
        trials = manager.list_trials()
    except VersionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INVALID_WORKSPACE) from exc

    if not names:
        typer.echo(f"no {prefix}_V<n> folder in {manager.workspace}")
    for name in names:
        typer.echo(_describe_version(manager, name))
    if trials:
        typer.echo(f"leftover trials: {', '.join(trials)}")


# `perf-agent run SCENARIO`: run the optimization loop and write workspace/reports/<run_id>/.
@app.command()
def run(
    scenario_path: Annotated[
        Path, typer.Argument(metavar="SCENARIO", help="Path to the scenario YAML file.")
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Use a fake model: no API key, no cost.")
    ] = False,
    max_trials: Annotated[
        int | None, typer.Option("--max-trials", min=1, help="Override limits.max_trials.")
    ] = None,
    resume: Annotated[
        str | None, typer.Option("--resume", metavar="RUN_ID", help="Continue a saved run.")
    ] = None,
) -> None:
    """Optimize the component: plan, code, build, test, benchmark, evaluate, repeat."""
    scenario = _load_runnable_scenario(scenario_path, max_trials)
    deps = _build_deps(scenario, dry_run)
    run_id = resume or _new_run_id()
    checkpoints = scenario.workspace / ".perf_agent" / "checkpoints.sqlite"
    checkpoints.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(checkpoints, check_same_thread=False)) as connection:
        loop = build_graph(deps, SqliteSaver(connection))
        config = run_config(run_id, scenario.limits.max_trials, scenario.limits.max_build_fixes)
        inputs: LoopState | None = {"run_id": run_id}
        if resume:
            try:
                prepare_resume(loop, config)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(EXIT_INVALID_SCENARIO) from exc
            inputs = None
        typer.echo(f"run {run_id}" + (" (dry run)" if dry_run else ""))
        progress = _Progress(scenario)
        try:
            with _ctrl_c_stops_after_trial(deps.cancel):
                for chunk in loop.stream(inputs, config, stream_mode="updates"):
                    for node, update in chunk.items():
                        progress.show(node, update or {})
        except KeyboardInterrupt as exc:
            typer.echo(f"aborted; continue later with: --resume {run_id}", err=True)
            raise typer.Exit(EXIT_ABORTED) from exc
        except BaselineError as exc:
            typer.echo(f"cannot start: {exc}", err=True)
            raise typer.Exit(EXIT_RUN_FAILED) from exc
        except Exception as exc:
            logger.debug("run failed", exc_info=True)
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(EXIT_RUN_FAILED) from exc
        summary = RunSummary.model_validate(loop.get_state(config).values["summary"])

    report = _write_run_json(scenario, summary)
    _print_summary(summary, report)


# `perf-agent graph`: print the optimization loop as a Mermaid diagram.
@app.command()
def graph() -> None:
    """Print the optimization loop as a Mermaid diagram."""
    typer.echo(build_graph(None).get_graph().draw_mermaid())


# Load and check the scenario for a run, applying --max-trials; exit 2 if it cannot run.
def _load_runnable_scenario(path: Path, max_trials: int | None) -> Scenario:
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_INVALID_SCENARIO) from exc
    problem = _check_base_version(scenario)
    if scenario.storage.kind == "neo4j":
        problem = "storage.kind 'neo4j' is not available yet (milestone M9); use 'memory'"
    if problem:
        typer.echo(f"invalid scenario {path}:\n  - {problem}", err=True)
        raise typer.Exit(EXIT_INVALID_SCENARIO)
    if max_trials is not None:
        limits = scenario.limits.model_copy(update={"max_trials": max_trials})
        scenario = scenario.model_copy(update={"limits": limits})
    return scenario


# The composition root: build every collaborator of the loop (the only place that does).
def _build_deps(scenario: Scenario, dry_run: bool) -> Deps:
    factory = (lambda _spec: DryRunChatModel()) if dry_run else default_factory
    registry = ModelRegistry(scenario.llm, scenario.limits.llm_budget_usd, factory=factory)
    return Deps(
        scenario=scenario,
        versions=VersionManager(scenario.workspace, scenario.version_prefix),
        project=JavaAdapter(scenario),
        registry=registry,
        planner=Planner(registry),
        coder=Coder(registry),
        critic=Critic(registry),
        repo=NoopRunRepository(),
    )


# A sortable, unique run id such as 20261027-140312-a1b2c3.
def _new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


# While running, the first Ctrl-C asks the loop to stop after the current trial.
@contextmanager
def _ctrl_c_stops_after_trial(cancel: threading.Event) -> Iterator[None]:
    def handler(_signum: int, _frame: object) -> None:
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()
        typer.echo("\nCtrl-C: stopping after the current trial (press again to abort)", err=True)

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # not in the main thread: keep the default behaviour
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


# Prints one line per meaningful step of the loop.
class _Progress:
    def __init__(self, scenario: Scenario) -> None:
        self._attempts = scenario.limits.max_build_fixes + 1
        self._trial = 0

    # Print the line for one node update, if it has something to say.
    def show(self, node: str, update: dict[str, Any]) -> None:
        line = self._line(node, update)
        if line:
            typer.echo(line)

    # The text for one node update, or None for silent steps.
    def _line(self, node: str, update: dict[str, Any]) -> str | None:
        if node in ("check_stop", "plan", "code", "fix") and update.get("stop_reason"):
            return f"stopping: {update['stop_reason']}"
        if node == "plan" and update.get("trial_index"):
            self._trial = update["trial_index"]
        prefix = f"[trial {self._trial}]"
        if node == "baseline":
            cases = update.get("baseline", {}).get("measurements", [])
            return f"baseline measured on {len(cases)} case(s)"
        if node == "plan" and update.get("idea"):
            return f"{prefix} idea: {update['idea']['title']}"
        if node == "build_test" and "build_ok" in update:
            result = "ok" if update["build_ok"] else "FAILED"
            attempt = f"{update.get('build_attempts')}/{self._attempts}"
            return f"{prefix} build/test {result} (attempt {attempt})"
        if node == "evaluate":
            return f"{prefix} {update.get('verdict', {}).get('reason', '')}"
        if node == "promote" and update.get("promoted_to"):
            return f"{prefix} promoted to {update['promoted_to']}"
        if node == "discard_failed":
            error = (update.get("trial_error") or "").splitlines()
            return f"{prefix} {update.get('trial_status')}: {error[0] if error else ''}"
        return None


# Save the summary as workspace/reports/<run_id>/run.json (the front end's input).
def _write_run_json(scenario: Scenario, summary: RunSummary) -> Path:
    out_dir = scenario.workspace / "reports" / summary.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


# Print the run's result: one row per trial, then the totals.
def _print_summary(summary: RunSummary, report: Path) -> None:
    table = Table(title=f"Run {summary.run_id}")
    for column in ("#", "idea", "status", "gain", "version"):
        table.add_column(column)
    for trial in summary.trials:
        gain = "" if trial.gain_pct is None else f"{trial.gain_pct:+.1f} %"
        table.add_row(
            str(trial.index), trial.idea.title, trial.status.value, gain, trial.promoted_to or ""
        )
    console = Console()
    console.print(table)
    console.print(
        f"{summary.initial_version} -> {summary.final_version}: "
        f"{summary.cumulative_gain_pct:+.1f} %, stopped: {summary.stop_reason.value}, "
        f"LLM cost ${summary.llm_cost_usd:.4f}"
    )
    console.print(f"report: {report}")


# One line for a version, with its gain and idea when VERSION.json exists.
def _describe_version(manager: VersionManager, name: str) -> str:
    try:
        info = manager.read_info(name)
    except ValueError:
        return f"{name}  (unreadable VERSION.json)"
    if info is None:
        return name
    return f"{name}  ({info.gain_pct:+.1f} %, from {info.parent}: {info.idea_title})"


# Check that the base version folder and the component file exist.
def _check_base_version(scenario: Scenario) -> str | None:
    """Return a message if the base version folder or the component file is missing."""
    if not scenario.base_dir.is_dir():
        return f"base_version: folder not found: {scenario.base_dir}"
    source = scenario.base_dir / scenario.component.source_file
    if not source.is_file():
        return f"component.source_file: file not found: {source}"
    return None
