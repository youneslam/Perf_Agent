"""Command-line interface (spec 02). The only module that configures logging or prints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from perf_agent.config.scenario import Scenario, ScenarioError, load_scenario
from perf_agent.versions.manager import VersionError, VersionManager

EXIT_INVALID_SCENARIO = 2
EXIT_INVALID_WORKSPACE = 2

app = typer.Typer(
    name="perf-agent",
    help="Iteratively optimize the performance of one Java component with LLMs.",
    no_args_is_help=True,
)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
) -> None:
    """Configure logging and load `.env` before any command runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()


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


def _describe_version(manager: VersionManager, name: str) -> str:
    try:
        info = manager.read_info(name)
    except ValueError:
        return f"{name}  (unreadable VERSION.json)"
    if info is None:
        return name
    return f"{name}  ({info.gain_pct:+.1f} %, from {info.parent}: {info.idea_title})"


def _check_base_version(scenario: Scenario) -> str | None:
    """Return a message if the base version folder or the component file is missing."""
    if not scenario.base_dir.is_dir():
        return f"base_version: folder not found: {scenario.base_dir}"
    source = scenario.base_dir / scenario.component.source_file
    if not source.is_file():
        return f"component.source_file: file not found: {source}"
    return None
