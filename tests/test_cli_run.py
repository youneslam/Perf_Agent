"""Tests for `perf-agent run` and `perf-agent graph` (spec 02), using --dry-run on the stub."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from perf_agent.cli import EXIT_INVALID_SCENARIO, EXIT_RUN_FAILED, app
from perf_agent.models import RunSummary, StopReason
from tests.conftest import STUBS_DIR

runner = CliRunner()


@pytest.fixture
def scenario_path(tmp_path: Path) -> Path:
    """A copy of the stub scenario + project whose commands run with this Python."""
    shutil.copytree(STUBS_DIR, tmp_path / "stubs")
    path = tmp_path / "stubs" / "scenario.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    py = sys.executable
    data["project"].update(
        build_cmd=[py, "build.py"],
        test_cmd=[py, "test.py"],
        run_cmd=[py, "run.py", "{graph}", "{query}"],
    )
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _edit(path: Path, section: str, **values: object) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data[section].update(values)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _run_id(output: str) -> str:
    match = re.search(r"^run (\S+)", output, re.MULTILINE)
    assert match, output
    return match.group(1)


def _summary(scenario_path: Path, run_id: str) -> RunSummary:
    path = scenario_path.parent / "project" / "reports" / run_id / "run.json"
    return RunSummary.model_validate_json(path.read_text(encoding="utf-8"))


def test_dry_run_end_to_end(scenario_path: Path) -> None:
    result = runner.invoke(app, ["run", str(scenario_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "(dry run)" in result.output
    assert "[trial 1] idea: Dry-run idea 1" in result.output
    assert "[trial 1] build/test ok (attempt 1/3)" in result.output
    assert "[trial 1] promoted to graph_V2" in result.output
    assert "stopping: objective_reached" in result.output
    summary = _summary(scenario_path, _run_id(result.output))
    assert summary.final_version == "graph_V2"
    assert summary.cumulative_gain_pct == pytest.approx(100)
    project = scenario_path.parent / "project"
    assert sorted(p.name for p in project.iterdir() if p.name.startswith("graph")) == [
        "graph_V1",
        "graph_V2",
    ]


def test_max_trials_override(scenario_path: Path) -> None:
    _edit(scenario_path, "objective", target_gain_pct=10_000)

    result = runner.invoke(app, ["run", str(scenario_path), "--dry-run", "--max-trials", "2"])

    assert result.exit_code == 0, result.output
    summary = _summary(scenario_path, _run_id(result.output))
    assert summary.stop_reason == StopReason.MAX_TRIALS
    assert summary.final_version == "graph_V3"


def test_resume_a_finished_run_with_more_trials(scenario_path: Path) -> None:
    _edit(scenario_path, "objective", target_gain_pct=10_000)
    first = runner.invoke(app, ["run", str(scenario_path), "--dry-run", "--max-trials", "1"])
    run_id = _run_id(first.output)

    second = runner.invoke(
        app, ["run", str(scenario_path), "--dry-run", "--max-trials", "2", "--resume", run_id]
    )

    assert second.exit_code == 0, second.output
    summary = _summary(scenario_path, run_id)
    assert [t.index for t in summary.trials] == [1, 2]
    assert summary.final_version == "graph_V3"


def test_resume_unknown_run_exits_2(scenario_path: Path) -> None:
    result = runner.invoke(app, ["run", str(scenario_path), "--dry-run", "--resume", "nope"])

    assert result.exit_code == EXIT_INVALID_SCENARIO
    assert "no saved run with id 'nope'" in result.output


def test_neo4j_storage_is_not_available_yet(scenario_path: Path) -> None:
    _edit(scenario_path, "storage", kind="neo4j", neo4j_uri="bolt://localhost:7687")

    result = runner.invoke(app, ["run", str(scenario_path), "--dry-run"])

    assert result.exit_code == EXIT_INVALID_SCENARIO
    assert "not available yet (milestone M9)" in result.output


def test_invalid_scenario_exits_2(scenario_path: Path) -> None:
    _edit(scenario_path, "objective", accept_threshold_pct=0)

    result = runner.invoke(app, ["run", str(scenario_path), "--dry-run"])

    assert result.exit_code == EXIT_INVALID_SCENARIO


def test_broken_base_version_exits_1(scenario_path: Path) -> None:
    component = (
        scenario_path.parent / "project/graph_V1/src/main/java/com/example/graph/Component.java"
    )
    component.write_text(
        "// BREAK_BUILD\n" + component.read_text(encoding="utf-8"), encoding="utf-8"
    )

    result = runner.invoke(app, ["run", str(scenario_path), "--dry-run"])

    assert result.exit_code == EXIT_RUN_FAILED
    assert "cannot start: graph_V1 does not build" in result.output


def test_graph_prints_mermaid() -> None:
    result = runner.invoke(app, ["graph"])

    assert result.exit_code == 0
    assert "graph TD;" in result.output
    assert "build_test -.-> fix;" in result.output
