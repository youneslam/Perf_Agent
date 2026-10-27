"""Tests for `perf-agent validate` (spec 02)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perf_agent.cli import EXIT_INVALID_SCENARIO, app
from tests.conftest import STUB_SCENARIO, STUBS_DIR

runner = CliRunner()


@pytest.fixture
def stub_copy(tmp_path: Path) -> Path:
    """A writable copy of tests/stubs/ (scenario + project); returns the scenario path."""
    shutil.copytree(STUBS_DIR, tmp_path / "stubs")
    return tmp_path / "stubs" / "scenario.yaml"


def test_validate_stub_project_exits_0() -> None:
    result = runner.invoke(app, ["validate", str(STUB_SCENARIO)])

    assert result.exit_code == 0, result.output
    assert "Scenario 'stub-speedup' is valid." in result.output
    assert "com.example.graph.Component" in result.output
    assert "combos    : 6" in result.output
    assert "+30 % total, accept a trial at +5 %" in result.output


def test_missing_component_file_exits_2(stub_copy: Path) -> None:
    component = stub_copy.parent / "project/graph_V1/src/main/java/com/example/graph/Component.java"
    component.unlink()

    result = runner.invoke(app, ["validate", str(stub_copy)])

    assert result.exit_code == EXIT_INVALID_SCENARIO
    assert "component.source_file: file not found" in result.output


def test_missing_base_version_folder_exits_2(stub_copy: Path) -> None:
    shutil.rmtree(stub_copy.parent / "project" / "graph_V1")

    result = runner.invoke(app, ["validate", str(stub_copy)])

    assert result.exit_code == EXIT_INVALID_SCENARIO
    assert "base_version: folder not found" in result.output


def test_invalid_scenario_exits_2(stub_copy: Path) -> None:
    stub_copy.write_text(stub_copy.read_text(encoding="utf-8") + "oops: 1\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(stub_copy)])

    assert result.exit_code == EXIT_INVALID_SCENARIO
    assert "oops: Extra inputs are not permitted" in result.output


def test_missing_scenario_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])

    assert result.exit_code == EXIT_INVALID_SCENARIO
    assert "cannot read scenario" in result.output


def test_verbose_flag_is_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "validate", str(STUB_SCENARIO)])

    assert result.exit_code == 0, result.output


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "validate" in result.output


def test_env_file_is_read_from_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("PERF_AGENT_TEST_VAR=from-cwd\n", encoding="utf-8")
    monkeypatch.delenv("PERF_AGENT_TEST_VAR", raising=False)
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["validate", str(STUB_SCENARIO)])

    assert os.environ.get("PERF_AGENT_TEST_VAR") == "from-cwd"
    monkeypatch.delenv("PERF_AGENT_TEST_VAR")
