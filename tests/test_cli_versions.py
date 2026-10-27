"""Tests for `perf-agent versions` (spec 02)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from perf_agent.cli import EXIT_INVALID_WORKSPACE, app
from perf_agent.versions.manager import VersionManager

runner = CliRunner()


def test_lists_versions_numerically_with_info_and_leftover_trials(tmp_path: Path) -> None:
    for n in (1, 10, 9):
        (tmp_path / f"graph_V{n}").mkdir()
    manager = VersionManager(tmp_path, "graph")
    trial = manager.create_trial("graph_V10")
    manager.promote(trial, parent="graph_V10", gain_pct=18.4, idea_title="Use ArrayDeque")
    (tmp_path / "graph_V12_trial").mkdir()
    (tmp_path / "graph_V9" / "VERSION.json").write_text("not json", encoding="utf-8")

    result = runner.invoke(app, ["versions", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "graph_V1",
        "graph_V9  (unreadable VERSION.json)",
        "graph_V10",
        "graph_V11  (+18.4 %, from graph_V10: Use ArrayDeque)",
        "leftover trials: graph_V12_trial",
    ]


def test_custom_prefix(tmp_path: Path) -> None:
    (tmp_path / "bfs_V1").mkdir()
    (tmp_path / "graph_V1").mkdir()

    result = runner.invoke(app, ["versions", str(tmp_path), "--prefix", "bfs"])

    assert result.output.splitlines() == ["bfs_V1"]


def test_empty_workspace(tmp_path: Path) -> None:
    result = runner.invoke(app, ["versions", str(tmp_path)])

    assert result.exit_code == 0
    assert "no graph_V<n> folder" in result.output


def test_missing_workspace_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["versions", str(tmp_path / "missing")])

    assert result.exit_code == EXIT_INVALID_WORKSPACE
    assert "workspace folder not found" in result.output
