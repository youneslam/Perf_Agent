"""Tests for the optimization loop (spec 09): fakes for speed, plus one real end-to-end run."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from perf_agent.adapters.java import JavaAdapter
from perf_agent.agent.deps import Deps
from perf_agent.agent.graph import NODES, build_graph, prepare_resume, run_config
from perf_agent.agent.nodes import BaselineError
from perf_agent.agents.coder import CodeAnswer, Coder
from perf_agent.agents.critic import Critic
from perf_agent.agents.planner import Planner
from perf_agent.config.scenario import Scenario
from perf_agent.llm.fake import ScriptedChatModel
from perf_agent.llm.models import ModelRegistry
from perf_agent.models import (
    BudgetExhausted,
    Critique,
    Idea,
    LLMTruncated,
    RunSummary,
    StopReason,
    TrialStatus,
)
from perf_agent.versions.manager import VersionManager
from tests.conftest import STUB_PROJECT, ScenarioData
from tests.fakes import (
    COMPONENT,
    FakeCoder,
    FakeCritic,
    FakePlanner,
    FakeProject,
    FakeRegistry,
    FakeRepository,
    source,
)


def _scenario(
    scenario_data: ScenarioData, workspace: Path, **overrides: dict[str, Any]
) -> Scenario:
    scenario_data["workspace"] = str(workspace)
    for section, values in overrides.items():
        scenario_data[section].update(values)
    return Scenario.model_validate(scenario_data)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace with graph_V1 holding the base component (SPEED=1, RESULT=ok)."""
    component = tmp_path / "ws" / "graph_V1" / COMPONENT
    component.parent.mkdir(parents=True)
    component.write_text(source("SPEED=1", "RESULT=ok"), encoding="utf-8")
    return tmp_path / "ws"


class Loop:
    """Builds the deps and runs the graph to the end."""

    def __init__(self, scenario_data: ScenarioData, workspace: Path) -> None:
        self.scenario_data = scenario_data
        self.workspace = workspace
        self.repo = FakeRepository()
        self.registry = FakeRegistry()
        self.planner = FakePlanner()
        self.critic = FakeCritic()
        self.coder = FakeCoder([])
        self.project = FakeProject()

    def deps(self, coder_script: list[str | BaseException], **overrides: dict[str, Any]) -> Deps:
        self.coder = FakeCoder(coder_script)
        scenario = _scenario(self.scenario_data, self.workspace, **overrides)
        return Deps(
            scenario=scenario,
            versions=VersionManager(self.workspace, scenario.version_prefix),
            project=self.project,
            registry=self.registry,
            planner=self.planner,
            coder=self.coder,
            critic=self.critic,
            repo=self.repo,
        )

    def run(
        self,
        coder_script: list[str | BaseException],
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        **overrides: dict[str, Any],
    ) -> RunSummary:
        deps = self.deps(coder_script, **overrides)
        self.current = deps
        limits = deps.scenario.limits
        config = run_config("run-1", limits.max_trials, limits.max_build_fixes)
        state = build_graph(deps, checkpointer).invoke({"run_id": "run-1"}, config)
        return RunSummary.model_validate(state["summary"])

    def folders(self) -> list[str]:
        return sorted(p.name for p in self.workspace.iterdir() if p.is_dir())


@pytest.fixture
def loop(scenario_data: ScenarioData, workspace: Path) -> Loop:
    return Loop(scenario_data, workspace)


def _statuses(summary: RunSummary) -> list[TrialStatus]:
    return [trial.status for trial in summary.trials]


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    return digest.hexdigest()


# --- graph structure --------------------------------------------------------------------------


def test_graph_compiles_and_draws_every_node() -> None:
    mermaid = build_graph(None).get_graph().draw_mermaid()

    for name in NODES:
        assert name in mermaid


def test_unbound_graph_cannot_run() -> None:
    with pytest.raises(RuntimeError, match="can only be drawn"):
        build_graph(None).invoke({"run_id": "x"})


# --- stop conditions --------------------------------------------------------------------------


def test_accepted_then_two_rejections_stops_on_no_progress(loop: Loop) -> None:
    summary = loop.run([source("SPEED=1.25"), source("SPEED=1.25"), source("SPEED=1.25")])

    assert _statuses(summary) == [
        TrialStatus.ACCEPTED,
        TrialStatus.REJECTED_NO_GAIN,
        TrialStatus.REJECTED_NO_GAIN,
    ]
    assert summary.stop_reason == StopReason.NO_PROGRESS
    assert summary.final_version == "graph_V2"
    assert summary.cumulative_gain_pct == pytest.approx(25)
    assert summary.trials[0].promoted_to == "graph_V2"
    assert summary.trials[1].base_version == "graph_V2"
    assert loop.folders() == ["graph_V1", "graph_V2"]


def test_objective_reached_after_one_trial(loop: Loop) -> None:
    summary = loop.run([source("SPEED=1.2")], objective={"target_gain_pct": 15})

    assert summary.stop_reason == StopReason.OBJECTIVE_REACHED
    assert len(summary.trials) == 1
    assert summary.trials[0].gain_pct == pytest.approx(20, abs=0.01)


def test_max_trials(loop: Loop) -> None:
    summary = loop.run(
        [source("SPEED=1.1"), source("SPEED=1.21")],
        limits={"max_trials": 2},
        objective={"target_gain_pct": 90},
    )

    assert summary.stop_reason == StopReason.MAX_TRIALS
    assert _statuses(summary) == [TrialStatus.ACCEPTED, TrialStatus.ACCEPTED]
    assert summary.final_version == "graph_V3"
    assert summary.cumulative_gain_pct == pytest.approx(21, abs=0.01)


def test_three_consecutive_errors(loop: Loop) -> None:
    summary = loop.run(
        [RuntimeError("boom 1"), RuntimeError("boom 2"), RuntimeError("boom 3")],
        limits={"max_no_progress": 5},
    )

    assert summary.stop_reason == StopReason.TOO_MANY_ERRORS
    assert _statuses(summary) == [TrialStatus.ERROR] * 3
    assert summary.trials[0].error == "coder failed: boom 1"
    assert loop.folders() == ["graph_V1"]


def test_budget_hit_mid_run_leaves_no_trial_folder(loop: Loop) -> None:
    summary = loop.run([source("SPEED=1.1"), BudgetExhausted("budget spent")])

    assert summary.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert _statuses(summary) == [TrialStatus.ACCEPTED]
    assert loop.folders() == ["graph_V1", "graph_V2"]


def test_budget_already_spent_stops_before_planning(loop: Loop) -> None:
    loop.registry.spent = 99.0

    summary = loop.run([])

    assert summary.stop_reason == StopReason.BUDGET_EXHAUSTED
    assert summary.trials == []
    assert loop.planner.calls == 0


def test_cancel_stops_after_the_current_trial(loop: Loop) -> None:
    loop.critic = FakeCritic(on_review=lambda _trial: loop.current.cancel.set())

    summary = loop.run([source("SPEED=1.1"), source("SPEED=1.2")])

    assert summary.stop_reason == StopReason.CANCELLED
    assert len(summary.trials) == 1


# --- one trial's path -------------------------------------------------------------------------


def test_build_fails_then_one_fix_passes(loop: Loop) -> None:
    summary = loop.run(
        [source("SPEED=2", "BREAK_BUILD"), source("SPEED=2")], objective={"target_gain_pct": 50}
    )

    trial = summary.trials[0]
    assert trial.status == TrialStatus.ACCEPTED
    assert trial.build_attempts == 2
    assert loop.coder.calls == [("implement", ""), ("fix", "error: ';' expected")]


def test_build_fails_three_times_gives_build_failed(loop: Loop) -> None:
    broken = source("SPEED=2", "BREAK_BUILD")

    summary = loop.run([broken, broken, broken], limits={"max_trials": 1, "max_build_fixes": 2})

    trial = summary.trials[0]
    assert trial.status == TrialStatus.BUILD_FAILED
    assert trial.build_attempts == 3
    assert trial.error is not None
    assert "build or tests failed after 3 attempt(s)" in trial.error
    assert loop.folders() == ["graph_V1"]


def test_failing_tests_go_through_fix_too(loop: Loop) -> None:
    summary = loop.run(
        [source("SPEED=2", "BREAK_TEST"), source("SPEED=2")], objective={"target_gain_pct": 50}
    )

    assert summary.trials[0].status == TrialStatus.ACCEPTED
    assert loop.coder.calls[1] == ("fix", "expected 3 but was 4")


def test_renamed_class_counts_as_a_failed_build(loop: Loop) -> None:
    summary = loop.run(
        [source("SPEED=2", "RENAMED"), source("SPEED=2")], objective={"target_gain_pct": 50}
    )

    assert summary.trials[0].status == TrialStatus.ACCEPTED
    assert summary.trials[0].build_attempts == 2
    assert "must declare public class Component" in loop.coder.calls[1][1]


def test_incorrect_output_is_rejected_even_if_faster(loop: Loop) -> None:
    summary = loop.run([source("SPEED=3", "RESULT=other")], limits={"max_trials": 1})

    trial = summary.trials[0]
    assert trial.status == TrialStatus.REJECTED_INCORRECT
    assert trial.gain_pct == pytest.approx(200, abs=0.01)
    assert summary.final_version == "graph_V1"
    assert loop.folders() == ["graph_V1"]


def test_benchmark_crash_is_an_error(loop: Loop) -> None:
    summary = loop.run([source("CRASH")], limits={"max_trials": 1})

    assert summary.trials[0].status == TrialStatus.ERROR
    assert "benchmark failed" in (summary.trials[0].error or "")
    assert loop.folders() == ["graph_V1"]


def test_truncated_coder_answer_is_an_error(loop: Loop) -> None:
    truncated = LLMTruncated("answer cut off at max_tokens=64000; raise llm.max_tokens")

    summary = loop.run([truncated], limits={"max_trials": 1})

    trial = summary.trials[0]
    assert trial.status == TrialStatus.ERROR
    assert trial.error == "coder failed: answer cut off at max_tokens=64000; raise llm.max_tokens"
    assert loop.folders() == ["graph_V1"]


def test_planner_failure_is_recorded_without_calling_the_critic(loop: Loop) -> None:
    loop.planner = FakePlanner(errors={1: ValueError("bad answer")})

    summary = loop.run([], limits={"max_trials": 1})

    trial = summary.trials[0]
    assert trial.status == TrialStatus.ERROR
    assert trial.error == "planner failed: bad answer"
    assert loop.critic.reviews == []


# --- history, critic and invariants -----------------------------------------------------------


def test_planner_sees_the_history_and_critic_sees_the_diff(loop: Loop) -> None:
    loop.run([source("SPEED=1.25"), source("SPEED=1.25"), source("SPEED=1.25")])

    assert [len(h) for h in loop.planner.histories] == [0, 1, 2]
    second_history = loop.planner.histories[1]
    assert second_history[0].critique == Critique(explanation="reviewed 1", next_direction="go on")
    _trial, verdict, diff = loop.critic.reviews[0]
    assert verdict is not None and verdict.accepted
    assert "-// SPEED=1\n" in diff and "+// SPEED=1.25\n" in diff


def test_repository_receives_the_whole_run(loop: Loop) -> None:
    loop.run([source("SPEED=2")], objective={"target_gain_pct": 50})

    assert loop.repo.events == [
        ("start_run", "run-1"),
        ("save_baseline", "graph_V1"),
        ("save_version", ("graph_V2", "graph_V1", 100.0)),
        ("save_trial", 1),
        ("finish_run", "objective_reached"),
    ]
    assert loop.repo.summary is not None


def test_base_version_is_never_modified(loop: Loop, workspace: Path) -> None:
    before = _tree_hash(workspace / "graph_V1")

    loop.run([source("SPEED=1.25"), source("BREAK_BUILD")] + [source("BREAK_BUILD")] * 2)

    assert _tree_hash(workspace / "graph_V1") == before


def test_leftover_trial_from_a_crash_is_cleaned_at_start(loop: Loop, workspace: Path) -> None:
    (workspace / "graph_V7_trial").mkdir()

    loop.run([], limits={"max_trials": 1})

    assert "graph_V7_trial" not in loop.folders()


def test_broken_base_version_is_fatal(loop: Loop, workspace: Path) -> None:
    (workspace / "graph_V1" / COMPONENT).write_text(source("BREAK_BUILD"), encoding="utf-8")

    with pytest.raises(BaselineError, match="graph_V1 does not build"):
        loop.run([])


# --- resume -----------------------------------------------------------------------------------


def test_resume_continues_a_cancelled_run_at_the_next_trial(loop: Loop, tmp_path: Path) -> None:
    loop.critic = FakeCritic(on_review=lambda trial: loop.current.cancel.set())
    with closing(sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False)) as connection:
        saver = SqliteSaver(connection)
        first = loop.run([source("SPEED=1.25")], checkpointer=saver)
        assert first.stop_reason == StopReason.CANCELLED

        loop.critic = FakeCritic()
        deps = loop.deps([source("SPEED=1.25"), source("SPEED=1.25")])
        graph = build_graph(deps, saver)
        config = run_config("run-1", deps.scenario.limits.max_trials, 2)
        prepare_resume(graph, config)
        state = graph.invoke(None, config)

    resumed = RunSummary.model_validate(state["summary"])
    assert [t.index for t in resumed.trials] == [1, 2, 3]
    assert resumed.trials[0].status == TrialStatus.ACCEPTED
    assert resumed.stop_reason == StopReason.NO_PROGRESS


def test_resume_unknown_run(loop: Loop, tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False)) as connection:
        graph = build_graph(loop.deps([]), SqliteSaver(connection))
        with pytest.raises(ValueError, match="no saved run with id 'nope'"):
            prepare_resume(graph, run_config("nope", 5, 2))


# --- real end-to-end run ----------------------------------------------------------------------


def test_end_to_end_on_the_stub_project(scenario_data: ScenarioData, tmp_path: Path) -> None:
    shutil.copytree(STUB_PROJECT, tmp_path / "project")
    workspace = tmp_path / "project"
    py = sys.executable
    scenario = _scenario(
        scenario_data,
        workspace,
        project={
            "build_cmd": [py, "build.py"],
            "test_cmd": [py, "test.py"],
            "run_cmd": [py, "run.py", "{graph}", "{query}"],
        },
    )
    original = (workspace / "graph_V1" / scenario.component.source_file).read_text(encoding="utf-8")
    model = ScriptedChatModel(
        script=[
            Idea(title="Double the speed", rationale="test", risk="low"),
            CodeAnswer(java_source=original.replace("SPEED=1", "SPEED=2"), summary="faster"),
            Critique(explanation="2x faster", next_direction="stop here"),
        ]
    )
    registry = ModelRegistry(scenario.llm, 5.0, factory=lambda _spec: model, retry_wait=False)
    deps = Deps(
        scenario=scenario,
        versions=VersionManager(workspace, scenario.version_prefix),
        project=JavaAdapter(scenario),
        registry=registry,
        planner=Planner(registry),
        coder=Coder(registry),
        critic=Critic(registry),
        repo=FakeRepository(),
    )

    state = build_graph(deps).invoke({"run_id": "e2e"}, run_config("e2e", 5, 2))

    summary = RunSummary.model_validate(state["summary"])
    assert summary.stop_reason == StopReason.OBJECTIVE_REACHED
    assert summary.final_version == "graph_V2"
    assert summary.cumulative_gain_pct == pytest.approx(100)
    assert (workspace / "graph_V2" / "VERSION.json").is_file()
    assert sorted(p.name for p in workspace.iterdir() if p.name.startswith("graph")) == [
        "graph_V1",
        "graph_V2",
    ]
    assert summary.trials[0].critique == Critique(
        explanation="2x faster", next_direction="stop here"
    )
