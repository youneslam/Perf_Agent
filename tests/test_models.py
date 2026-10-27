"""Tests for perf_agent/models.py (spec 00)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from perf_agent import models
from perf_agent.models import (
    BenchmarkResult,
    Combo,
    Critique,
    Idea,
    Measurement,
    RunSummary,
    StopReason,
    Trial,
    TrialStatus,
)


def _measurement(graph: str, query: str, time_ns: int) -> Measurement:
    return Measurement(combo=Combo(graph=graph, query=query), time_ns=time_ns, output_hash="ab12")


def test_combo_key() -> None:
    assert Combo(graph="small", query="q1").key == "small::q1"


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Combo(graph="small", query="q1", oops=1)  # type: ignore[call-arg]


def test_measurement_needs_an_integer_time() -> None:
    with pytest.raises(ValidationError, match="time_ns"):
        Measurement(combo=Combo(graph="g", query="q"), time_ns="fast", output_hash="x")  # type: ignore[arg-type]


def test_idea_defaults() -> None:
    idea = Idea(title="Use ArrayDeque", rationale="Fewer allocations")
    assert idea.risk == "medium"
    assert idea.expected_gain_pct is None


def test_idea_rejects_unknown_risk() -> None:
    with pytest.raises(ValidationError, match="'low', 'medium' or 'high'"):
        Idea(title="x", rationale="y", risk="extreme")  # type: ignore[arg-type]


def test_enums_are_plain_strings() -> None:
    assert TrialStatus.ACCEPTED == "accepted"
    assert StopReason("max_trials") is StopReason.MAX_TRIALS


def test_build_and_test_results() -> None:
    assert models.BuildResult(ok=False, log="error: ';' expected", duration_ns=10).ok is False
    assert models.TestResult(ok=True, log="").ok is True


def test_run_summary_json_round_trip() -> None:
    now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    baseline = BenchmarkResult(version="graph_V1", measurements=[_measurement("small", "q1", 200)])
    trial = Trial(
        index=1,
        base_version="graph_V1",
        trial_dir="graph_V2_trial",
        idea=Idea(title="Use ArrayDeque", rationale="Fewer allocations", risk="low"),
        status=TrialStatus.ACCEPTED,
        build_attempts=1,
        gain_pct=18.4,
        promoted_to="graph_V2",
        critique=Critique(explanation="Less GC", next_direction="Try an int[] queue"),
        error=None,
        started_at=now,
        finished_at=now,
    )
    summary = RunSummary(
        run_id="run-1",
        scenario_name="stub-speedup",
        initial_version="graph_V1",
        final_version="graph_V2",
        baseline=baseline,
        trials=[trial],
        cumulative_gain_pct=18.4,
        stop_reason=StopReason.MAX_TRIALS,
        llm_cost_usd=0.42,
    )

    assert RunSummary.model_validate_json(summary.model_dump_json()) == summary


def test_errors_are_exceptions() -> None:
    assert issubclass(models.BudgetExhausted, Exception)
    assert issubclass(models.LLMTruncated, Exception)
