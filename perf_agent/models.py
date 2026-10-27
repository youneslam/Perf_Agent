"""Shared data models passed between modules (spec 00).

`BuildResult`/`TestResult` (spec 04) and `BudgetExhausted` (spec 06) live here too,
so that `ports.py` depends only on this module and never on a concrete adapter.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Combo(_Model):
    """One benchmark case: a graph and a query passed to `run_cmd`."""

    graph: str
    query: str

    @property
    def key(self) -> str:
        return f"{self.graph}::{self.query}"


class Measurement(_Model):
    """One execution of `run_cmd` for a combo; warmup is done inside the Java class itself."""

    combo: Combo
    time_ns: int
    output_hash: str


class BenchmarkResult(_Model):
    version: str
    measurements: list[Measurement]


class BuildResult(_Model):
    ok: bool
    log: str
    duration_ns: int


class TestResult(_Model):
    ok: bool
    log: str


class Idea(_Model):
    title: str
    rationale: str
    expected_gain_pct: float | None = None
    risk: Literal["low", "medium", "high"] = "medium"


class Critique(_Model):
    explanation: str
    next_direction: str


class TrialStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_NO_GAIN = "rejected_no_gain"
    REJECTED_INCORRECT = "rejected_incorrect"
    BUILD_FAILED = "build_failed"
    ERROR = "error"


class Trial(_Model):
    index: int
    base_version: str
    trial_dir: str
    idea: Idea
    status: TrialStatus
    build_attempts: int
    gain_pct: float | None
    promoted_to: str | None
    critique: Critique | None
    error: str | None
    started_at: datetime
    finished_at: datetime


class StopReason(StrEnum):
    OBJECTIVE_REACHED = "objective_reached"
    NO_PROGRESS = "no_progress"
    MAX_TRIALS = "max_trials"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOO_MANY_ERRORS = "too_many_errors"
    CANCELLED = "cancelled"


class RunSummary(_Model):
    run_id: str
    scenario_name: str
    initial_version: str
    final_version: str
    baseline: BenchmarkResult
    trials: list[Trial]
    cumulative_gain_pct: float
    stop_reason: StopReason
    llm_cost_usd: float


class BudgetExhausted(Exception):
    """Raised by `ModelRegistryPort.check_budget` once the LLM budget is spent."""


class LLMTruncated(Exception):
    """Raised when an LLM answer was cut off by `max_tokens`; the partial answer is never used."""
