"""Shared data models passed between modules (spec 00).

`BuildResult`/`TestResult` (spec 04) and `BudgetExhausted` (spec 06) live here too,
so that `ports.py` depends only on this module and never on a concrete adapter.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


# Base class for all models: unknown fields are rejected.
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# One benchmark case: a graph and a query.
class Combo(_Model):
    """One benchmark case: a graph and a query passed to `run_cmd`."""

    graph: str
    query: str

    @property
    def key(self) -> str:
        return f"{self.graph}::{self.query}"


# The result of running one combo once: its time and its output fingerprint.
class Measurement(_Model):
    """One execution of `run_cmd` for a combo; warmup is done inside the Java class itself."""

    combo: Combo
    time_ns: int
    output_hash: str


# All measurements of one version folder.
class BenchmarkResult(_Model):
    version: str
    measurements: list[Measurement]


# Whether a build succeeded, with its log and duration.
class BuildResult(_Model):
    ok: bool
    log: str
    duration_ns: int


# Whether the tests passed, with their log.
class TestResult(_Model):
    ok: bool
    log: str


# The Planner's proposal: one optimization to try.
class Idea(_Model):
    title: str
    rationale: str
    expected_gain_pct: float | None = None
    risk: Literal["low", "medium", "high"] = "medium"


# The Critic's feedback on a trial and the direction for the next one.
class Critique(_Model):
    explanation: str
    next_direction: str


# How a trial ended; only ACCEPTED creates a new version.
class TrialStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_NO_GAIN = "rejected_no_gain"
    REJECTED_INCORRECT = "rejected_incorrect"
    BUILD_FAILED = "build_failed"
    ERROR = "error"


# The full record of one optimization attempt.
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


# Why the optimization loop stopped.
class StopReason(StrEnum):
    OBJECTIVE_REACHED = "objective_reached"
    NO_PROGRESS = "no_progress"
    MAX_TRIALS = "max_trials"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOO_MANY_ERRORS = "too_many_errors"
    CANCELLED = "cancelled"


# The final result of a run, saved as run.json for the front end.
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


# Raised when the LLM budget is spent; the run then stops.
class BudgetExhausted(Exception):
    """Raised by `ModelRegistryPort.check_budget` once the LLM budget is spent."""


# Raised when an LLM answer hit max_tokens; the trial becomes ERROR.
class LLMTruncated(Exception):
    """Raised when an LLM answer was cut off by `max_tokens`; the partial answer is never used."""
