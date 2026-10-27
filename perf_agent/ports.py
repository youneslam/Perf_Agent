"""Protocols the core logic depends on (spec 00).

Graph nodes, agents and the evaluator only know these interfaces; concrete adapters
(folders on disk, Java commands, LangChain models, memory or Neo4j storage) implement them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from perf_agent.config.scenario import Scenario
from perf_agent.models import (
    BenchmarkResult,
    BuildResult,
    Combo,
    RunSummary,
    TestResult,
    Trial,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class VersionManagerPort(Protocol):
    """Version folders on disk: `graph_V1/`, `graph_V2/`, … and the current `*_trial` folder."""

    def check_base(self, name: str) -> Path:
        """Return the folder of version `name`; raise `VersionError` if it is missing."""
        ...

    def create_trial(self, base: str) -> Path:
        """Copy version `base` to a new `<prefix>_V<n+1>_trial/` folder and return its path."""
        ...

    def promote(self, trial_dir: Path, parent: str, gain_pct: float, idea_title: str) -> str:
        """Write `VERSION.json`, then turn an accepted trial folder into the next version;
        return its name (`graph_V2`)."""
        ...

    def discard(self, trial_dir: Path) -> None:
        """Delete a rejected or failed trial folder."""
        ...

    def cleanup_stale_trials(self) -> list[Path]:
        """Delete trial folders left by an interrupted run; return them."""
        ...


class ProjectAdapterPort(Protocol):
    """Read, rewrite, build, test and benchmark the component inside one version folder."""

    def read_component(self, version_dir: Path) -> str:
        """Return the Java source of the component in `version_dir`."""
        ...

    def write_component(self, version_dir: Path, source: str) -> None:
        """Overwrite the component in a `*_trial` folder (same path, package and class name)."""
        ...

    def build(self, version_dir: Path) -> BuildResult:
        """Run `build_cmd` in `version_dir`."""
        ...

    def test(self, version_dir: Path) -> TestResult:
        """Run `test_cmd` in `version_dir`."""
        ...

    def benchmark(self, version_dir: Path, combos: list[Combo]) -> BenchmarkResult:
        """Run `run_cmd` once per combo and measure its time and output hash."""
        ...


class ModelRegistryPort(Protocol):
    """The single LLM shared by the Planner, Coder and Critic, with the run's LLM budget."""

    def chat_model(self) -> BaseChatModel:
        """Return the raw LangChain chat model (budget tracking attached)."""
        ...

    def structured(self, schema: type[SchemaT]) -> Runnable[LanguageModelInput, SchemaT]:
        """Return a runnable whose answer is validated into `schema` (with retry and fallback).

        Raises `LLMTruncated` when the answer was cut off by `max_tokens`.
        """
        ...

    @property
    def spent_usd(self) -> float:
        """Total LLM cost of the run so far, in US dollars."""
        ...

    def check_budget(self) -> None:
        """Raise `BudgetExhausted` if the LLM budget is spent."""
        ...


class RunRepositoryPort(Protocol):
    """History of a run: baseline, trials, version lineage and final summary."""

    def start_run(self, run_id: str, scenario: Scenario) -> None:
        """Record that run `run_id` started with `scenario`."""
        ...

    def save_baseline(self, run_id: str, result: BenchmarkResult) -> None:
        """Record the benchmark of the starting version."""
        ...

    def save_trial(self, run_id: str, trial: Trial, bench: BenchmarkResult | None) -> None:
        """Record a finished trial, with its benchmark if it got that far."""
        ...

    def save_version(self, run_id: str, name: str, parent: str, gain_pct: float) -> None:
        """Record a promoted version and the version it was derived from."""
        ...

    def finish_run(self, run_id: str, summary: RunSummary) -> None:
        """Record the final summary of the run."""
        ...

    def history(self, run_id: str) -> list[Trial]:
        """Return the trials of the run, ordered by index."""
        ...
