"""Placeholder storage until the Neo4j repository (M9): nothing is persisted.

The loop keeps its own history in its state, and the CLI writes `run.json`, so a run works
end to end; only the long-term history across runs is missing.
"""

from __future__ import annotations

import logging

from perf_agent.config.scenario import Scenario
from perf_agent.models import BenchmarkResult, RunSummary, Trial

logger = logging.getLogger(__name__)


# Implements RunRepositoryPort by logging each event at DEBUG level and storing nothing.
class NoopRunRepository:
    # A run starts.
    def start_run(self, run_id: str, scenario: Scenario) -> None:
        logger.debug("run %s started (%s): not persisted", run_id, scenario.name)

    # The base version was benchmarked.
    def save_baseline(self, run_id: str, result: BenchmarkResult) -> None:
        logger.debug("run %s baseline %s: not persisted", run_id, result.version)

    # A trial finished.
    def save_trial(self, run_id: str, trial: Trial, bench: BenchmarkResult | None) -> None:
        logger.debug("run %s trial %d: not persisted", run_id, trial.index)

    # A new version was promoted.
    def save_version(self, run_id: str, name: str, parent: str, gain_pct: float) -> None:
        logger.debug("run %s version %s from %s: not persisted", run_id, name, parent)

    # The run finished.
    def finish_run(self, run_id: str, summary: RunSummary) -> None:
        logger.debug("run %s finished (%s): not persisted", run_id, summary.stop_reason)

    # No history is stored, so there is none to return.
    def history(self, run_id: str) -> list[Trial]:
        return []
