"""Planner: proposes ONE optimization idea for the next trial (spec 07.1)."""

from __future__ import annotations

import logging
from typing import Any

from perf_agent.agents.prompting import chat_prompt
from perf_agent.config.scenario import Objective
from perf_agent.models import BenchmarkResult, Idea, Trial
from perf_agent.ports import ModelRegistryPort

logger = logging.getLogger(__name__)


# Asks the LLM for one idea, given the code, current timings, past trials and the objective.
class Planner:
    def __init__(self, registry: ModelRegistryPort) -> None:
        self._registry = registry
        self._chain = chat_prompt("planner_system", "planner_task") | registry.structured(Idea)

    # Return a new idea; if it repeats a past trial's title, ask once more for something else.
    def propose(
        self,
        component_source: str,
        baseline: BenchmarkResult,
        history: list[Trial],
        objective: Objective,
    ) -> Idea:
        values: dict[str, Any] = {
            "target_gain_pct": f"{objective.target_gain_pct:g}",
            "accept_threshold_pct": f"{objective.accept_threshold_pct:g}",
            "baseline_times": format_times(baseline),
            "history": format_history(history),
            "component_source": component_source,
            "extra_instruction": "",
        }
        idea = self._ask(values)
        if not _already_tried(idea, history):
            return idea

        tried = ", ".join(f'"{trial.idea.title}"' for trial in history)
        values["extra_instruction"] = (
            f'\nThe idea "{idea.title}" was already tried. '
            f"Propose something different from: {tried}."
        )
        retry = self._ask(values)
        if _already_tried(retry, history):
            logger.warning("the Planner repeated a past idea twice: %r", retry.title)
        return retry

    # One LLM call, after checking that budget is left.
    def _ask(self, values: dict[str, Any]) -> Idea:
        self._registry.check_budget()
        return self._chain.invoke(values)


# One line per benchmark case with its execution time in milliseconds.
def format_times(result: BenchmarkResult) -> str:
    return "\n".join(
        f"- {m.combo.key}: {m.time_ns / 1_000_000:.3f} ms" for m in result.measurements
    )


# Compact history: title, outcome, gain and the critic's suggestion of each past trial.
def format_history(history: list[Trial]) -> str:
    if not history:
        return "No trial yet: this is the first one."
    lines = []
    for trial in sorted(history, key=lambda t: t.index):
        gain = "" if trial.gain_pct is None else f", gain {trial.gain_pct:+.1f} %"
        lines.append(f'- Trial {trial.index}: "{trial.idea.title}" -> {trial.status.value}{gain}')
        if trial.critique:
            lines.append(f"  Critic: {trial.critique.next_direction}")
    latest = next((t.critique for t in sorted(history, key=lambda t: -t.index) if t.critique), None)
    if latest:
        lines.append(f"\nLatest suggestion from the critic: {latest.next_direction}")
    return "\n".join(lines)


# True if this idea's title matches a past trial's title (ignoring case and spaces).
def _already_tried(idea: Idea, history: list[Trial]) -> bool:
    title = idea.title.strip().casefold()
    return any(trial.idea.title.strip().casefold() == title for trial in history)
