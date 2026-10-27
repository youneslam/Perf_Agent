"""Critic: explains a trial's result and suggests the next direction (spec 07.3).

The Critic never decides acceptance: the Evaluator already did, with fixed rules.
"""

from __future__ import annotations

import difflib

from perf_agent.agents.prompting import chat_prompt, head_lines
from perf_agent.evaluation.evaluator import Verdict
from perf_agent.models import Critique, Trial
from perf_agent.ports import ModelRegistryPort

# How many lines of the code diff are shown to the Critic.
MAX_DIFF_LINES = 400


# Asks the LLM why a trial worked or failed, and what to try next.
class Critic:
    def __init__(self, registry: ModelRegistryPort) -> None:
        self._registry = registry
        self._chain = chat_prompt("critic_system", "critic_task") | registry.structured(Critique)

    # Review one finished trial; `verdict` is None when it never reached the benchmark.
    def review(self, trial: Trial, verdict: Verdict | None, diff: str) -> Critique:
        values = {
            "index": str(trial.index),
            "idea_title": trial.idea.title,
            "idea_rationale": trial.idea.rationale,
            "status": trial.status.value,
            "gain": "not measured" if trial.gain_pct is None else f"{trial.gain_pct:+.1f} %",
            "reason": verdict.reason if verdict else "no verdict (the trial was not benchmarked)",
            "error": trial.error or "none",
            "ratios": _format_ratios(verdict),
            "diff": head_lines(diff, MAX_DIFF_LINES) if diff.strip() else "(no change)",
        }
        self._registry.check_budget()
        return self._chain.invoke(values)


# Unified diff between two versions of the component, for the Critic and the report.
def make_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


# One line per benchmark case with its speed ratio, or a note if there were no measurements.
def _format_ratios(verdict: Verdict | None) -> str:
    if verdict is None or not verdict.per_combo_ratio:
        return "not measured"
    return "\n".join(f"- {key}: {ratio:.2f}x" for key, ratio in verdict.per_combo_ratio.items())
