"""Decide, with fixed rules and no LLM, whether a trial is correct and fast enough (spec 05).

Pure functions: no I/O. `baseline` is always the current base version (the trial's parent),
so every promotion must beat its parent, not only the original V1.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from perf_agent.models import BenchmarkResult, Measurement

# Any combo whose speed ratio drops below this rejects the trial (20 % speed loss).
MAX_REGRESSION_RATIO = 0.8


# The Evaluator's decision for one trial, with the numbers that justify it.
class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct: bool
    mismatched_combos: list[str]  # combo keys whose output_hash differs
    per_combo_ratio: dict[str, float]  # baseline.time_ns / trial.time_ns (> 1 = faster)
    gain_pct: float  # (geomean(ratios) - 1) * 100
    accepted: bool
    reason: str


# Compare a trial with its parent version and decide: correct? fast enough? accepted?
def evaluate(
    baseline: BenchmarkResult, trial: BenchmarkResult, accept_threshold_pct: float
) -> Verdict:
    base_by_key, trial_by_key = _by_key(baseline), _by_key(trial)
    _check_same_combos(base_by_key, trial_by_key)

    keys = list(base_by_key)
    mismatched = [k for k in keys if trial_by_key[k].output_hash != base_by_key[k].output_hash]
    ratios = {k: base_by_key[k].time_ns / trial_by_key[k].time_ns for k in keys}
    gain = _gain_pct(list(ratios.values()))
    regressions = [k for k, ratio in ratios.items() if ratio < MAX_REGRESSION_RATIO]

    problems = []
    if mismatched:
        problems.append(f"output differs on {', '.join(mismatched)}")
    for key in regressions:
        slower_pct = (1 - ratios[key]) * 100
        problems.append(
            f"{key} is {slower_pct:.1f} % slower (limit {(1 - MAX_REGRESSION_RATIO) * 100:.0f} %)"
        )
    if gain < accept_threshold_pct:
        problems.append(f"gain {gain:+.1f} % < {accept_threshold_pct:+g} % threshold")

    accepted = not problems
    reason = (
        f"accepted: gain {gain:+.1f} % >= {accept_threshold_pct:+g} % threshold"
        if accepted
        else "rejected: " + "; ".join(problems)
    )
    return Verdict(
        correct=not mismatched,
        mismatched_combos=mismatched,
        per_combo_ratio=ratios,
        gain_pct=gain,
        accepted=accepted,
        reason=reason,
    )


# Total gain of the current version over the initial one, used for the stop objective.
def cumulative_gain_pct(initial: BenchmarkResult, current: BenchmarkResult) -> float:
    """Gain of `current` over `initial` (e.g. the final version over V1), in percent."""
    initial_by_key, current_by_key = _by_key(initial), _by_key(current)
    _check_same_combos(initial_by_key, current_by_key)
    return _gain_pct([initial_by_key[k].time_ns / current_by_key[k].time_ns for k in initial_by_key])


# Index measurements by combo key, rejecting duplicates and non-positive times.
def _by_key(result: BenchmarkResult) -> dict[str, Measurement]:
    by_key: dict[str, Measurement] = {}
    for measurement in result.measurements:
        key = measurement.combo.key
        if key in by_key:
            raise ValueError(f"{result.version}: combo {key} is measured twice")
        if measurement.time_ns <= 0:
            raise ValueError(f"{result.version}: {key} has a non-positive time_ns")
        by_key[key] = measurement
    return by_key


# Both benchmarks must measure exactly the same combos, otherwise it is a bug.
def _check_same_combos(left: dict[str, Measurement], right: dict[str, Measurement]) -> None:
    if not left:
        raise ValueError("no measurements to compare")
    if left.keys() != right.keys():
        only_left = sorted(left.keys() - right.keys())
        only_right = sorted(right.keys() - left.keys())
        raise ValueError(
            f"benchmarks cover different combos (only in baseline: {only_left}, "
            f"only in trial: {only_right})"
        )


# Geometric mean of the speed ratios as a percentage gain (2x faster = +100 %).
def _gain_pct(ratios: list[float]) -> float:
    """(geometric mean of the ratios - 1) * 100, rounded to hide float noise."""
    geomean = math.exp(math.fsum(math.log(r) for r in ratios) / len(ratios))
    return round((geomean - 1) * 100, 9)
