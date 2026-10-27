"""The loop's steps: one function per graph node, `(deps, state) -> partial state` (spec 09).

Nodes that work on a trial never raise for a trial problem: they record it in `trial_error`
(or `build_ok=False`) and the router sends the trial to `discard_failed`. A spent budget sets
`stop_reason` instead. Only a broken base version is fatal for the whole run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from perf_agent.adapters.java import ComponentContractError
from perf_agent.agent.deps import Deps
from perf_agent.agent.state import LoopState, dump, load, new_trial_fields, require, trials
from perf_agent.agents.critic import make_diff
from perf_agent.agents.prompting import tail_lines
from perf_agent.evaluation.evaluator import Verdict, cumulative_gain_pct, evaluate
from perf_agent.models import (
    BenchmarkResult,
    BudgetExhausted,
    Critique,
    Idea,
    RunSummary,
    StopReason,
    Trial,
    TrialStatus,
)

logger = logging.getLogger(__name__)

# How many lines of a failed build/test log are kept in the trial's error message.
ERROR_LOG_LINES = 40
# Trial statuses that count as errors for the TOO_MANY_ERRORS stop rule.
_ERROR_STATUSES = {TrialStatus.ERROR, TrialStatus.BUILD_FAILED}


# The base version cannot be built, tested or benchmarked: the run cannot start.
class BaselineError(RuntimeError):
    pass


# Clean leftovers from an interrupted run, check the base version and start the run record.
def prepare(deps: Deps, state: LoopState) -> LoopState:
    removed = deps.versions.cleanup_stale_trials()
    if removed:
        logger.warning("removed %d leftover trial folder(s)", len(removed))
    base = deps.scenario.base_version
    deps.versions.check_base(base)
    deps.repo.start_run(state["run_id"], deps.scenario)
    return {"base_version": base, "trials": [], "trial_index": 0, "stop_reason": None}


# Build, test and benchmark the base version; every trial is compared with these numbers.
def baseline(deps: Deps, state: LoopState) -> LoopState:
    name = state["base_version"]
    base_dir = deps.versions.check_base(name)
    build = deps.project.build(base_dir)
    if not build.ok:
        raise BaselineError(f"{name} does not build:\n{tail_lines(build.log, ERROR_LOG_LINES)}")
    test = deps.project.test(base_dir)
    if not test.ok:
        raise BaselineError(f"{name} fails its tests:\n{tail_lines(test.log, ERROR_LOG_LINES)}")
    result = deps.project.benchmark(base_dir, deps.scenario.combos())
    deps.repo.save_baseline(state["run_id"], result)
    return {"initial_baseline": dump(result), "baseline": dump(result)}


# Decide whether to stop before the next trial (first matching rule wins).
def check_stop(deps: Deps, state: LoopState) -> LoopState:
    if state.get("stop_reason"):
        return {}
    limits, objective = deps.scenario.limits, deps.scenario.objective
    done = trials(state)
    statuses = [t.status for t in done]
    initial = require(state, "initial_baseline", BenchmarkResult)
    current = require(state, "baseline", BenchmarkResult)

    reason: StopReason | None = None
    if deps.cancel.is_set():
        reason = StopReason.CANCELLED
    elif deps.registry.spent_usd >= limits.llm_budget_usd:
        reason = StopReason.BUDGET_EXHAUSTED
    elif cumulative_gain_pct(initial, current) >= objective.target_gain_pct:
        reason = StopReason.OBJECTIVE_REACHED
    elif len(done) >= limits.max_trials:
        reason = StopReason.MAX_TRIALS
    elif _last_n_all(statuses, limits.max_consecutive_errors, lambda s: s in _ERROR_STATUSES):
        reason = StopReason.TOO_MANY_ERRORS
    elif _last_n_all(statuses, limits.max_no_progress, lambda s: s != TrialStatus.ACCEPTED):
        reason = StopReason.NO_PROGRESS
    if reason:
        logger.info("stopping: %s", reason.value)
    return {"stop_reason": reason.value if reason else None}


# Start a new trial: read the base source and ask the Planner for one idea.
def plan(deps: Deps, state: LoopState) -> LoopState:
    base = state["base_version"]
    update: LoopState = {
        **new_trial_fields(),
        "trial_index": state["trial_index"] + 1,
        "trial_base": base,
        "trial_started_at": _now(),
        "source_before": "",
    }
    try:
        source = deps.project.read_component(deps.versions.check_base(base))
        update["source_before"] = source
        idea = deps.planner.propose(
            source,
            require(state, "baseline", BenchmarkResult),
            trials(state),
            deps.scenario.objective,
        )
    except BudgetExhausted:
        return {**update, "stop_reason": StopReason.BUDGET_EXHAUSTED.value}
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR trial
        logger.warning("the Planner failed: %s", exc)
        placeholder = Idea(title="(no idea: the Planner failed)", rationale=str(exc))
        return {**update, "idea": dump(placeholder), "trial_error": f"planner failed: {exc}"}
    logger.info("trial %d: %s", update["trial_index"], idea.title)
    return {**update, "idea": dump(idea)}


# Copy the base into a trial folder and write the Coder's implementation of the idea.
def code(deps: Deps, state: LoopState) -> LoopState:
    trial_dir: Path | None = None
    try:
        trial_dir = deps.versions.create_trial(state["base_version"])
        answer = deps.coder.implement(
            state["source_before"], require(state, "idea", Idea), deps.scenario.component.class_name
        )
    except BudgetExhausted:
        _discard_quietly(deps, trial_dir)
        return {"trial_dir": None, "stop_reason": StopReason.BUDGET_EXHAUSTED.value}
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR trial
        return {"trial_dir": _str(trial_dir), "trial_error": f"coder failed: {exc}"}
    return _write_source(deps, trial_dir, answer.java_source, state.get("build_attempts", 0))


# Build then test the trial; a failure is sent to `fix` while attempts are left.
def build_test(deps: Deps, state: LoopState) -> LoopState:
    trial_dir = Path(_need(state, "trial_dir"))
    attempts = state.get("build_attempts", 0) + 1
    try:
        build = deps.project.build(trial_dir)
        if not build.ok:
            return {"build_ok": False, "last_log": build.log, "build_attempts": attempts}
        test = deps.project.test(trial_dir)
        if not test.ok:
            return {"build_ok": False, "last_log": test.log, "build_attempts": attempts}
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR trial
        return {"build_attempts": attempts, "trial_error": f"build/test could not run: {exc}"}
    return {"build_ok": True, "last_log": "", "build_attempts": attempts}


# Ask the Coder to correct the failed source, using the build/test log.
def fix(deps: Deps, state: LoopState) -> LoopState:
    trial_dir = Path(_need(state, "trial_dir"))
    try:
        answer = deps.coder.fix(
            state.get("source_after") or state["source_before"],
            state.get("last_log", ""),
            require(state, "idea", Idea),
            deps.scenario.component.class_name,
        )
    except BudgetExhausted:
        _discard_quietly(deps, trial_dir)
        return {"trial_dir": None, "stop_reason": StopReason.BUDGET_EXHAUSTED.value}
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR trial
        return {"trial_error": f"coder fix failed: {exc}"}
    return _write_source(deps, trial_dir, answer.java_source, state.get("build_attempts", 0))


# Run every benchmark case once on the trial.
def bench(deps: Deps, state: LoopState) -> LoopState:
    try:
        result = deps.project.benchmark(Path(_need(state, "trial_dir")), deps.scenario.combos())
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR trial
        return {"trial_error": f"benchmark failed: {exc}"}
    return {"bench": dump(result)}


# Compare the trial with its base using the fixed rules, and set the trial's status.
def evaluate_trial(deps: Deps, state: LoopState) -> LoopState:
    verdict = evaluate(
        require(state, "baseline", BenchmarkResult),
        require(state, "bench", BenchmarkResult),
        deps.scenario.objective.accept_threshold_pct,
    )
    if verdict.accepted:
        status = TrialStatus.ACCEPTED
    elif not verdict.correct:
        status = TrialStatus.REJECTED_INCORRECT
    else:
        status = TrialStatus.REJECTED_NO_GAIN
    logger.info("trial %d: %s", state["trial_index"], verdict.reason)
    return {"verdict": dump(verdict), "trial_status": status.value}


# Accepted: the trial folder becomes the next version and the new base. Otherwise: delete it.
def promote(deps: Deps, state: LoopState) -> LoopState:
    trial_dir = Path(_need(state, "trial_dir"))
    if state.get("trial_status") != TrialStatus.ACCEPTED.value:
        _discard_quietly(deps, trial_dir)
        return {}
    verdict = require(state, "verdict", Verdict)
    idea = require(state, "idea", Idea)
    base = state["base_version"]
    try:
        name = deps.versions.promote(trial_dir, base, verdict.gain_pct, idea.title)
        deps.repo.save_version(state["run_id"], name, base, verdict.gain_pct)
    except Exception as exc:  # noqa: BLE001 - recorded as an ERROR trial
        _discard_quietly(deps, trial_dir)
        return {"trial_status": TrialStatus.ERROR.value, "trial_error": f"promotion failed: {exc}"}
    new_baseline = require(state, "bench", BenchmarkResult).model_copy(update={"version": name})
    logger.info("promoted %s (%+.1f %%)", name, verdict.gain_pct)
    return {"base_version": name, "baseline": dump(new_baseline), "promoted_to": name}


# A trial that failed to build/test or hit an error: delete its folder and set its status.
def discard_failed(deps: Deps, state: LoopState) -> LoopState:
    trial_dir = state.get("trial_dir")
    if trial_dir:
        _discard_quietly(deps, Path(trial_dir))
    error = state.get("trial_error")
    if error:
        return {"trial_status": TrialStatus.ERROR.value, "trial_error": error}
    attempts = state.get("build_attempts", 0)
    log = tail_lines(state.get("last_log", ""), ERROR_LOG_LINES)
    return {
        "trial_status": TrialStatus.BUILD_FAILED.value,
        "trial_error": f"build or tests failed after {attempts} attempt(s):\n{log}",
    }


# Ask the Critic about the finished trial, then record the trial in the history.
def critique(deps: Deps, state: LoopState) -> LoopState:
    verdict = load(state, "verdict", Verdict)
    trial = _finished_trial(state, critique=None, verdict=verdict)
    update: LoopState = {}
    review: Critique | None = None
    if state.get("source_after") is not None:  # skip the Critic when no code was written
        diff = make_diff(
            state["source_before"],
            state.get("source_after") or "",
            deps.scenario.component.source_file,
        )
        try:
            review = deps.critic.review(trial, verdict, diff)
        except BudgetExhausted:
            update["stop_reason"] = StopReason.BUDGET_EXHAUSTED.value
        except Exception as exc:  # noqa: BLE001 - a missing critique must not lose the trial
            logger.warning("the Critic failed: %s", exc)
    trial = trial.model_copy(update={"critique": review})
    deps.repo.save_trial(state["run_id"], trial, load(state, "bench", BenchmarkResult))
    return {**update, "trials": [*state.get("trials", []), dump(trial)], "trial_dir": None}


# End of the run: make sure no trial folder is left, then build and record the summary.
def finish(deps: Deps, state: LoopState) -> LoopState:
    trial_dir = state.get("trial_dir")
    if trial_dir:
        _discard_quietly(deps, Path(trial_dir))
    deps.versions.cleanup_stale_trials()
    initial = require(state, "initial_baseline", BenchmarkResult)
    current = require(state, "baseline", BenchmarkResult)
    summary = RunSummary(
        run_id=state["run_id"],
        scenario_name=deps.scenario.name,
        initial_version=deps.scenario.base_version,
        final_version=state["base_version"],
        baseline=initial,
        trials=trials(state),
        cumulative_gain_pct=cumulative_gain_pct(initial, current),
        stop_reason=StopReason(state.get("stop_reason") or StopReason.MAX_TRIALS.value),
        llm_cost_usd=deps.registry.spent_usd,
    )
    deps.repo.finish_run(state["run_id"], summary)
    return {"trial_dir": None, "summary": dump(summary)}


# Write new source into the trial; a renamed class counts as a failed build attempt.
def _write_source(deps: Deps, trial_dir: Path, source: str, attempts: int) -> LoopState:
    update: LoopState = {"trial_dir": str(trial_dir), "source_after": source}
    try:
        deps.project.write_component(trial_dir, source)
    except ComponentContractError as exc:
        return {**update, "build_ok": False, "last_log": str(exc), "build_attempts": attempts + 1}
    return {**update, "build_ok": None}


# The Trial record for the history, from the current-trial fields of the state.
def _finished_trial(state: LoopState, critique: Critique | None, verdict: Verdict | None) -> Trial:
    trial_dir = state.get("trial_dir")
    return Trial(
        index=state["trial_index"],
        base_version=state["trial_base"],
        trial_dir=Path(trial_dir).name if trial_dir else "",
        idea=require(state, "idea", Idea),
        status=TrialStatus(state.get("trial_status") or TrialStatus.ERROR.value),
        build_attempts=state.get("build_attempts", 0),
        gain_pct=verdict.gain_pct if verdict else None,
        promoted_to=state.get("promoted_to"),
        critique=critique,
        error=state.get("trial_error"),
        started_at=datetime.fromisoformat(state["trial_started_at"]),
        finished_at=datetime.now(UTC),
    )


# True if the last `n` statuses all match `condition` (and there are at least `n`).
def _last_n_all(
    statuses: list[TrialStatus], n: int, condition: Callable[[TrialStatus], bool]
) -> bool:
    return len(statuses) >= n and all(condition(s) for s in statuses[-n:])


# Delete a trial folder without letting a cleanup problem hide the real result.
def _discard_quietly(deps: Deps, trial_dir: Path | None) -> None:
    if trial_dir is None:
        return
    try:
        deps.versions.discard(trial_dir)
    except Exception as exc:  # noqa: BLE001 - cleanup must not crash the loop
        logger.error("could not delete trial folder %s: %s", trial_dir, exc)


# The current time as an ISO string for the state.
def _now() -> str:
    return datetime.now(UTC).isoformat()


# `str(path)` that keeps None.
def _str(path: Path | None) -> str | None:
    return None if path is None else str(path)


# A state key that must be set at this point of the loop.
def _need(state: LoopState, key: str) -> str:
    value = state.get(key)
    if not value:
        raise RuntimeError(f"loop state has no {key!r} yet")
    return str(value)
