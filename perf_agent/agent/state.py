"""LangGraph state of the optimization loop (spec 09).

Only JSON-serialisable values are stored (models as `model_dump(mode="json")` dicts), so the
checkpointer can save the state after every node and a run can be resumed.
"""

from __future__ import annotations

from typing import Any, TypedDict, TypeVar

from pydantic import BaseModel

from perf_agent.models import Trial

ModelT = TypeVar("ModelT", bound=BaseModel)


# Everything the loop knows; each node returns only the keys it changes.
class LoopState(TypedDict, total=False):
    run_id: str
    base_version: str  # current base, e.g. "graph_V2"
    initial_baseline: dict[str, Any]  # BenchmarkResult of the starting version
    baseline: dict[str, Any]  # BenchmarkResult of the current base
    trials: list[dict[str, Any]]  # finished Trial objects
    # current trial
    trial_index: int
    trial_base: str  # version the current trial started from
    trial_dir: str | None
    idea: dict[str, Any] | None
    source_before: str
    source_after: str | None
    build_attempts: int
    build_ok: bool | None  # False = last build/test (or class check) failed
    last_log: str
    bench: dict[str, Any] | None
    verdict: dict[str, Any] | None
    trial_status: str | None
    trial_error: str | None
    promoted_to: str | None
    trial_started_at: str
    # end
    stop_reason: str | None
    summary: dict[str, Any] | None


# Fields that start empty for every new trial.
def new_trial_fields() -> LoopState:
    return {
        "trial_dir": None,
        "idea": None,
        "source_after": None,
        "build_attempts": 0,
        "build_ok": None,
        "last_log": "",
        "bench": None,
        "verdict": None,
        "trial_status": None,
        "trial_error": None,
        "promoted_to": None,
    }


# A model as a JSON-ready dict for the state.
def dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


# Read `state[key]` back as a model, or None when it is empty.
def load(state: LoopState, key: str, cls: type[ModelT]) -> ModelT | None:
    value = state.get(key)
    return None if value is None else cls.model_validate(value)


# Like `load`, for keys that must be set at this point of the loop.
def require(state: LoopState, key: str, cls: type[ModelT]) -> ModelT:
    model = load(state, key, cls)
    if model is None:
        raise RuntimeError(f"loop state has no {key!r} yet")
    return model


# The finished trials as models.
def trials(state: LoopState) -> list[Trial]:
    return [Trial.model_validate(t) for t in state.get("trials", [])]
