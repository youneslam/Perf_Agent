"""Wire the nodes into a LangGraph StateGraph, with routing, run config and resume (spec 09)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from perf_agent.agent import nodes
from perf_agent.agent.deps import Deps
from perf_agent.agent.state import LoopState

LoopGraph = CompiledStateGraph[LoopState, None, LoopState, LoopState]
NodeFn = Callable[[Deps, LoopState], LoopState]

# Graph node names and the functions behind them, in loop order.
NODES: dict[str, NodeFn] = {
    "prepare": nodes.prepare,
    "baseline": nodes.baseline,
    "check_stop": nodes.check_stop,
    "plan": nodes.plan,
    "code": nodes.code,
    "build_test": nodes.build_test,
    "fix": nodes.fix,
    "bench": nodes.bench,
    "evaluate": nodes.evaluate_trial,
    "promote": nodes.promote,
    "discard_failed": nodes.discard_failed,
    "critique": nodes.critique,
    "finish": nodes.finish,
}
# Node visits per trial besides build fixes (plan, code, build_test, bench, evaluate, ...).
_NODES_PER_TRIAL = 8


# Build the compiled graph; `deps=None` gives a graph that can only be drawn (`perf-agent graph`).
def build_graph(
    deps: Deps | None, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> LoopGraph:
    max_fixes = deps.scenario.limits.max_build_fixes if deps else 0
    graph: StateGraph[LoopState, None, LoopState, LoopState] = StateGraph(LoopState)
    for name, function in NODES.items():
        # cast: LangGraph's add_node typing does not accept functools.partial objects.
        graph.add_node(name, cast(Any, _bind(function, deps)))

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "baseline")
    graph.add_edge("baseline", "check_stop")
    graph.add_conditional_edges("check_stop", route_check_stop, ["plan", "finish"])
    graph.add_conditional_edges("plan", route_plan, ["code", "discard_failed", "finish"])
    after_code = partial(route_after_write, max_build_fixes=max_fixes)
    graph.add_conditional_edges(
        "code", after_code, ["build_test", "fix", "discard_failed", "finish"]
    )
    graph.add_conditional_edges(
        "fix", after_code, ["build_test", "fix", "discard_failed", "finish"]
    )
    graph.add_conditional_edges(
        "build_test",
        partial(route_build_test, max_build_fixes=max_fixes),
        ["bench", "fix", "discard_failed"],
    )
    graph.add_conditional_edges("bench", route_bench, ["evaluate", "discard_failed"])
    graph.add_edge("evaluate", "promote")
    graph.add_edge("promote", "critique")
    graph.add_edge("discard_failed", "critique")
    graph.add_edge("critique", "check_stop")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


# LangGraph config for one run: the checkpoint thread and a step limit large enough.
def run_config(run_id: str, max_trials: int, max_build_fixes: int) -> RunnableConfig:
    steps_per_trial = _NODES_PER_TRIAL + 2 * max_build_fixes
    return {
        "configurable": {"thread_id": run_id},
        "recursion_limit": max_trials * steps_per_trial + 20,
    }


# Make a checkpointed run continue: a finished run re-enters `check_stop` with new limits.
def prepare_resume(graph: LoopGraph, config: RunnableConfig) -> None:
    snapshot = graph.get_state(config)
    if not snapshot.values:
        thread = config.get("configurable", {}).get("thread_id")
        raise ValueError(f"no saved run with id {thread!r}")
    if not snapshot.next:
        graph.update_state(config, {"stop_reason": None, "summary": None}, as_node="critique")


# check_stop → finish when a stop reason is set, otherwise start a new trial.
def route_check_stop(state: LoopState) -> Literal["plan", "finish"]:
    return "finish" if state.get("stop_reason") else "plan"


# plan → finish (budget), discard_failed (Planner error) or code.
def route_plan(state: LoopState) -> Literal["code", "discard_failed", "finish"]:
    if state.get("stop_reason"):
        return "finish"
    return "discard_failed" if state.get("trial_error") else "code"


# code/fix → build the new source, or handle a renamed class like a failed build.
def route_after_write(
    state: LoopState, max_build_fixes: int
) -> Literal["build_test", "fix", "discard_failed", "finish"]:
    if state.get("stop_reason"):
        return "finish"
    if state.get("trial_error"):
        return "discard_failed"
    if state.get("build_ok") is False:
        return _fix_or_give_up(state, max_build_fixes)
    return "build_test"


# build_test → benchmark when it passed, otherwise fix while attempts are left.
def route_build_test(
    state: LoopState, max_build_fixes: int
) -> Literal["bench", "fix", "discard_failed"]:
    if state.get("trial_error"):
        return "discard_failed"
    if state.get("build_ok"):
        return "bench"
    return _fix_or_give_up(state, max_build_fixes)


# bench → evaluate, or discard_failed when a benchmark run failed.
def route_bench(state: LoopState) -> Literal["evaluate", "discard_failed"]:
    return "discard_failed" if state.get("trial_error") else "evaluate"


# One build + up to `max_build_fixes` fixes per trial.
def _fix_or_give_up(state: LoopState, max_build_fixes: int) -> Literal["fix", "discard_failed"]:
    return "fix" if state.get("build_attempts", 0) <= max_build_fixes else "discard_failed"


# Bind the dependencies into a node function (or a placeholder when only drawing the graph).
def _bind(function: NodeFn, deps: Deps | None) -> Callable[[LoopState], LoopState]:
    if deps is not None:
        return partial(function, deps)

    def unbound(_state: LoopState) -> LoopState:
        raise RuntimeError("this graph was built without dependencies and can only be drawn")

    return unbound
