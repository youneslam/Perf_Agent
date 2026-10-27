"""Tests for perf_agent/agents (spec 07), through the real ModelRegistry and a scripted model."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from perf_agent.agents.coder import LOG_TAIL_LINES, CodeAnswer, Coder, strip_code_fence
from perf_agent.agents.critic import MAX_DIFF_LINES, Critic, make_diff
from perf_agent.agents.planner import Planner
from perf_agent.agents.prompting import chat_prompt
from perf_agent.config.scenario import LLMConfig, Objective
from perf_agent.evaluation.evaluator import Verdict
from perf_agent.llm.fake import ScriptedChatModel, Truncated
from perf_agent.llm.models import ModelRegistry
from perf_agent.models import (
    BenchmarkResult,
    BudgetExhausted,
    Combo,
    Critique,
    Idea,
    LLMTruncated,
    Measurement,
    Trial,
    TrialStatus,
)

CLASS_NAME = "com.example.graph.BfsTraversal"
SOURCE = "package com.example.graph;\n\npublic class BfsTraversal {\n}\n"
IDEA = Idea(title="Use ArrayDeque", rationale="Fewer allocations than LinkedList", risk="low")
OBJECTIVE = Objective(target_gain_pct=30, accept_threshold_pct=5)
BASELINE = BenchmarkResult(
    version="graph_V1",
    measurements=[
        Measurement(combo=Combo(graph="small", query="q1"), time_ns=1_250_000, output_hash="a"),
        Measurement(combo=Combo(graph="large", query="q2"), time_ns=98_000_000, output_hash="b"),
    ],
)


def _registry(model: ScriptedChatModel, budget_usd: float = 5.0) -> ModelRegistry:
    config = LLMConfig(provider="anthropic", model="test-model")
    return ModelRegistry(config, budget_usd, factory=lambda _spec: model, retry_wait=False)


def _prompt_text(model: ScriptedChatModel, call: int = -1) -> str:
    """Everything the agent sent to the model in one call (system + task)."""
    return "\n".join(str(message.content) for message in model.calls[call])


def _trial(index: int, title: str, **fields: Any) -> Trial:
    now = datetime(2026, 10, 27, tzinfo=UTC)
    values: dict[str, Any] = {
        "index": index,
        "base_version": "graph_V1",
        "trial_dir": f"graph_V{index + 1}_trial",
        "idea": Idea(title=title, rationale="because"),
        "status": TrialStatus.REJECTED_NO_GAIN,
        "build_attempts": 1,
        "gain_pct": 1.5,
        "promoted_to": None,
        "critique": None,
        "error": None,
        "started_at": now,
        "finished_at": now,
    }
    values.update(fields)
    return Trial(**values)


# --- Planner ----------------------------------------------------------------------------------


def test_planner_returns_the_idea_with_full_context() -> None:
    model = ScriptedChatModel(script=[IDEA])
    history = [
        _trial(1, "Cache neighbour arrays", status=TrialStatus.ACCEPTED, gain_pct=12.0),
        _trial(
            2,
            "Unroll the inner loop",
            critique=Critique(explanation="JIT already unrolls", next_direction="Try a bitset"),
        ),
    ]

    idea = Planner(_registry(model)).propose(SOURCE, BASELINE, history, OBJECTIVE)

    assert idea == IDEA
    prompt = _prompt_text(model)
    assert "senior Java performance engineer" in prompt  # system prompt
    assert '"Cache neighbour arrays" -> accepted, gain +12.0 %' in prompt
    assert '"Unroll the inner loop" -> rejected_no_gain' in prompt
    assert "Latest suggestion from the critic: Try a bitset" in prompt
    assert "- small::q1: 1.250 ms" in prompt
    assert "- large::q2: 98.000 ms" in prompt
    assert "cumulative gain of 30 %" in prompt
    assert "at least 5 % faster" in prompt
    assert "public class BfsTraversal" in prompt


def test_planner_first_trial_has_no_history() -> None:
    model = ScriptedChatModel(script=[IDEA])

    Planner(_registry(model)).propose(SOURCE, BASELINE, [], OBJECTIVE)

    assert "No trial yet" in _prompt_text(model)


def test_planner_asks_once_more_for_a_duplicate_idea() -> None:
    fresh = Idea(title="Use a bitset for visited nodes", rationale="Less memory traffic")
    model = ScriptedChatModel(script=[Idea(title="  use arraydeque ", rationale="x"), fresh])
    history = [_trial(1, "Use ArrayDeque")]

    idea = Planner(_registry(model)).propose(SOURCE, BASELINE, history, OBJECTIVE)

    assert idea == fresh
    assert len(model.calls) == 2
    retry_prompt = _prompt_text(model, 1)
    assert 'was already tried. Propose something different from: "Use ArrayDeque"' in retry_prompt
    assert "Propose something different from" not in _prompt_text(model, 0)


def test_planner_accepts_a_second_duplicate_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = ScriptedChatModel(script=[IDEA, IDEA])

    idea = Planner(_registry(model)).propose(SOURCE, BASELINE, [_trial(1, IDEA.title)], OBJECTIVE)

    assert idea == IDEA
    assert len(model.calls) == 2
    assert "repeated a past idea twice" in caplog.text


# --- Coder ------------------------------------------------------------------------------------


def test_coder_implement_sends_idea_and_source() -> None:
    answer = CodeAnswer(java_source=SOURCE, summary="Switched the queue to ArrayDeque.")
    model = ScriptedChatModel(script=[answer])

    result = Coder(_registry(model)).implement(SOURCE, IDEA, CLASS_NAME)

    assert result == answer
    prompt = _prompt_text(model)
    assert "Return the COMPLETE new file" in prompt  # system prompt rules
    assert "Title: Use ArrayDeque" in prompt
    assert "`BfsTraversal` in package `com.example.graph`" in prompt
    assert "public class BfsTraversal" in prompt


def test_coder_strips_a_java_fence() -> None:
    fenced = CodeAnswer(java_source=f"```java\n{SOURCE}```\n", summary="x")

    result = Coder(_registry(ScriptedChatModel(script=[fenced]))).implement(
        SOURCE, IDEA, CLASS_NAME
    )

    assert result.java_source == SOURCE


def test_coder_fix_sends_only_the_tail_of_a_long_log() -> None:
    log = "\n".join(f"log line {n}" for n in range(1, 1001))
    model = ScriptedChatModel(script=[CodeAnswer(java_source=SOURCE, summary="fixed")])

    Coder(_registry(model)).fix(SOURCE, log, IDEA, CLASS_NAME)

    prompt = _prompt_text(model)
    assert "log line 1000" in prompt
    assert f"log line {1000 - LOG_TAIL_LINES + 1}" in prompt
    assert f"log line {1000 - LOG_TAIL_LINES}\n" not in prompt
    assert "log line 1\n" not in prompt
    assert "850 earlier lines omitted" in prompt
    assert "failed to build or to pass the tests" in prompt


def test_coder_never_returns_a_truncated_answer() -> None:
    model = ScriptedChatModel(script=[Truncated('{"java_source": "package com.exa')])

    with pytest.raises(LLMTruncated):
        Coder(_registry(model)).implement(SOURCE, IDEA, CLASS_NAME)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (f"```java\n{SOURCE}```", SOURCE),
        (f"  ```\n{SOURCE}```  \n", SOURCE),
        (SOURCE, SOURCE),
        ('class A { String s = "```"; }', 'class A { String s = "```"; }'),
    ],
)
def test_strip_code_fence(source: str, expected: str) -> None:
    assert strip_code_fence(source) == expected


# --- Critic -----------------------------------------------------------------------------------


def test_critic_prompt_contains_the_result_and_the_diff() -> None:
    critique = Critique(explanation="Fewer allocations", next_direction="Try a bitset")
    model = ScriptedChatModel(script=[critique])
    verdict = Verdict(
        correct=True,
        mismatched_combos=[],
        per_combo_ratio={"small::q1": 1.25, "large::q2": 1.1},
        gain_pct=17.3,
        accepted=True,
        reason="accepted: gain +17.3 % >= +5 % threshold",
    )
    trial = _trial(3, IDEA.title, status=TrialStatus.ACCEPTED, gain_pct=17.3)
    diff = make_diff(SOURCE, SOURCE.replace("{\n}", "{\n  int x;\n}"), "src/BfsTraversal.java")

    result = Critic(_registry(model)).review(trial, verdict, diff)

    assert result == critique
    prompt = _prompt_text(model)
    assert "You do not decide whether the trial is kept" in prompt
    assert "## Trial 3" in prompt
    assert "Status: accepted" in prompt
    assert "Gain over the parent version: +17.3 %" in prompt
    assert "- small::q1: 1.25x" in prompt
    assert "+  int x;" in prompt
    assert "--- a/src/BfsTraversal.java" in prompt


def test_critic_without_verdict_explains_the_error() -> None:
    model = ScriptedChatModel(script=[Critique(explanation="x", next_direction="y")])
    trial = _trial(
        1,
        IDEA.title,
        status=TrialStatus.BUILD_FAILED,
        gain_pct=None,
        error="error: cannot find symbol ArrayDeque",
    )

    Critic(_registry(model)).review(trial, None, "")

    prompt = _prompt_text(model)
    assert "Gain over the parent version: not measured" in prompt
    assert "no verdict (the trial was not benchmarked)" in prompt
    assert "Error: error: cannot find symbol ArrayDeque" in prompt
    assert "(no change)" in prompt


def test_critic_truncates_a_long_diff() -> None:
    model = ScriptedChatModel(script=[Critique(explanation="x", next_direction="y")])
    diff = "\n".join(f"+line {n}" for n in range(1, 1001))

    Critic(_registry(model)).review(_trial(1, IDEA.title), None, diff)

    prompt = _prompt_text(model)
    assert f"+line {MAX_DIFF_LINES}\n" in prompt
    assert f"+line {MAX_DIFF_LINES + 1}\n" not in prompt
    assert "600 more lines omitted" in prompt


# --- shared rules -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "task", "variables"),
    [
        (
            "planner_system",
            "planner_task",
            {
                "target_gain_pct",
                "accept_threshold_pct",
                "baseline_times",
                "history",
                "component_source",
                "extra_instruction",
            },
        ),
        (
            "coder_system",
            "coder_implement",
            {"idea_title", "idea_rationale", "class_name", "package", "component_source"},
        ),
        (
            "coder_system",
            "coder_fix",
            {"idea_title", "idea_rationale", "class_name", "current_source", "log_tail"},
        ),
        (
            "critic_system",
            "critic_task",
            {
                "index",
                "idea_title",
                "idea_rationale",
                "status",
                "gain",
                "reason",
                "error",
                "ratios",
                "diff",
            },
        ),
    ],
)
def test_prompt_templates_format_with_exactly_their_variables(
    system: str, task: str, variables: set[str]
) -> None:
    prompt = chat_prompt(system, task)

    assert set(prompt.input_variables) == variables
    messages = prompt.format_messages(**{name: f"<{name}> {{braces}}" for name in variables})
    assert len(messages) == 2
    assert all(f"<{name}>" in str(messages[1].content) for name in variables)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: Planner(r).propose(SOURCE, BASELINE, [], OBJECTIVE),
        lambda r: Coder(r).implement(SOURCE, IDEA, CLASS_NAME),
        lambda r: Coder(r).fix(SOURCE, "error", IDEA, CLASS_NAME),
        lambda r: Critic(r).review(_trial(1, IDEA.title), None, ""),
    ],
    ids=["planner", "coder-implement", "coder-fix", "critic"],
)
def test_budget_exhausted_propagates_before_any_call(call: Any) -> None:
    model = ScriptedChatModel(script=[IDEA])

    with pytest.raises(BudgetExhausted):
        call(_registry(model, budget_usd=0))
    assert model.calls == []
