"""Tests for perf_agent/llm (spec 06): registry, budget, pricing and the scripted model.

No network: every model is a `ScriptedChatModel`, except the `live` test (skipped by default).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from perf_agent.config.scenario import LLMConfig, ModelSpec
from perf_agent.llm.fake import ScriptedChatModel, Truncated
from perf_agent.llm.models import ModelRegistry, default_factory
from perf_agent.llm.pricing import cost_usd
from perf_agent.models import BudgetExhausted, Critique, Idea, LLMTruncated

IDEA = Idea(title="Use ArrayDeque", rationale="Fewer allocations than LinkedList", risk="low")
PROMPT = [HumanMessage(content="Propose one optimization for BfsTraversal")]
# With the fake usage (100 in, 50 out), one call costs 100*10/1e6 + 50*20/1e6 = $0.002.
TEST_PRICES = {"main": (10.0, 20.0), "backup": (10.0, 20.0)}


def _config(fallback: bool = False) -> LLMConfig:
    return LLMConfig(
        provider="anthropic",
        model="main",
        max_tokens=1234,
        fallback=ModelSpec(provider="ollama", model="backup") if fallback else None,
    )


def _registry(
    main: ScriptedChatModel,
    backup: ScriptedChatModel | None = None,
    budget_usd: float = 5.0,
) -> ModelRegistry:
    models = {"main": main, "backup": backup or ScriptedChatModel()}

    def factory(spec: ModelSpec) -> ScriptedChatModel:
        return models[spec.model]

    return ModelRegistry(
        _config(fallback=backup is not None),
        budget_usd,
        factory=factory,
        prices=TEST_PRICES,
        retry_wait=False,
    )


# --- registry ---------------------------------------------------------------------------------


def test_model_is_built_once_and_shared() -> None:
    built: list[ModelSpec] = []

    def factory(spec: ModelSpec) -> ScriptedChatModel:
        built.append(spec)
        return ScriptedChatModel()

    registry = ModelRegistry(_config(), 5.0, factory=factory)

    assert registry.chat_model() is registry.chat_model()
    assert [(s.model, s.max_tokens) for s in built] == [("main", 1234)]


def test_structured_returns_the_parsed_object_and_records_the_prompt() -> None:
    main = ScriptedChatModel(script=[IDEA])

    idea = _registry(main).structured(Idea).invoke(PROMPT)

    assert idea == IDEA
    assert main.calls == [PROMPT]


def test_structured_works_for_any_schema() -> None:
    critique = Critique(explanation="Less GC pressure", next_direction="Try an int[] queue")

    result = _registry(ScriptedChatModel(script=[critique])).structured(Critique).invoke(PROMPT)

    assert result == critique


def test_transient_error_is_retried() -> None:
    main = ScriptedChatModel(script=[TimeoutError("slow"), IDEA])

    assert _registry(main).structured(Idea).invoke(PROMPT) == IDEA
    assert len(main.calls) == 2


def test_unparseable_answer_is_retried() -> None:
    main = ScriptedChatModel(script=["this is not JSON", IDEA])

    assert _registry(main).structured(Idea).invoke(PROMPT) == IDEA
    assert len(main.calls) == 2


def test_gives_up_after_three_attempts() -> None:
    main = ScriptedChatModel(script=[TimeoutError("1"), TimeoutError("2"), TimeoutError("3")])

    with pytest.raises(TimeoutError, match="3"):
        _registry(main).structured(Idea).invoke(PROMPT)
    assert len(main.calls) == 3


def test_truncated_answer_raises_without_retry() -> None:
    main = ScriptedChatModel(script=[Truncated('{"title": "Use Arr'), IDEA])

    with pytest.raises(LLMTruncated, match="cut off at max_tokens=1234; raise llm.max_tokens"):
        _registry(main).structured(Idea).invoke(PROMPT)
    assert len(main.calls) == 1


def test_truncated_answer_is_not_sent_to_the_fallback() -> None:
    main = ScriptedChatModel(script=[Truncated("{")])
    backup = ScriptedChatModel(script=[IDEA])

    with pytest.raises(LLMTruncated):
        _registry(main, backup).structured(Idea).invoke(PROMPT)
    assert backup.calls == []


def test_fallback_answers_when_the_main_model_keeps_failing() -> None:
    main = ScriptedChatModel(script=[TimeoutError()] * 3)
    backup = ScriptedChatModel(script=[IDEA])
    registry = _registry(main, backup)

    assert registry.structured(Idea).invoke(PROMPT) == IDEA
    assert len(main.calls) == 3
    assert [call.model for call in registry.calls] == ["backup"]


# --- budget -----------------------------------------------------------------------------------


def test_each_call_is_priced_and_recorded() -> None:
    registry = _registry(ScriptedChatModel(script=[IDEA, IDEA]))

    registry.structured(Idea).invoke(PROMPT)
    registry.structured(Idea).invoke(PROMPT)

    assert registry.spent_usd == pytest.approx(0.004)
    assert [(c.model, c.input_tokens, c.output_tokens) for c in registry.calls] == [
        ("main", 100, 50),
        ("main", 100, 50),
    ]
    assert all(c.duration_s >= 0 for c in registry.calls)


def test_budget_is_exhausted_after_one_expensive_call() -> None:
    registry = _registry(ScriptedChatModel(script=[IDEA]), budget_usd=0.001)
    registry.check_budget()

    registry.structured(Idea).invoke(PROMPT)

    with pytest.raises(BudgetExhausted, match=r"spent \$0.0020 of \$0.00"):
        registry.check_budget()


def test_failed_calls_are_not_charged() -> None:
    registry = _registry(ScriptedChatModel(script=[TimeoutError(), IDEA]))

    registry.structured(Idea).invoke(PROMPT)

    assert len(registry.calls) == 1


def test_each_call_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    registry = _registry(ScriptedChatModel(script=[IDEA]))

    with caplog.at_level(logging.INFO, logger="perf_agent.llm.budget"):
        registry.structured(Idea).invoke(PROMPT)

    assert "LLM call main: 100 in / 50 out tokens, $0.0020" in caplog.text
    assert "Propose one optimization" not in caplog.text  # prompts only at DEBUG


# --- pricing ----------------------------------------------------------------------------------


def test_known_claude_price() -> None:
    assert cost_usd("anthropic", "claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)


def test_ollama_is_free_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert cost_usd("ollama", "qwen2.5-coder", 10_000, 10_000) == 0.0
    assert caplog.text == ""


def test_unknown_model_costs_zero_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert cost_usd("openai", "gpt-unknown-test", 1000, 1000) == 0.0
        assert cost_usd("openai", "gpt-unknown-test", 1000, 1000) == 0.0
    assert caplog.text.count("no price for model 'gpt-unknown-test'") == 1


# --- default factory (builds real provider objects, no network) --------------------------------


def test_default_factory_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    model = default_factory(ModelSpec(provider="anthropic", model="claude-sonnet-5-5"))

    assert isinstance(model, ChatAnthropic)
    assert model.max_tokens == 64000
    assert model.streaming is True
    assert model.temperature is None


def test_default_factory_ollama_uses_num_predict() -> None:
    model = default_factory(ModelSpec(provider="ollama", model="qwen2.5-coder", max_tokens=4096))

    assert isinstance(model, ChatOllama)
    assert model.num_predict == 4096


# --- scripted model ---------------------------------------------------------------------------


def test_scripted_model_plain_answers_carry_usage() -> None:
    model = ScriptedChatModel(script=["hello"], usage=(7, 3))

    answer = model.invoke(PROMPT)

    assert isinstance(answer, AIMessage)
    assert answer.content == "hello"
    assert answer.usage_metadata is not None
    assert (answer.usage_metadata["input_tokens"], answer.usage_metadata["output_tokens"]) == (7, 3)


def test_scripted_model_structured_without_raw() -> None:
    assert ScriptedChatModel(script=[IDEA]).with_structured_output(Idea).invoke(PROMPT) == IDEA


def test_scripted_model_empty_script() -> None:
    with pytest.raises(IndexError, match="no answer left"):
        ScriptedChatModel().invoke(PROMPT)


# --- live smoke test (costs money; run with: pytest -m live) ----------------------------------

needs_key: Callable[[Callable[[], None]], Callable[[], None]] = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


@pytest.mark.live
@needs_key
def test_live_anthropic_structured_output() -> None:
    config = LLMConfig(provider="anthropic", model="claude-haiku-5-5", max_tokens=2000)
    registry = ModelRegistry(config, budget_usd=0.10)

    idea = registry.structured(Idea).invoke(PROMPT)

    assert idea.title
    assert registry.spent_usd > 0


# --- retry policy and OpenAI prices ------------------------------------------------------------


class _HTTPError(Exception):
    """Mimics a provider SDK error, which carries the HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_errors_are_not_retried(status: int) -> None:
    main = ScriptedChatModel(script=[_HTTPError(status), IDEA])

    with pytest.raises(_HTTPError):
        _registry(main).structured(Idea).invoke(PROMPT)
    assert len(main.calls) == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503, 529])
def test_temporary_http_errors_are_retried(status: int) -> None:
    main = ScriptedChatModel(script=[_HTTPError(status), IDEA])

    assert _registry(main).structured(Idea).invoke(PROMPT) == IDEA
    assert len(main.calls) == 2


def test_permanent_error_still_switches_to_the_fallback() -> None:
    main = ScriptedChatModel(script=[_HTTPError(404)])
    backup = ScriptedChatModel(script=[IDEA])

    assert _registry(main, backup).structured(Idea).invoke(PROMPT) == IDEA
    assert len(main.calls) == 1


def test_openai_models_are_priced() -> None:
    assert cost_usd("openai", "gpt-5.4-nano", 1_000_000, 1_000_000) == pytest.approx(1.45)
    assert cost_usd("openai", "gpt-6.1-sol", 148, 105) == pytest.approx(0.001346)
