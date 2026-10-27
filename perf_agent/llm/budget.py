"""Track the tokens and cost of every LLM call against the run's budget (spec 06)."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, ConfigDict

from perf_agent.config.scenario import ModelSpec
from perf_agent.llm.pricing import cost_usd
from perf_agent.models import BudgetExhausted

logger = logging.getLogger(__name__)


# One LLM call: which model, how many tokens, what it cost and how long it took.
class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_s: float


# The run's shared budget: every model's callback records its calls here.
class BudgetTracker:
    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.calls: list[CallRecord] = []

    # Total cost of all calls so far, in USD.
    @property
    def spent_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    # Raise BudgetExhausted once the budget is spent (called before each LLM call).
    def check(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExhausted(
                f"LLM budget exhausted: spent ${self.spent_usd:.4f} of ${self.budget_usd:.2f}"
            )


# LangChain callback attached to one model: measures each call and records it in the tracker.
class BudgetCallback(BaseCallbackHandler):
    def __init__(
        self,
        tracker: BudgetTracker,
        spec: ModelSpec,
        prices: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._tracker = tracker
        self._spec = spec
        self._prices = prices
        self._started: dict[UUID, float] = {}

    # A chat model call starts: remember the time (prompt logged at DEBUG only).
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[run_id] = time.perf_counter()
        logger.debug("LLM prompt (%s): %s", self._spec.model, messages)

    # Same for plain (non-chat) model calls.
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = time.perf_counter()

    # A call finished: read the token usage, price it and record it.
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._started.pop(run_id, time.perf_counter())
        input_tokens, output_tokens = _usage(response)
        cost = cost_usd(
            self._spec.provider, self._spec.model, input_tokens, output_tokens, self._prices
        )
        self._tracker.calls.append(
            CallRecord(
                model=self._spec.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_s=time.perf_counter() - started,
            )
        )
        logger.info(
            "LLM call %s: %d in / %d out tokens, $%.4f (total $%.4f)",
            self._spec.model,
            input_tokens,
            output_tokens,
            cost,
            self._tracker.spent_usd,
        )

    # A call failed: forget its start time.
    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._started.pop(run_id, None)


# Sum the input/output tokens reported in a call's `usage_metadata`.
def _usage(response: LLMResult) -> tuple[int, int]:
    input_tokens = output_tokens = 0
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
                usage = generation.message.usage_metadata
                if usage:
                    input_tokens += usage["input_tokens"]
                    output_tokens += usage["output_tokens"]
    return input_tokens, output_tokens
