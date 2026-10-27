"""ModelRegistry: the single LangChain chat model shared by all agents (spec 06).

Agents never import a provider package: they call `registry.structured(Schema)` and get a
runnable that returns a validated `Schema` object, with retry, optional fallback, budget
tracking and truncation detection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from perf_agent.config.scenario import LLMConfig, ModelSpec
from perf_agent.llm.budget import BudgetCallback, BudgetTracker, CallRecord
from perf_agent.models import BudgetExhausted, LLMTruncated

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ModelFactory = Callable[[ModelSpec], BaseChatModel]

# Seconds before one LLM request is abandoned (long answers can take minutes).
REQUEST_TIMEOUT_S = 600
# Attempts for the main model and for the fallback model.
PRIMARY_ATTEMPTS = 3
FALLBACK_ATTEMPTS = 2


# Build the real LangChain model for a provider; no temperature is ever sent.
def default_factory(spec: ModelSpec) -> BaseChatModel:
    kwargs: dict[str, Any]
    if spec.provider == "ollama":
        # ChatOllama ignores max_tokens: its output limit is called num_predict.
        kwargs = {"num_predict": spec.max_tokens, "client_kwargs": {"timeout": REQUEST_TIMEOUT_S}}
    else:
        kwargs = {"max_tokens": spec.max_tokens, "timeout": REQUEST_TIMEOUT_S, "streaming": True}
    model = init_chat_model(spec.model, model_provider=spec.provider, **kwargs)
    if not isinstance(model, BaseChatModel):
        raise TypeError(f"init_chat_model returned {type(model).__name__}, not a chat model")
    return model


# Owns the run's model (and fallback), its budget, and builds structured-output runnables.
class ModelRegistry:
    def __init__(
        self,
        config: LLMConfig,
        budget_usd: float,
        factory: ModelFactory = default_factory,
        prices: dict[str, tuple[float, float]] | None = None,
        retry_wait: bool = True,
    ) -> None:
        self._config = config
        self._factory = factory
        self._prices = prices
        self._retry_wait = retry_wait
        self._budget = BudgetTracker(budget_usd)
        self._model = self._build(config)
        self._fallback = self._build(config.fallback) if config.fallback else None

    # The raw LangChain model, with budget tracking attached.
    def chat_model(self) -> BaseChatModel:
        return self._model

    # A runnable whose answer is validated into `schema`, with retry and optional fallback.
    def structured(self, schema: type[SchemaT]) -> Runnable[LanguageModelInput, SchemaT]:
        primary = self._structured(self._model, self._config, schema, PRIMARY_ATTEMPTS)
        if self._fallback is None or self._config.fallback is None:
            return primary
        fallback = self._structured(
            self._fallback, self._config.fallback, schema, FALLBACK_ATTEMPTS
        )
        return _with_fallback(primary, fallback)

    # Total LLM cost of the run so far, in USD.
    @property
    def spent_usd(self) -> float:
        return self._budget.spent_usd

    # Every LLM call made so far (model, tokens, cost, duration).
    @property
    def calls(self) -> list[CallRecord]:
        return list(self._budget.calls)

    # Raise BudgetExhausted once the budget is spent; agents call it before each LLM call.
    def check_budget(self) -> None:
        self._budget.check()

    # Create a model through the factory and attach its budget callback.
    def _build(self, spec: ModelSpec) -> BaseChatModel:
        model = self._factory(spec)
        callback = BudgetCallback(self._budget, spec, self._prices)
        existing = model.callbacks if isinstance(model.callbacks, list) else []
        model.callbacks = [*existing, callback]
        logger.debug(
            "built %s model %s (max_tokens=%d)", spec.provider, spec.model, spec.max_tokens
        )
        return model

    # Structured output on one model, then truncation/parse checks, then retry.
    def _structured(
        self, model: BaseChatModel, spec: ModelSpec, schema: type[SchemaT], attempts: int
    ) -> Runnable[LanguageModelInput, SchemaT]:
        # json_schema = the provider's native structured output (no forced tool call, which
        # recent Claude models reject).
        bound = model.with_structured_output(schema, include_raw=True, method="json_schema")

        def unwrap(output: Any) -> SchemaT:
            if not isinstance(output, dict):
                raise TypeError(f"unexpected structured output: {type(output).__name__}")
            raw = output.get("raw")
            if isinstance(raw, BaseMessage) and _was_truncated(raw):
                raise LLMTruncated(
                    f"answer cut off at max_tokens={spec.max_tokens}; raise llm.max_tokens"
                )
            error = output.get("parsing_error")
            if isinstance(error, BaseException):
                raise error
            parsed = output.get("parsed")
            if not isinstance(parsed, schema):
                raise TypeError(f"{spec.model} returned no valid {schema.__name__}")
            return parsed

        checked = cast(Runnable[LanguageModelInput, SchemaT], bound | RunnableLambda(unwrap))
        return checked.with_retry(
            retry_if_exception_type=_is_retryable,
            wait_exponential_jitter=self._retry_wait,
            stop_after_attempt=attempts,
        )


# Retry temporary failures only: not a cut-off answer, a spent budget or a permanent 4xx error.
def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, (LLMTruncated, BudgetExhausted)):
        return False
    # Provider SDK errors carry the HTTP status: other 4xx (bad key, unknown model, invalid
    # request) fail the same way every time.
    status = getattr(error, "status_code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status not in _RETRY_STATUSES)


# HTTP 4xx statuses that are temporary: request timeout, conflict, rate limit.
_RETRY_STATUSES = {408, 409, 429}


# Use `fallback` when `primary` fails, except for truncation and budget errors.
def _with_fallback(
    primary: Runnable[LanguageModelInput, SchemaT],
    fallback: Runnable[LanguageModelInput, SchemaT],
) -> Runnable[LanguageModelInput, SchemaT]:
    def invoke(value: LanguageModelInput, config: RunnableConfig) -> SchemaT:
        try:
            return primary.invoke(value, config)
        except (LLMTruncated, BudgetExhausted):
            raise
        except Exception as exc:  # noqa: BLE001 - any other failure switches to the fallback
            logger.warning(
                "main model failed (%s: %s); using the fallback model", type(exc).__name__, exc
            )
            return fallback.invoke(value, config)

    return RunnableLambda(invoke)


# True when the provider says the answer stopped because it hit the token limit.
def _was_truncated(message: BaseMessage) -> bool:
    meta = message.response_metadata
    return (
        meta.get("stop_reason") == "max_tokens"  # Anthropic
        or meta.get("finish_reason") == "length"  # OpenAI
        or meta.get("done_reason") == "length"  # Ollama
    )
