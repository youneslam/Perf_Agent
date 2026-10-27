# 06 — LLM models with LangChain (ModelRegistry)

Modules: `perf_agent/llm/{models,budget,pricing,fake}.py` · Milestone M6
Implements `ModelRegistryPort`.

## Purpose

One place that builds the LangChain chat model for each role (Planner, Coder, Critic), with
retries, a fallback model and a shared LLM budget. Agents never import a provider package directly.

## Types

```python
class Role(StrEnum):
    PLANNER = "planner"; CODER = "coder"; CRITIC = "critic"

class BudgetExhausted(Exception): ...
```

## ModelRegistry (`llm/models.py`)

```python
class ModelRegistry:
    def __init__(self, config: LLMConfig, budget_usd: float,
                 factory: Callable[[RoleModel], BaseChatModel] = default_factory): ...
    def for_role(self, role: Role) -> BaseChatModel                       # raw model (with budget callback)
    def structured(self, role: Role, schema: type[BaseModel]) -> Runnable # what agents use
    spent_usd: float
    def check_budget(self) -> None
```

- `default_factory` uses `langchain.chat_models.init_chat_model(model, model_provider=provider,
  temperature=..., max_tokens=..., timeout=...)` with `provider` in `anthropic | openai | ollama`.
  API keys come from the env vars the provider packages already read (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`); Ollama uses `OLLAMA_HOST` / default `http://localhost:11434`.
- Per-role defaults: planner `temperature=0.7, max_tokens=2000`; coder `0.2, 8000`; critic `0.3, 1500`.
  Overridable in the scenario (`llm.<role>.temperature`, `max_tokens`).
- `structured(role, schema)` builds, in this exact order (structured output must be applied to
  the raw model, before the retry/fallback wrappers, which return generic Runnables):

  ```python
  primary = self.for_role(role).with_structured_output(schema).with_retry(
      stop_after_attempt=3, wait_exponential_jitter=True)
  if self.config.fallback:
      fb = self._build(self.config.fallback, role).with_structured_output(schema).with_retry(stop_after_attempt=2)
      return primary.with_fallbacks([fb])
  return primary
  ```
- The registry attaches a single shared `BudgetCallback` to every model (`callbacks=[...]`).

## Budget (`llm/budget.py`, `llm/pricing.py`)

```python
class BudgetCallback(BaseCallbackHandler):
    spent_usd: float
    calls: list[CallRecord]          # role, model, input_tokens, output_tokens, cost_usd, duration_s
    def on_llm_end(self, response: LLMResult, **kwargs) -> None: ...
```

- Reads token counts from `AIMessage.usage_metadata` (`input_tokens`, `output_tokens`) of each generation.
- Price table in `pricing.py`: `{model_name: (usd_per_M_input, usd_per_M_output)}`. Unknown model → 0 with a warning. Ollama → 0.
- `check_budget()` raises `BudgetExhausted` if `spent_usd >= budget_usd`. Agents call it **before** each LLM call.
- Logs one INFO line per call (role, model, tokens, cost). Prompt content only at DEBUG.

## Fake model for tests and `--dry-run` (`llm/fake.py`)

LangChain's built-in fake chat models don't support `with_structured_output`, so write our own:

```python
class ScriptedChatModel(BaseChatModel):
    """Returns scripted outputs. Each item is either a str (plain text answer)
    or a Pydantic object (returned as-is by with_structured_output)."""
    script: list[str | BaseModel]
    calls: list[list[BaseMessage]]           # recorded inputs for assertions
    usage: tuple[int, int] = (100, 50)      # fake token counts → feed the BudgetCallback
    def with_structured_output(self, schema, **kw) -> Runnable: ...
```

- `_generate` pops the next script item, returns it as `AIMessage(content=..., usage_metadata=...)`.
- `with_structured_output` returns a `RunnableLambda` that records the call, pops the next item
  and checks it is an instance of `schema` (or validates a dict into it).
- An item may be an `Exception` instance → raised (to test retries/fallbacks).
- `--dry-run` uses a `DryRunScript` that returns a fixed idea, a rewritten component built from the
  current source (same package and class, with the stub marker bumped to `// SPEED=2`) and a fixed critique.
  On the stub project this produces an accepted `graph_V2`; on a real Java project it only exercises the pipeline.

## Acceptance criteria

- `ModelRegistry` with a fake factory returns a model per role with the configured temperature/max_tokens.
- Budget: budget 0.001, fake usage priced so one call costs 0.002 → after one call, `check_budget()` raises.
- Retry: script `[TimeoutError(), Idea(...)]` → `structured(PLANNER, Idea).invoke(...)` returns the idea.
- Fallback: primary always raising, fallback scripted → fallback's answer is returned and the callback records the fallback model.
- No network in tests. One `@pytest.mark.live` smoke test per real provider, skipped by default.
