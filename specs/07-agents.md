# 07 — Agents: Planner, Coder, Critic (LangChain)

Modules: `perf_agent/agents/{planner,coder,critic}.py`, prompts in `perf_agent/agents/prompts/*.md`
Milestone M7. Each agent depends only on `ModelRegistryPort`.

Each agent is a small LangChain chain: `ChatPromptTemplate | registry.structured(role, Schema)`.
Prompts are Markdown files (loaded with `importlib.resources`) used as the system message template.

## 7.1 Planner — proposes ONE idea

```python
class Planner:
    def __init__(self, registry: ModelRegistryPort): ...
    def propose(self, component_source: str, baseline: BenchmarkResult,
                history: list[Trial], objective: Objective) -> Idea
```

- Chain: `planner_prompt | registry.structured(Role.PLANNER, Idea)` → returns an `Idea` directly
  (LangChain structured output; no manual JSON parsing).
- Context: current base source, per-combo baseline medians (ms), compact history
  (title, status, gain, critic's `next_direction`), objective.
- If the returned title equals (case-insensitive) a past trial title → ask once more with an
  extra instruction "propose something different from: …"; if still duplicate, accept with a warning.
- Call `registry.check_budget()` before every invoke.

## 7.2 Coder — rewrites the component

```python
class CodeAnswer(BaseModel):
    java_source: str          # the COMPLETE new file
    summary: str              # 1–3 sentences: what changed

class Coder:
    def implement(self, component_source: str, idea: Idea, class_name: str) -> CodeAnswer
    def fix(self, current_source: str, log_tail: str, idea: Idea, class_name: str) -> CodeAnswer
```

- Chain: `coder_prompt | registry.structured(Role.CODER, CodeAnswer)`.
- System prompt rules: keep the same package, class name and public method signatures; standard
  library only; no new files; no `System.exit`; no I/O or threads unless the idea requires it.
- If `java_source` is wrapped in a ```java fence, strip it.
- `fix` receives the last 150 lines of the build/test log. The graph calls it at most
  `limits.max_build_fixes` times (spec 09).

## 7.3 Critic — explains and suggests the next step

```python
class Critic:
    def review(self, trial: Trial, verdict: Verdict | None, diff: str) -> Critique
```

- Chain: `critic_prompt | registry.structured(Role.CRITIC, Critique)`.
- Input: idea, status, gain, per-combo ratios, unified diff (truncated to 400 lines), error if any.
- The Critic never decides acceptance.

## Acceptance criteria (with `ScriptedChatModel` from spec 06)

- Planner returns the scripted `Idea`; the recorded prompt contains the history titles and the last `next_direction`.
- Duplicate title → exactly one extra call, prompt mentions the duplicate.
- Coder strips a ```java fence from `java_source`.
- `fix` prompt contains the tail of the log (and not the first lines of a 1000-line log).
- Critic prompt contains the diff.
- Every prompt template formats without missing variables (test with sample inputs).
- `BudgetExhausted` raised by `check_budget` propagates out of each agent (not swallowed).
