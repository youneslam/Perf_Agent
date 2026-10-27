# 00 — Overview

## Goal

`perf_agent run scenario.yaml` takes a Java project folder (e.g. `graph_V1/`), a component
to optimize (e.g. `BfsTraversal.java`) and a performance objective, then loops:

1. an LLM **Planner** proposes one optimization idea;
2. the project is **copied** to a trial folder and an LLM **Coder** rewrites the component (same file, same class name);
3. the trial is **built and tested** for correctness (outputs identical to the baseline);
4. the trial is **benchmarked** on the same cases as the baseline;
5. a fixed-rule **Evaluator** computes the gain and accepts or rejects it;
6. **Promotion**: accepted → the trial folder becomes `graph_V<n+1>/` (the new base); rejected → deleted;
7. an LLM **Critic** explains the result and suggests the next direction.

The loop stops on: objective reached, no progress for 2 trials, 5 trials done, LLM budget
exhausted, 3 consecutive errors, or cancellation.

Final output: a chain of complete folders `graph_V1 → graph_V2 → …`, each faster than the
previous one, plus a report (gain per version, idea, decision and critique of every trial).

## Scope of this repo

This repo contains **only the agent** (`perf_agent`) and its own unit tests.
The Java project to optimize, its tests and benchmarks, and the front end live in a
**separate repo** and are wired in later. The agent knows them only through:

- **input**: the scenario YAML — `workspace` path (can point into the other repo) and the
  `build_cmd`, `test_cmd`, `run_cmd` commands (spec 01);
- **output**: `workspace/reports/<run_id>/run.json` (stable schema, spec 10) and the Neo4j
  history (spec 08), which the front end reads.

Do NOT create a Java project, Java test classes or a front end in this repo. Agent tests use
small stub commands (shell or Python scripts under `tests/stubs/`) that behave like a build,
a test suite and a benchmark runner.

## Architecture

```
scenario.yaml ─► CLI ─► LangGraph StateGraph (agent/graph.py)
                          │   nodes: prepare → baseline → check_stop → plan → code
                          │          → build_test ⇄ fix → bench → evaluate → promote → critique → check_stop …
                          │
                          ├─ Planner / Coder / Critic ──► LangChain chat models (ModelRegistry)
                          │                                 └─► Claude · OpenAI · Ollama
                          ├─ VersionManager ───────────► disk: graph_V1/, graph_V2/, …
                          ├─ ProjectAdapter (Java) ────► build / test / run inside a version folder
                          ├─ Evaluator (pure function)
                          └─ RunRepository ────────────► Neo4j (or in-memory)
                          ▼
                       Report (Markdown + JSON)
```

## Framework choices (and their limits)

- **LangGraph** runs the optimization loop as a `StateGraph`: each diagram step is a node,
  loops and stop conditions are conditional edges. We get for free: a visible graph
  (`graph.get_graph().draw_mermaid()`), step-by-step streaming for the CLI, and a
  checkpointer to resume an interrupted run.
- **LangChain** (`langchain-core` + provider packages) gives one chat-model interface for
  Claude, OpenAI and Ollama, plus `with_structured_output(PydanticModel)` (no hand-written
  JSON parsing), `.with_retry()` and `.with_fallbacks()`.
- **Not** in LangGraph/LangChain: Java build/test/benchmark, version folders, the Evaluator,
  Neo4j history. These stay plain Python behind ports so they are deterministic and unit-testable.
  The LLM never calls tools by itself: nodes call code, the LLM only plans, writes code and critiques.

Ports & adapters: graph nodes only know the Protocols in `perf_agent/ports.py`.

## Package layout

```
perf_agent/
  __init__.py
  cli.py                     # spec 02
  ports.py                   # all Protocols (this spec)
  models.py                  # shared data models (this spec)
  config/scenario.py         # spec 01
  versions/manager.py        # spec 03
  adapters/java.py           # spec 04
  infra/process.py           # spec 04
  evaluation/evaluator.py    # spec 05
  llm/models.py              # spec 06: ModelRegistry (LangChain chat models per role)
  llm/budget.py              # spec 06: BudgetCallback (cost tracking)
  llm/pricing.py             # spec 06
  llm/fake.py                # spec 06: fake chat model for tests / --dry-run
  agents/{planner,coder,critic}.py, agents/prompts/*.md   # spec 07
  storage/{memory,neo4j}_repository.py                    # spec 08
  agent/state.py             # spec 09: LangGraph state
  agent/nodes.py             # spec 09: one function per node
  agent/graph.py             # spec 09: StateGraph wiring + routing
  report/report.py           # spec 10
tests/
  stubs/project/graph_V1/   # fake project: build.sh, test.sh, run.py + a dummy Component.java (spec 04)
  fakes.py                  # fake VersionManager / ProjectAdapter / Repository for graph tests
```

## Shared models (`perf_agent/models.py`)

```python
class Combo(BaseModel):            # one benchmark case
    graph: str
    query: str
    @property
    def key(self) -> str: return f"{self.graph}::{self.query}"

class Measurement(BaseModel):
    combo: Combo
    times_ns: list[int]            # measured runs (after warmup)
    median_ns: int
    output_hash: str               # sha256 of normalized program output

class BenchmarkResult(BaseModel):
    version: str                   # "graph_V1" or "graph_V2_trial"
    measurements: list[Measurement]

class Idea(BaseModel):
    title: str
    rationale: str
    expected_gain_pct: float | None = None
    risk: Literal["low", "medium", "high"] = "medium"

class Critique(BaseModel):
    explanation: str
    next_direction: str

class TrialStatus(StrEnum):
    ACCEPTED, REJECTED_NO_GAIN, REJECTED_INCORRECT, BUILD_FAILED, ERROR

class Trial(BaseModel):
    index: int                     # 1-based within the run
    base_version: str
    trial_dir: str
    idea: Idea
    status: TrialStatus
    build_attempts: int
    gain_pct: float | None
    promoted_to: str | None        # "graph_V2" if accepted
    critique: Critique | None
    error: str | None
    started_at: datetime
    finished_at: datetime

class StopReason(StrEnum):
    OBJECTIVE_REACHED, NO_PROGRESS, MAX_TRIALS, BUDGET_EXHAUSTED, TOO_MANY_ERRORS, CANCELLED

class RunSummary(BaseModel):
    run_id: str
    scenario_name: str
    initial_version: str
    final_version: str
    baseline: BenchmarkResult
    trials: list[Trial]
    cumulative_gain_pct: float     # final version vs initial version
    stop_reason: StopReason
    llm_cost_usd: float
```

## Protocols (`perf_agent/ports.py`)

```python
class VersionManagerPort(Protocol):
    def check_base(self, name: str) -> Path: ...
    def create_trial(self, base: str) -> Path: ...
    def promote(self, trial_dir: Path) -> str: ...
    def discard(self, trial_dir: Path) -> None: ...

class ProjectAdapterPort(Protocol):
    def read_component(self, version_dir: Path) -> str: ...
    def write_component(self, version_dir: Path, source: str) -> None: ...
    def build(self, version_dir: Path) -> BuildResult: ...
    def test(self, version_dir: Path) -> TestResult: ...
    def benchmark(self, version_dir: Path, combos: list[Combo]) -> BenchmarkResult: ...

class ModelRegistryPort(Protocol):           # spec 06
    def for_role(self, role: Role) -> BaseChatModel: ...   # langchain_core BaseChatModel
    def structured(self, role: Role, schema: type[BaseModel]) -> Runnable: ...
    @property
    def spent_usd(self) -> float: ...
    def check_budget(self) -> None: ...      # raises BudgetExhausted

class RunRepositoryPort(Protocol):
    def start_run(self, run_id: str, scenario: Scenario) -> None: ...
    def save_baseline(self, run_id: str, result: BenchmarkResult) -> None: ...
    def save_trial(self, run_id: str, trial: Trial, bench: BenchmarkResult | None) -> None: ...
    def save_version(self, run_id: str, name: str, parent: str, gain_pct: float) -> None: ...
    def finish_run(self, run_id: str, summary: RunSummary) -> None: ...
    def history(self, run_id: str) -> list[Trial]: ...
```

`BuildResult`, `TestResult` are defined in spec 04, `Role` and `BudgetExhausted` in spec 06.

## Dependencies

Runtime (pin major versions):
`langgraph>=1.0,<2`, `langgraph-checkpoint-sqlite`, `langchain-core>=1.0,<2`, `langchain>=1.0,<2`,
`langchain-anthropic`, `langchain-openai`, `langchain-ollama`,
`typer`, `pydantic>=2`, `pyyaml`, `neo4j`, `rich`.
Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`, `types-PyYAML`.
Project managed with `uv` and `pyproject.toml` (hatchling). See `SETUP.md` for the machine setup.
No JDK is needed to run this repo's tests (they use stub commands). A JDK is only needed for real runs against the Java repo.

## Milestones (implement in this order)

| # | Milestone | Spec |
|---|-----------|------|
| M1 | Project skeleton, models, ports, scenario loading, CLI `validate` | 00, 01, 02 |
| M2 | Version manager | 03 |
| M3 | Process runner + command-based project adapter (tested with stubs) | 04 |
| M4 | Evaluator | 05 |
| M5 | In-memory repository | 08 (memory part) |
| M6 | ModelRegistry (LangChain) + budget callback + fake model | 06 |
| M7 | Planner, Coder, Critic (structured output) | 07 |
| M8 | LangGraph optimization graph + CLI `run` (end-to-end with fake model) | 09, 02 |
| M9 | Neo4j repository | 08 (neo4j part) |
| M10 | Report | 10 |

## Open questions (do not block; use the defaults, keep them configurable)

- Which LLMs are allowed (external vs internal)? → default: configurable per role, Ollama supported.
- Only `.class` files may be provided, not `.java` sources. → the tool requires the component's `.java` source; `check_base` fails clearly if it is missing.
- Minimal gain for accepting an optimization → `objective.accept_threshold_pct`, default 5.
- Running LLM-generated code is a security risk → v1: subprocess with timeouts and a minimal env; container sandbox is out of scope but keep `infra/process.py` the single entry point so it can be added later.
