# perf_agent — instructions for Claude Code

You are building `perf_agent`, a Python tool that uses LLMs to iteratively optimize the
performance of ONE Java component, keeping every accepted improvement as a full copy of
the project in its own folder (`graph_V1/`, `graph_V2/`, …). There is no git involved.

This repo is **the agent only** (plus its unit tests with stubs and fakes). The Java project,
its tests/benchmarks and the front end live in another repo: never create them here. The agent
talks to them only through the scenario commands (input) and `run.json` / Neo4j (output).

## How to work on this repo

1. Read `specs/00-overview.md` first. It defines the architecture, the package layout,
   the shared data types and the milestone order.
2. Implement milestones **in order** (M1 → M10). Each spec has an "Acceptance criteria"
   section: a milestone is done only when all its criteria pass under `pytest`.
3. Before starting a milestone, re-read its spec. Do not implement things from later
   specs early, except the interfaces (Protocols) they depend on.
4. After each milestone: run `ruff check .`, `mypy perf_agent`, `pytest -q`. Fix everything.

## Conventions

- Python 3.11+, type hints everywhere, `mypy --strict` clean.
- Pydantic v2 for all config and data models. No raw dicts crossing module boundaries.
- Ports & adapters: core logic (`agent/`, `evaluation/`) depends only on Protocols
  defined in `perf_agent/ports.py`, never on concrete adapters.
- The optimization loop is a **LangGraph** `StateGraph` (spec 09). LLM access goes through
  **LangChain** chat models built by `ModelRegistry` (spec 06); agents use
  `with_structured_output` with Pydantic schemas, never hand-parsed JSON.
- Keep build/test/benchmark, version folders, Evaluator and storage as plain Python behind ports.
  Do not turn them into LLM tools: the LLM only plans, writes code and critiques.
- All subprocesses go through `perf_agent/infra/process.py` (timeouts, captured output).
- No network calls in unit tests. LLM calls use `ScriptedChatModel` (spec 06) in tests.
- If a LangChain/LangGraph API in a spec doesn't match the installed version, check the
  installed package's docs, adapt the code, and note the change in the spec.
- Logging with the stdlib `logging` module, one logger per module. No `print` outside the CLI.
- Paths are `pathlib.Path`. Never `os.chdir`.

## Hard rules

- NEVER modify a promoted version folder (`graph_V<n>/`). Only `*_trial` folders are written to.
- NEVER rename the Java class the LLM rewrites. Same file path, same package, same class name.
- Every external command has a timeout.
