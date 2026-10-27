# 02 — CLI

Module: `perf_agent/cli.py` (Typer) · Milestones M1 (`validate`) and M8 (`run`)
Entry point in `pyproject.toml`: `perf-agent = "perf_agent.cli:app"`.

## Commands

### `perf-agent validate SCENARIO`
Loads the scenario, checks that `workspace/base_version` exists and contains the component
source file. Prints a short summary (component, combos count, objective). Exit 0 if OK, 2 otherwise.

### `perf-agent run SCENARIO [--dry-run] [--max-trials N] [--resume RUN_ID] [--verbose]`
- Builds all dependencies (composition root lives here, nowhere else):
  `VersionManager`, `JavaAdapter`, `ModelRegistry`, `Planner`/`Coder`/`Critic`,
  repository (memory or Neo4j), then the LangGraph graph (`build_graph`, spec 09).
- Loads `.env` at startup (`python-dotenv`) so API keys can live in a local `.env` file.
- `--dry-run`: uses `ScriptedChatModel` with the dry-run script for all roles (spec 06); no API key needed.
- `--max-trials` overrides `limits.max_trials`.
- `--resume RUN_ID`: uses the SQLite checkpointer and continues that run (spec 09).
- Progress: streams graph updates and prints one line per node (`[trial 2] build_test ✗ (attempt 1/3)`).
- Ctrl-C → sets the cancel event; the graph stops with `StopReason.CANCELLED`
  after cleaning the current trial folder.

### `perf-agent graph`
Prints the Mermaid diagram of the optimization graph (`graph.get_graph().draw_mermaid()`).
- At the end: writes the report (spec 10) to `workspace/reports/<run_id>/` and prints a
  Rich table: version, gain, status per trial, stop reason, LLM cost.
- Exit codes: 0 finished (any stop reason except error), 1 unexpected error, 2 invalid scenario.

### `perf-agent versions WORKSPACE [--prefix graph]`
Lists existing version folders in order (`graph_V1`, `graph_V2`, …) and leftover `*_trial` folders.

## Requirements

- Logging configured here only (`--verbose` → DEBUG). Other modules never configure logging.
- Secrets (API keys, Neo4j password) only from env vars; never printed.

## Acceptance criteria

- `validate` on the stub project exits 0; on a missing component file exits 2 with a clear message.
- `run --dry-run` on the stub project (spec 04) completes end-to-end and creates a report (M8).
- `versions` lists folders sorted numerically (`V10` after `V9`).
- Tests use `typer.testing.CliRunner`.
