# 01 — Scenario (YAML config)

Module: `perf_agent/config/scenario.py` · Milestone M1

## Purpose

A scenario file describes ONE optimization run: which version folder to start from, which
Java component to optimize, which cases to benchmark, and the objective.

## Example

```yaml
name: bfs-speedup
workspace: ./workspace            # folder containing graph_V1/, graph_V2/, …
base_version: graph_V1            # must exist in workspace
version_prefix: graph             # versions are named <prefix>_V<n>

component:
  source_file: src/main/java/com/example/graph/BfsTraversal.java   # relative to the version folder
  class_name: com.example.graph.BfsTraversal

project:
  build_cmd: ["./build.sh"]       # run inside the version folder
  test_cmd: ["./test.sh"]
  run_cmd: ["java", "-cp", "out", "com.example.Runner", "{graph}", "{query}"]
  build_timeout_s: 300
  test_timeout_s: 300
  run_timeout_s: 120

benchmark:
  graphs: [small, medium, large]  # passed as {graph}
  queries: [q1, q2]               # passed as {query}
  warmup_runs: 3
  measured_runs: 10

objective:
  target_gain_pct: 30             # stop when cumulative gain >= this
  accept_threshold_pct: 5         # a trial is accepted if its gain >= this

limits:
  max_trials: 5
  max_no_progress: 2
  max_consecutive_errors: 3
  max_build_fixes: 2              # Coder retries after a failed build
  llm_budget_usd: 5.0

llm:
  planner: {provider: anthropic, model: claude-opus-5-5}
  coder:   {provider: anthropic, model: claude-sonnet-5-5}
  critic:  {provider: anthropic, model: claude-haiku-5-5}
  fallback: {provider: ollama, model: qwen2.5-coder}   # optional

storage:
  kind: memory                    # memory | neo4j
  neo4j_uri: bolt://localhost:7687
  neo4j_user: neo4j
  neo4j_password_env: NEO4J_PASSWORD   # name of env var, never the password itself
```

## Requirements

- Pydantic models: `Scenario`, `ComponentSpec`, `ProjectSpec`, `BenchmarkSpec`, `Objective`,
  `Limits`, `RoleModel`, `LLMConfig`, `StorageConfig`. `extra="forbid"` everywhere.
- `load_scenario(path: Path) -> Scenario`: reads YAML, resolves `workspace` relative to the YAML file.
- `Scenario.combos() -> list[Combo]`: cartesian product graphs × queries, stable order.
- Validation:
  - `base_version` matches `^<version_prefix>_V\d+$`.
  - `source_file` is relative and ends with `.java`; the file name stem equals the last segment of `class_name`.
  - `run_cmd` contains `{graph}` and `{query}`.
  - `0 < accept_threshold_pct <= target_gain_pct`; `measured_runs >= 3`; all timeouts > 0.
  - `storage.kind == "neo4j"` requires `neo4j_uri`.
- Errors raise `ScenarioError` with a message naming the offending field.

## Acceptance criteria

- Valid example above loads; `combos()` returns 6 combos in order `small::q1, small::q2, medium::q1, …`.
- Each validation rule has one failing-case test.
- Unknown keys are rejected.
- `workspace` given as a relative path resolves relative to the YAML file, not the CWD.
