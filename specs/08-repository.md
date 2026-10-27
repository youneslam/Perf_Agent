# 08 — Run repository (history)

Modules: `perf_agent/storage/memory_repository.py` (M5), `perf_agent/storage/neo4j_repository.py` (M9)
Implements `RunRepositoryPort`.

## Purpose

Persists the state and history of a run: baseline, trials, versions lineage, final summary.
The loop reads `history()` to feed the Planner.

## 8.1 In-memory repository (M5)

Dict-based implementation, used in tests and when `storage.kind == memory`.
Also dumps everything to `workspace/reports/<run_id>/run.json` on `finish_run` (via the report module or directly).

## 8.2 Neo4j repository (M9)

Graph model:

```
(:Run {run_id, scenario, started_at, finished_at, stop_reason, cumulative_gain_pct, llm_cost_usd})
(:Version {name, created_at, gain_pct})
(:Trial {index, status, gain_pct, idea_title, idea_rationale, risk, build_attempts,
         critique_explanation, next_direction, error, started_at, finished_at})
(:Combo {key, graph, query})

(Run)-[:STARTED_FROM]->(Version)
(Version)-[:DERIVED_FROM {gain_pct}]->(Version)          # V2 -> V1
(Run)-[:HAS_TRIAL]->(Trial)
(Trial)-[:BASED_ON]->(Version)
(Trial)-[:PROMOTED_TO]->(Version)                         # only if accepted
(Trial)-[:MEASURED {median_ns, output_hash}]->(Combo)
(Run)-[:BASELINE {median_ns, output_hash}]->(Combo)
```

- Official `neo4j` Python driver, one session per operation, parameterized Cypher only (no string formatting of values).
- Constraints created at startup (idempotent): unique `Run.run_id`, unique `(Version.name, run scope)` — use a composite key `version_id = run_id + ":" + name`.
- Password read from the env var named in `storage.neo4j_password_env`.
- `history()` returns trials ordered by index, mapped back to `Trial` models.

## Acceptance criteria

- Shared contract test suite (`tests/storage/contract.py`) run against both implementations:
  start → baseline → 3 trials (1 accepted) → finish; `history()` returns 3 trials in order with the right statuses.
- Neo4j tests marked `@pytest.mark.neo4j`, run against a local container (document the `docker run` command in the test module), skipped if unreachable.
- Saving the same trial twice does not duplicate nodes (MERGE).
