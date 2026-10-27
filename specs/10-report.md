# 10 — Final report

Module: `perf_agent/report/report.py` · Milestone M10

## Outputs

Written to `workspace/reports/<run_id>/`:
- `run.json` — `RunSummary.model_dump_json(indent=2)`. **This is the contract with the front end**
  (separate repo): add a top-level `"schema_version": 1` field, and also export the JSON Schema
  with `perf-agent schema > run.schema.json` (`RunSummary.model_json_schema()`) so the front can
  generate its types. Any breaking change to `RunSummary` bumps `schema_version`.
- `report.md` — human-readable report.

## `report.md` content

1. **Header**: scenario name, run id, dates, component, initial → final version, cumulative gain, stop reason, LLM cost.
2. **Version chain**: `graph_V1 → graph_V2 (+18.4 %) → graph_V3 (+9.1 %)`, with the idea that produced each version.
3. **Baseline vs final table**: one row per combo — baseline median, final median, ratio.
4. **Trials table**: #, base version, idea title, status, build attempts, gain, promoted to.
5. **Trial details**: for each trial — idea rationale, verdict reason, critic explanation and next direction, error if any.

Times are displayed in ms with 2 decimals; gains with a sign and 1 decimal.

## API

```python
def write_report(summary: RunSummary, out_dir: Path) -> tuple[Path, Path]
def render_markdown(summary: RunSummary) -> str     # pure, for tests
```

## Acceptance criteria

- Snapshot test of `render_markdown` on a fixed `RunSummary` (3 trials, 1 accepted).
- `run.json` round-trips: `RunSummary.model_validate_json(...)` equals the original.
- A run with zero accepted trials renders "No improvement accepted" and final version == initial.
- `perf-agent schema` prints valid JSON Schema containing `schema_version`, `trials` and `stop_reason`.
