# 05 — Evaluator (fixed rule, no LLM)

Module: `perf_agent/evaluation/evaluator.py` · Milestone M4

## Purpose

Deterministic decision on a trial: is it correct, and is it fast enough to be accepted?

## API

```python
class Verdict(BaseModel):
    correct: bool
    mismatched_combos: list[str]          # combo keys whose output_hash differs
    per_combo_ratio: dict[str, float]     # baseline_median / trial_median (>1 = faster)
    gain_pct: float                       # (geomean(ratios) - 1) * 100
    accepted: bool
    reason: str

def evaluate(baseline: BenchmarkResult, trial: BenchmarkResult, accept_threshold_pct: float) -> Verdict
def cumulative_gain_pct(initial: BenchmarkResult, current: BenchmarkResult) -> float
```

## Rules

1. Both results must cover exactly the same combo keys, else `ValueError`.
2. Correctness: for every combo, `trial.output_hash == baseline.output_hash`. Any mismatch → `correct=False`, `accepted=False`.
3. Ratio per combo = `baseline.median_ns / trial.median_ns`.
4. Gain = geometric mean of ratios, minus 1, in percent.
5. `accepted = correct and gain_pct >= accept_threshold_pct`.
6. Additionally reject if any single combo regresses by more than 20 % (ratio < 0.8) — record it in `reason`.
7. `baseline` here is always the **current base version** (not V1), so each promotion must beat its parent.

## Acceptance criteria

- Identical results → gain 0, not accepted (threshold 5).
- Trial 2× faster everywhere → gain 100 %, accepted.
- Ratios [2.0, 0.5] → gain 0 % (geomean), and rejected by the 20 % regression rule.
- One hash mismatch → `correct=False` regardless of speed.
- Mismatched combo sets raise `ValueError`.
- Pure functions: no I/O, fully unit-tested, 100 % branch coverage.
