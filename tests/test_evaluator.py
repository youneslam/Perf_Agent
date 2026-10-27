"""Tests for perf_agent/evaluation/evaluator.py (spec 05)."""

from __future__ import annotations

import pytest

from perf_agent.evaluation.evaluator import cumulative_gain_pct, evaluate
from perf_agent.models import BenchmarkResult, Combo, Measurement

KEYS = ["small::q1", "large::q1"]


def _result(version: str, times: list[int], hashes: list[str] | None = None) -> BenchmarkResult:
    hashes = hashes or ["h"] * len(times)
    return BenchmarkResult(
        version=version,
        measurements=[
            Measurement(combo=Combo(graph=g, query=q), time_ns=t, output_hash=h)
            for (g, q), t, h in zip(
                (key.split("::") for key in KEYS[: len(times)]), times, hashes, strict=True
            )
        ],
    )


def test_identical_results_are_not_accepted() -> None:
    verdict = evaluate(_result("V1", [100, 200]), _result("V2_trial", [100, 200]), 5)

    assert verdict.correct is True
    assert verdict.gain_pct == 0
    assert verdict.accepted is False
    assert verdict.reason == "rejected: gain +0.0 % < +5 % threshold"


def test_twice_as_fast_everywhere_is_accepted() -> None:
    verdict = evaluate(_result("V1", [100, 200]), _result("V2_trial", [50, 100]), 5)

    assert verdict.per_combo_ratio == {"small::q1": 2.0, "large::q1": 2.0}
    assert verdict.gain_pct == 100
    assert verdict.accepted is True
    assert verdict.reason == "accepted: gain +100.0 % >= +5 % threshold"


def test_geomean_cancels_out_and_regression_rejects() -> None:
    verdict = evaluate(_result("V1", [100, 100]), _result("V2_trial", [50, 200]), 5)

    assert verdict.per_combo_ratio == {"small::q1": 2.0, "large::q1": 0.5}
    assert verdict.gain_pct == 0
    assert verdict.accepted is False
    assert "large::q1 is 50.0 % slower (limit 20 %)" in verdict.reason


def test_regression_rejects_even_with_a_big_overall_gain() -> None:
    verdict = evaluate(_result("V1", [1000, 100]), _result("V2_trial", [100, 130]), 5)

    assert verdict.gain_pct > 100
    assert verdict.accepted is False
    assert verdict.reason == "rejected: large::q1 is 23.1 % slower (limit 20 %)"


def test_slight_slowdown_on_one_combo_is_allowed() -> None:
    verdict = evaluate(_result("V1", [1000, 100]), _result("V2_trial", [500, 120]), 5)

    assert verdict.per_combo_ratio["large::q1"] == pytest.approx(0.8333, abs=1e-4)
    assert verdict.accepted is True


def test_hash_mismatch_is_incorrect_regardless_of_speed() -> None:
    verdict = evaluate(
        _result("V1", [100, 100], ["a", "b"]), _result("V2_trial", [10, 10], ["a", "X"]), 5
    )

    assert verdict.correct is False
    assert verdict.mismatched_combos == ["large::q1"]
    assert verdict.gain_pct == pytest.approx(900)
    assert verdict.accepted is False
    assert verdict.reason == "rejected: output differs on large::q1"


def test_gain_exactly_at_threshold_is_accepted() -> None:
    verdict = evaluate(_result("V1", [105]), _result("V2_trial", [100]), 5)

    assert verdict.gain_pct == 5
    assert verdict.accepted is True


def test_all_problems_are_listed() -> None:
    verdict = evaluate(
        _result("V1", [100, 100], ["a", "b"]), _result("V2_trial", [100, 200], ["a", "X"]), 5
    )

    assert verdict.reason == (
        "rejected: output differs on large::q1; large::q1 is 50.0 % slower (limit 20 %); "
        "gain -29.3 % < +5 % threshold"
    )


def test_mismatched_combo_sets_raise() -> None:
    with pytest.raises(ValueError, match="only in baseline: \\['large::q1'\\]"):
        evaluate(_result("V1", [100, 100]), _result("V2_trial", [100]), 5)


def test_empty_benchmarks_raise() -> None:
    empty = BenchmarkResult(version="V1", measurements=[])

    with pytest.raises(ValueError, match="no measurements"):
        evaluate(empty, empty, 5)


def test_duplicate_combo_raises() -> None:
    m = Measurement(combo=Combo(graph="small", query="q1"), time_ns=100, output_hash="h")
    duplicated = BenchmarkResult(version="V1", measurements=[m, m])

    with pytest.raises(ValueError, match="measured twice"):
        evaluate(duplicated, duplicated, 5)


def test_non_positive_time_raises() -> None:
    with pytest.raises(ValueError, match="non-positive time_ns"):
        evaluate(_result("V1", [100]), _result("V2_trial", [0]), 5)


def test_cumulative_gain() -> None:
    assert cumulative_gain_pct(_result("V1", [100, 200]), _result("V3", [50, 100])) == 100
    assert cumulative_gain_pct(_result("V1", [100]), _result("V3", [100])) == 0


def test_cumulative_gain_needs_the_same_combos() -> None:
    with pytest.raises(ValueError, match="different combos"):
        cumulative_gain_pct(_result("V1", [100, 200]), _result("V3", [50]))
