"""Tests for perf_agent/adapters/java.py against the stub project (spec 04.2)."""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from perf_agent.adapters.java import (
    BenchmarkError,
    ComponentContractError,
    ComponentWriteError,
    JavaAdapter,
    output_hash,
)
from perf_agent.config.scenario import Scenario
from tests.conftest import STUB_PROJECT, ScenarioData

PY = sys.executable
COMPONENT = "src/main/java/com/example/graph/Component.java"
GOOD_SOURCE = "package com.example.graph;\n\npublic class Component {\n}\n"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A copy of the stub project with a `graph_V2_trial` next to `graph_V1`."""
    shutil.copytree(STUB_PROJECT, tmp_path, dirs_exist_ok=True)
    shutil.copytree(tmp_path / "graph_V1", tmp_path / "graph_V2_trial")
    return tmp_path


@pytest.fixture
def make_adapter(
    scenario_data: ScenarioData, workspace: Path
) -> Callable[..., JavaAdapter]:
    """Build an adapter on the workspace copy, with the stub commands run by this Python."""

    def make(**project: Any) -> JavaAdapter:
        scenario_data["workspace"] = str(workspace)
        scenario_data["project"].update(
            build_cmd=[PY, "build.py"],
            test_cmd=[PY, "test.py"],
            run_cmd=[PY, "run.py", "{graph}", "{query}"],
        )
        scenario_data["project"].update(project)
        return JavaAdapter(Scenario.model_validate(scenario_data))

    return make


@pytest.fixture
def adapter(make_adapter: Callable[..., JavaAdapter]) -> JavaAdapter:
    return make_adapter()


@pytest.fixture
def trial(workspace: Path) -> Path:
    return workspace / "graph_V2_trial"


def _set_markers(trial: Path, *markers: str) -> None:
    """Rewrite the trial component with the given stub markers (e.g. `// SPEED=2`)."""
    text = "\n".join(markers) + "\n" + GOOD_SOURCE
    (trial / COMPONENT).write_text(text, encoding="utf-8")


def _combos(adapter_scenario_data: ScenarioData) -> list[Any]:
    return Scenario.model_validate(adapter_scenario_data).combos()


# --- read / write -----------------------------------------------------------------------------


def test_read_component(adapter: JavaAdapter, workspace: Path) -> None:
    source = adapter.read_component(workspace / "graph_V1")

    assert "public class Component" in source
    assert "// SPEED=1" in source


def test_write_component_into_the_trial(adapter: JavaAdapter, trial: Path) -> None:
    new_source = "// SPEED=2\r\n" + GOOD_SOURCE

    adapter.write_component(trial, new_source)

    assert (trial / COMPONENT).read_bytes() == new_source.encode("utf-8")


def test_write_component_into_a_version_is_refused(
    adapter: JavaAdapter, workspace: Path
) -> None:
    original = (workspace / "graph_V1" / COMPONENT).read_bytes()

    with pytest.raises(ComponentWriteError, match="not a \\*_trial folder"):
        adapter.write_component(workspace / "graph_V1", GOOD_SOURCE)

    assert (workspace / "graph_V1" / COMPONENT).read_bytes() == original


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("package com.example.graph;\npublic class Faster {}\n", "must declare public class Component"),
        ("package com.example.other;\npublic class Component {}\n", "declares package 'com.example.other'"),
        ("public class Component {}\n", "declares package '(none)'"),
        (
            "package com.example.graph;\n// public class Component\npublic class Other {}\n",
            "public types found: Other",
        ),
        ("package com.example.graph;\nclass Component {}\n", "public types found: none"),
    ],
)
def test_write_component_checks_package_and_class_name(
    adapter: JavaAdapter, trial: Path, source: str, message: str
) -> None:
    original = (trial / COMPONENT).read_bytes()

    with pytest.raises(ComponentContractError, match=re.escape(message)):
        adapter.write_component(trial, source)

    assert (trial / COMPONENT).read_bytes() == original


def test_write_component_accepts_modifiers_and_helper_types(
    adapter: JavaAdapter, trial: Path
) -> None:
    source = (
        "/* optimized */\npackage com.example.graph;\n\nimport java.util.ArrayDeque;\n\n"
        "public final class Component {\n  private static final class Node {}\n}\n"
        "class Helper {}\n"
    )

    adapter.write_component(trial, source)

    assert adapter.read_component(trial) == source


# --- build / test -----------------------------------------------------------------------------


def test_build_and_test_pass_on_the_stub(adapter: JavaAdapter, trial: Path) -> None:
    build = adapter.build(trial)
    test = adapter.test(trial)

    assert build.ok is True
    assert "BUILD SUCCESSFUL" in build.log
    assert build.duration_ns > 0
    assert test.ok is True


def test_broken_build(adapter: JavaAdapter, trial: Path) -> None:
    _set_markers(trial, "// BREAK_BUILD")

    build = adapter.build(trial)

    assert build.ok is False
    assert "error: ';' expected" in build.log
    assert "exit code 1" in build.log


def test_failing_tests(adapter: JavaAdapter, trial: Path) -> None:
    _set_markers(trial, "// BREAK_TEST")

    test = adapter.test(trial)

    assert test.ok is False
    assert "expected 3 but was 4" in test.log


def test_build_timeout_is_a_failed_build(
    make_adapter: Callable[..., JavaAdapter], trial: Path
) -> None:
    adapter = make_adapter(build_cmd=[PY, "-c", "import time; time.sleep(5)"], build_timeout_s=0.2)

    build = adapter.build(trial)

    assert build.ok is False
    assert "timed out" in build.log


# --- benchmark --------------------------------------------------------------------------------


def test_benchmark_one_measurement_per_combo(
    adapter: JavaAdapter, trial: Path, scenario_data: ScenarioData
) -> None:
    combos = _combos(scenario_data)

    result = adapter.benchmark(trial, combos)

    assert result.version == "graph_V2_trial"
    assert [m.combo.key for m in result.measurements] == [c.key for c in combos]
    assert {m.time_ns for m in result.measurements} == {1_000_000}


def test_speed_2_halves_the_time_and_result_changes_the_hash(
    adapter: JavaAdapter, workspace: Path, trial: Path, scenario_data: ScenarioData
) -> None:
    combos = _combos(scenario_data)
    baseline = adapter.benchmark(workspace / "graph_V1", combos)

    _set_markers(trial, "// SPEED=2", "// RESULT=ok")
    faster = adapter.benchmark(trial, combos)
    _set_markers(trial, "// SPEED=2", "// RESULT=other")
    wrong = adapter.benchmark(trial, combos)

    for base, fast, bad in zip(
        baseline.measurements, faster.measurements, wrong.measurements, strict=True
    ):
        assert fast.time_ns == base.time_ns // 2
        assert fast.output_hash == base.output_hash
        assert bad.output_hash != base.output_hash


def test_hash_differs_between_combos(
    adapter: JavaAdapter, trial: Path, scenario_data: ScenarioData
) -> None:
    result = adapter.benchmark(trial, _combos(scenario_data))

    assert len({m.output_hash for m in result.measurements}) == len(result.measurements)


def test_crashing_run_raises(
    adapter: JavaAdapter, trial: Path, scenario_data: ScenarioData
) -> None:
    _set_markers(trial, "// CRASH")

    with pytest.raises(BenchmarkError, match="small::q1 failed in graph_V2_trial"):
        adapter.benchmark(trial, _combos(scenario_data))


def test_run_timeout_raises(
    make_adapter: Callable[..., JavaAdapter], trial: Path, scenario_data: ScenarioData
) -> None:
    adapter = make_adapter(
        run_cmd=[PY, "-c", "import time; time.sleep(5)", "{graph}", "{query}"], run_timeout_s=0.2
    )

    with pytest.raises(BenchmarkError, match="timed out"):
        adapter.benchmark(trial, _combos(scenario_data)[:1])


def test_without_time_marker_the_wall_clock_time_is_used(
    make_adapter: Callable[..., JavaAdapter], trial: Path, scenario_data: ScenarioData
) -> None:
    adapter = make_adapter(
        run_cmd=[PY, "-c", "import sys; print(sys.argv[1:])", "{graph}", "{query}"]
    )

    result = adapter.benchmark(trial, _combos(scenario_data)[:1])

    assert result.measurements[0].time_ns > 1_000_000


@pytest.mark.parametrize("line", ["#TIME_NS fast", "#TIME_NS 0", "#TIME_NS -5", "#TIME_NS"])
def test_invalid_time_marker_raises(
    make_adapter: Callable[..., JavaAdapter],
    trial: Path,
    scenario_data: ScenarioData,
    line: str,
) -> None:
    adapter = make_adapter(run_cmd=[PY, "-c", f"print({line!r})", "{graph}", "{query}"])

    with pytest.raises(BenchmarkError, match="#TIME_NS"):
        adapter.benchmark(trial, _combos(scenario_data)[:1])


def test_last_time_marker_wins(
    make_adapter: Callable[..., JavaAdapter], trial: Path, scenario_data: ScenarioData
) -> None:
    code = "print('#TIME_NS 10'); print('#TIME_NS 42')"
    adapter = make_adapter(run_cmd=[PY, "-c", code, "{graph}", "{query}"])

    result = adapter.benchmark(trial, _combos(scenario_data)[:1])

    assert result.measurements[0].time_ns == 42


def test_output_hash_normalization() -> None:
    reference = output_hash("result small q1 ok\nline 2\n")

    assert output_hash("result small q1 ok  \r\nline 2\r\n") == reference
    assert output_hash("result small q1 ok\n#TIME_NS 123\nline 2\n#TIMER x\n") == reference
    assert output_hash("result small q1 ok\nline 3\n") != reference
