"""Tests for perf_agent/config/scenario.py (spec 01)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from perf_agent.config.scenario import ScenarioError, load_scenario
from tests.conftest import STUB_PROJECT, STUB_SCENARIO, ScenarioData

WriteScenario = Callable[[ScenarioData], Path]


def _set(data: ScenarioData, dotted_key: str, value: Any) -> None:
    """Set `data["a"]["b"] = value` for `dotted_key == "a.b"`."""
    *parents, last = dotted_key.split(".")
    node = data
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


# --- valid scenario ---------------------------------------------------------------------------


def test_stub_scenario_loads() -> None:
    scenario = load_scenario(STUB_SCENARIO)

    assert scenario.name == "stub-speedup"
    assert scenario.component.class_name == "com.example.graph.Component"
    assert scenario.llm.fallback is not None
    assert scenario.llm.fallback.provider == "ollama"


def test_combos_are_graphs_times_queries_in_order() -> None:
    keys = [combo.key for combo in load_scenario(STUB_SCENARIO).combos()]

    assert keys == [
        "small::q1",
        "small::q2",
        "medium::q1",
        "medium::q2",
        "large::q1",
        "large::q2",
    ]


def test_optional_sections_get_defaults(
    scenario_data: ScenarioData, write_scenario: WriteScenario
) -> None:
    for key in ("limits", "storage", "version_prefix"):
        del scenario_data[key]
    del scenario_data["llm"]["max_tokens"]
    del scenario_data["objective"]["accept_threshold_pct"]

    scenario = load_scenario(write_scenario(scenario_data))

    assert scenario.version_prefix == "graph"
    assert scenario.limits.max_trials == 5
    assert scenario.limits.llm_budget_usd == 5.0
    assert scenario.storage.kind == "memory"
    assert scenario.llm.max_tokens == 64000
    assert scenario.objective.accept_threshold_pct == 5


def test_base_dir_is_workspace_slash_base_version() -> None:
    scenario = load_scenario(STUB_SCENARIO)

    assert scenario.base_dir == (STUB_PROJECT / "graph_V1").resolve()


def test_loaded_scenario_is_frozen() -> None:
    scenario = load_scenario(STUB_SCENARIO)

    with pytest.raises(ValidationError):
        scenario.name = "changed"  # type: ignore[misc]


# --- workspace resolution ---------------------------------------------------------------------


def test_relative_workspace_resolves_against_yaml_folder_not_cwd(
    tmp_path: Path, scenario_data: ScenarioData, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_dir = tmp_path / "scenarios"
    yaml_dir.mkdir()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    scenario_data["workspace"] = "./ws"
    path = yaml_dir / "scenario.yaml"
    path.write_text(yaml.safe_dump(scenario_data), encoding="utf-8")
    monkeypatch.chdir(other_dir)

    scenario = load_scenario(path)

    assert scenario.workspace == (yaml_dir / "ws").resolve()


def test_absolute_workspace_is_kept(
    tmp_path: Path, scenario_data: ScenarioData, write_scenario: WriteScenario
) -> None:
    absolute = tmp_path / "abs-workspace"
    scenario_data["workspace"] = str(absolute)

    assert load_scenario(write_scenario(scenario_data)).workspace == absolute


# --- one failing case per validation rule -----------------------------------------------------


@pytest.mark.parametrize(
    ("dotted_key", "value", "expected"),
    [
        ("base_version", "graphV1", "base_version 'graphV1' must look like graph_V<number>"),
        ("base_version", "graph_v1", "must look like graph_V<number>"),
        (
            "component.source_file",
            "/abs/com/example/graph/Component.java",
            "component: source_file must be relative",
        ),
        (
            "component.source_file",
            "src/main/java/com/example/graph/Component.class",
            "component: source_file must end with .java",
        ),
        (
            "component.source_file",
            "src/main/java/com/example/graph/Other.java",
            "must match class_name 'com.example.graph.Component'",
        ),
        ("project.run_cmd", ["python", "run.py", "{graph}"], "project.run_cmd: must contain {query}"),
        (
            "project.run_cmd",
            ["python", "run.py"],
            "project.run_cmd: must contain {graph} and {query}",
        ),
        ("project.build_cmd", [], "project.build_cmd"),
        ("project.build_timeout_s", 0, "project.build_timeout_s: Input should be greater than 0"),
        ("project.test_timeout_s", -1, "project.test_timeout_s: Input should be greater than 0"),
        ("project.run_timeout_s", 0, "project.run_timeout_s: Input should be greater than 0"),
        ("objective.accept_threshold_pct", 40, "objective: accept_threshold_pct must be > 0"),
        ("objective.accept_threshold_pct", 0, "objective: accept_threshold_pct must be > 0"),
        ("benchmark.graphs", [], "benchmark.graphs"),
        ("limits.max_trials", 0, "limits.max_trials: Input should be greater than 0"),
        ("llm.provider", "gemini", "llm.provider: Input should be 'anthropic', 'openai' or 'ollama'"),
        ("llm.max_tokens", 0, "llm.max_tokens: Input should be greater than 0"),
        ("storage.kind", "neo4j", "storage: neo4j_uri is required when kind is 'neo4j'"),
    ],
)
def test_invalid_value_is_rejected_with_field_name(
    scenario_data: ScenarioData,
    write_scenario: WriteScenario,
    dotted_key: str,
    value: Any,
    expected: str,
) -> None:
    _set(scenario_data, dotted_key, value)

    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(write_scenario(scenario_data))

    assert expected in str(excinfo.value)


def test_neo4j_storage_with_uri_is_valid(
    scenario_data: ScenarioData, write_scenario: WriteScenario
) -> None:
    scenario_data["storage"] = {"kind": "neo4j", "neo4j_uri": "bolt://localhost:7687"}

    assert load_scenario(write_scenario(scenario_data)).storage.kind == "neo4j"


@pytest.mark.parametrize("dotted_key", ["oops", "project.oops", "llm.temperature"])
def test_unknown_keys_are_rejected(
    scenario_data: ScenarioData, write_scenario: WriteScenario, dotted_key: str
) -> None:
    _set(scenario_data, dotted_key, 1)

    with pytest.raises(ScenarioError, match=rf"{dotted_key}: Extra inputs are not permitted"):
        load_scenario(write_scenario(scenario_data))


def test_missing_required_section_is_reported(
    scenario_data: ScenarioData, write_scenario: WriteScenario
) -> None:
    del scenario_data["llm"]

    with pytest.raises(ScenarioError, match="llm: Field required"):
        load_scenario(write_scenario(scenario_data))


def test_all_errors_are_reported_at_once(
    scenario_data: ScenarioData, write_scenario: WriteScenario
) -> None:
    _set(scenario_data, "project.run_cmd", ["python", "run.py", "{graph}"])
    _set(scenario_data, "llm.provider", "gemini")

    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(write_scenario(scenario_data))

    assert "project.run_cmd" in str(excinfo.value)
    assert "llm.provider" in str(excinfo.value)


# --- file problems ----------------------------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="cannot read scenario"):
        load_scenario(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ScenarioError, match="invalid YAML"):
        load_scenario(path)


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "just text\n"])
def test_top_level_must_be_a_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ScenarioError, match="must be a YAML mapping"):
        load_scenario(path)
