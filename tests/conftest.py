"""Shared fixtures: the stub scenario as a dict, and a helper that writes a scenario YAML."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

STUBS_DIR = Path(__file__).parent / "stubs"
STUB_SCENARIO = STUBS_DIR / "scenario.yaml"
STUB_PROJECT = STUBS_DIR / "project"

ScenarioData = dict[str, Any]


@pytest.fixture
def scenario_data() -> ScenarioData:
    """A fresh, valid scenario dict (the stub scenario); tests may modify it freely."""
    data: ScenarioData = yaml.safe_load(STUB_SCENARIO.read_text(encoding="utf-8"))
    return copy.deepcopy(data)


@pytest.fixture
def write_scenario(tmp_path: Path) -> Callable[[ScenarioData], Path]:
    """Return a function that writes a scenario dict to `<tmp_path>/scenario.yaml`."""

    def write(data: ScenarioData) -> Path:
        path = tmp_path / "scenario.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return write
