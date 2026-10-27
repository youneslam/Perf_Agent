"""This module defines the schema for a scenario file, which describes a performance optimization experiment.
it is reffered to the scenario file gicven to the perf_agent as input. The scenario file is a YAML file that contains the following sections:
- name: the name of the scenario
- workspace: the path to the workspace folder where the code will be checked out and built
- base_version: the version of the code to start from, e.g. graph_V1"""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePath
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from perf_agent.models import Combo

logger = logging.getLogger(__name__)

#No scenario file is valid if it is missing any of the required fields. The ScenarioError exception is raised when the scenario file is missing, unreadable or invalid. The message names the bad field.
class ScenarioError(Exception):
    """The scenario file is missing, unreadable or invalid. The message names the bad field."""

#the base class for all Pydantic models in this module. It forbids extra fields and is frozen (immutable) by default.
class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

#Checks that the source_file is a relative path, ends with .java, and matches the class_name. The class_name must be a fully qualified Java class name (e.g. com.example.MyClass) and the source_file must be the corresponding file path (e.g. com/example/MyClass.java).
class ComponentSpec(_Config):
    source_file: str
    class_name: str

    @model_validator(mode="after")
    def _check_source_file(self) -> Self:
        path = PurePath(self.source_file)
        if path.anchor:
            raise ValueError(f"source_file must be relative, got {self.source_file!r}")
        if path.suffix != ".java":
            raise ValueError(f"source_file must end with .java, got {self.source_file!r}")
        simple_name = self.class_name.rsplit(".", 1)[-1]
        if path.stem != simple_name:
            raise ValueError(
                f"source_file name {path.name!r} must match "
                f"class_name {self.class_name!r} (expected {simple_name}.java)"
            )
        return self


#Specification for the project configuration, including build, test, and run commands.
class ProjectSpec(_Config):
    build_cmd: list[str] = Field(min_length=1)
    test_cmd: list[str] = Field(min_length=1)
    run_cmd: list[str] = Field(min_length=1)
    build_timeout_s: float = Field(default=300, gt=0)
    test_timeout_s: float = Field(default=300, gt=0)
    run_timeout_s: float = Field(default=120, gt=0)
    #check if run_cmd contains the placeholders {graph} and {query}. These placeholders will be replaced with the actual graph and query names when the command is run.
    @field_validator("run_cmd")
    @classmethod
    def _check_placeholders(cls, run_cmd: list[str]) -> list[str]:
        joined = " ".join(run_cmd)
        missing = [p for p in ("{graph}", "{query}") if p not in joined]
        if missing:
            raise ValueError(f"must contain {' and '.join(missing)}")
        return run_cmd

#Holds the list of graphs and queries to benchmark. The graphs and queries must be non-empty lists of strings.
class BenchmarkSpec(_Config):
    graphs: list[str] = Field(min_length=1)
    queries: list[str] = Field(min_length=1)

#Objective for the optimization process, including the target gain and the acceptance threshold. The acceptance threshold must be greater than 0 and less than or equal to the target gain.
class Objective(_Config):
    target_gain_pct: float
    accept_threshold_pct: float = 5

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        if not 0 < self.accept_threshold_pct <= self.target_gain_pct:
            raise ValueError(
                "accept_threshold_pct must be > 0 and <= target_gain_pct "
                f"(got {self.accept_threshold_pct} and {self.target_gain_pct})"
            )
        return self

#Here are the limits for the optimization process, including the maximum number of trials, the maximum number of trials without progress, the maximum number of consecutive errors, the maximum number of build fixes, and the budget for using the LLM. All limits must be positive integers, except for max_build_fixes which can be zero or positive, and llm_budget_usd which must be a positive float.
class Limits(_Config):
    max_trials: int = Field(default=5, gt=0)
    max_no_progress: int = Field(default=2, gt=0)
    max_consecutive_errors: int = Field(default=3, gt=0)
    max_build_fixes: int = Field(default=2, ge=0)
    llm_budget_usd: float = Field(default=5.0, gt=0)

#Specification for the language model configuration, including the provider, model name, and maximum number of tokens.
class ModelSpec(_Config):
    provider: Literal["anthropic", "openai", "ollama"]
    model: str
    max_tokens: int = Field(default=64000, gt=0)

#The configuration for the language model used by the Planner, Coder, and Critic. It includes a single model specification and an optional fallback model specification.
class LLMConfig(ModelSpec):
    """The single model shared by the Planner, Coder and Critic, plus an optional fallback."""

    fallback: ModelSpec | None = None

#Storage configuration for the optimization process, including the kind of storage to use (memory or neo4j) and the connection details for neo4j if applicable. If the kind is neo4j, the neo4j_uri must be provided.
class StorageConfig(_Config):
    kind: Literal["memory", "neo4j"] = "memory"
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password_env: str = "NEO4J_PASSWORD"

    @model_validator(mode="after")
    def _check_neo4j(self) -> Self:
        if self.kind == "neo4j" and not self.neo4j_uri:
            raise ValueError("neo4j_uri is required when kind is 'neo4j'")
        return self

#
class Scenario(_Config):
    name: str
    workspace: Path
    base_version: str
    version_prefix: str = "graph"
    component: ComponentSpec
    project: ProjectSpec
    benchmark: BenchmarkSpec
    objective: Objective
    limits: Limits = Field(default_factory=Limits)
    llm: LLMConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @field_validator("workspace")
    @classmethod
    def _resolve_workspace(cls, workspace: Path, info: ValidationInfo) -> Path:
        base_dir = info.context.get("base_dir") if info.context else None
        if base_dir is not None and not workspace.is_absolute():
            return (Path(base_dir) / workspace).resolve()
        return workspace

    @model_validator(mode="after")
    def _check_base_version(self) -> Self:
        pattern = rf"^{re.escape(self.version_prefix)}_V\d+$"
        if not re.match(pattern, self.base_version):
            raise ValueError(
                f"base_version {self.base_version!r} must look like "
                f"{self.version_prefix}_V<number> (e.g. {self.version_prefix}_V1)"
            )
        return self

    @property
    def base_dir(self) -> Path:
        """Folder of the version the run starts from, e.g. `<workspace>/graph_V1`."""
        return self.workspace / self.base_version

    def combos(self) -> list[Combo]:
        """Every graph x query pair, graphs first, in the order written in the file."""
        return [
            Combo(graph=graph, query=query)
            for graph in self.benchmark.graphs
            for query in self.benchmark.queries
        ]


def load_scenario(path: Path) -> Scenario:
    """Read and validate a scenario file. A relative `workspace` is resolved against the
    folder of the YAML file, not the current directory."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"cannot read scenario {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"scenario {path} must be a YAML mapping at the top level")
    try:
        scenario = Scenario.model_validate(data, context={"base_dir": path.parent})
    except ValidationError as exc:
        raise ScenarioError(_format_errors(path, exc)) from exc
    logger.debug("loaded scenario %r from %s", scenario.name, path)
    return scenario


def _format_errors(path: Path, exc: ValidationError) -> str:
    lines = [f"invalid scenario {path}:"]
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"  - {field}: {message}" if field else f"  - {message}")
    return "\n".join(lines)
