"""Everything the loop's nodes use, injected once when the graph is built (spec 09)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

from perf_agent.agents.coder import CodeAnswer
from perf_agent.config.scenario import Objective, Scenario
from perf_agent.evaluation.evaluator import Verdict
from perf_agent.models import BenchmarkResult, Critique, Idea, Trial
from perf_agent.ports import (
    ModelRegistryPort,
    ProjectAdapterPort,
    RunRepositoryPort,
    VersionManagerPort,
)


# What the loop needs from the Planner (the real one is agents.planner.Planner).
class PlannerPort(Protocol):
    def propose(
        self,
        component_source: str,
        baseline: BenchmarkResult,
        history: list[Trial],
        objective: Objective,
    ) -> Idea: ...


# What the loop needs from the Coder (the real one is agents.coder.Coder).
class CoderPort(Protocol):
    def implement(self, component_source: str, idea: Idea, class_name: str) -> CodeAnswer: ...

    def fix(self, current_source: str, log: str, idea: Idea, class_name: str) -> CodeAnswer: ...


# What the loop needs from the Critic (the real one is agents.critic.Critic).
class CriticPort(Protocol):
    def review(self, trial: Trial, verdict: Verdict | None, diff: str) -> Critique: ...


# The scenario plus every collaborator; `cancel` is set by Ctrl-C to stop after the trial.
@dataclass
class Deps:
    scenario: Scenario
    versions: VersionManagerPort
    project: ProjectAdapterPort
    registry: ModelRegistryPort
    planner: PlannerPort
    coder: CoderPort
    critic: CriticPort
    repo: RunRepositoryPort
    cancel: threading.Event = field(default_factory=threading.Event)
