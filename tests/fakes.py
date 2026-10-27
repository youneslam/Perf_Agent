"""Fast fakes for the loop tests: no subprocess, no LLM (spec 09).

The project fake reads markers from the real component file in each version folder, like the
stub project does, so the real VersionManager can create, promote and delete real folders.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from perf_agent.adapters.java import BenchmarkError, ComponentContractError, ComponentWriteError
from perf_agent.agents.coder import CodeAnswer
from perf_agent.config.scenario import Objective, Scenario
from perf_agent.evaluation.evaluator import Verdict
from perf_agent.models import (
    BenchmarkResult,
    BudgetExhausted,
    BuildResult,
    Combo,
    Critique,
    Idea,
    Measurement,
    RunSummary,
    TestResult,
    Trial,
)

BASE_TIME_NS = 1_000_000
COMPONENT = "src/main/java/com/example/graph/Component.java"


# The component source with stub markers, e.g. source("SPEED=2", "RESULT=other", "BREAK_BUILD").
def source(*markers: str) -> str:
    lines = [f"// {marker}" for marker in markers]
    return "\n".join(
        [*lines, "package com.example.graph;", "", "public class Component {", "}", ""]
    )


# ProjectAdapterPort without processes: SPEED/RESULT/BREAK_* markers drive the results.
class FakeProject:
    def __init__(self) -> None:
        self.builds: list[str] = []

    def read_component(self, version_dir: Path) -> str:
        return (version_dir / COMPONENT).read_text(encoding="utf-8")

    def write_component(self, version_dir: Path, text: str) -> None:
        if not version_dir.name.endswith("_trial"):
            raise ComponentWriteError(f"not a trial folder: {version_dir}")
        if "RENAMED" in text:
            raise ComponentContractError("the new source must declare public class Component")
        (version_dir / COMPONENT).write_text(text, encoding="utf-8")

    def build(self, version_dir: Path) -> BuildResult:
        self.builds.append(version_dir.name)
        ok = "BREAK_BUILD" not in self.read_component(version_dir)
        return BuildResult(ok=ok, log="" if ok else "error: ';' expected", duration_ns=1)

    def test(self, version_dir: Path) -> TestResult:
        ok = "BREAK_TEST" not in self.read_component(version_dir)
        return TestResult(ok=ok, log="" if ok else "expected 3 but was 4")

    def benchmark(self, version_dir: Path, combos: list[Combo]) -> BenchmarkResult:
        text = self.read_component(version_dir)
        if "CRASH" in text:
            raise BenchmarkError(f"{combos[0].key} failed: NullPointerException")
        speed = float(_marker(text, "SPEED", "1"))
        result = _marker(text, "RESULT", "ok")
        return BenchmarkResult(
            version=version_dir.name,
            measurements=[
                Measurement(
                    combo=combo,
                    time_ns=int(BASE_TIME_NS / speed),
                    output_hash=hashlib.sha256(f"{combo.key}:{result}".encode()).hexdigest(),
                )
                for combo in combos
            ],
        )


# Planner that returns "idea 1", "idea 2", … or raises the exception planned for a trial.
class FakePlanner:
    def __init__(self, errors: dict[int, BaseException] | None = None) -> None:
        self.calls = 0
        self.histories: list[list[Trial]] = []
        self._errors = errors or {}

    def propose(
        self,
        component_source: str,
        baseline: BenchmarkResult,
        history: list[Trial],
        objective: Objective,
    ) -> Idea:
        self.calls += 1
        self.histories.append(history)
        if self.calls in self._errors:
            raise self._errors[self.calls]
        return Idea(title=f"idea {self.calls}", rationale="test")


# Coder that answers implement/fix calls, in order, with scripted sources or exceptions.
class FakeCoder:
    def __init__(self, script: list[str | BaseException]) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, str]] = []  # (method, log or "")

    def implement(self, component_source: str, idea: Idea, class_name: str) -> CodeAnswer:
        self.calls.append(("implement", ""))
        return self._next()

    def fix(self, current_source: str, log: str, idea: Idea, class_name: str) -> CodeAnswer:
        self.calls.append(("fix", log))
        return self._next()

    def _next(self) -> CodeAnswer:
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return CodeAnswer(java_source=item, summary="test")


# Critic that records what it reviewed; `on_review` lets a test act after a trial (e.g. cancel).
class FakeCritic:
    def __init__(self, on_review: Callable[[Trial], None] | None = None) -> None:
        self.reviews: list[tuple[Trial, Verdict | None, str]] = []
        self._on_review = on_review

    def review(self, trial: Trial, verdict: Verdict | None, diff: str) -> Critique:
        self.reviews.append((trial, verdict, diff))
        if self._on_review:
            self._on_review(trial)
        return Critique(explanation=f"reviewed {trial.index}", next_direction="go on")


# ModelRegistryPort with a settable cost; check_budget raises once it reaches the budget.
class FakeRegistry:
    def __init__(self, budget_usd: float = 5.0) -> None:
        self.spent = 0.0
        self.budget_usd = budget_usd

    def chat_model(self) -> Any:
        raise NotImplementedError

    def structured(self, schema: Any) -> Any:
        raise NotImplementedError

    @property
    def spent_usd(self) -> float:
        return self.spent

    def check_budget(self) -> None:
        if self.spent >= self.budget_usd:
            raise BudgetExhausted("budget spent")


# RunRepositoryPort that keeps every call for assertions.
class FakeRepository:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.summary: RunSummary | None = None

    def start_run(self, run_id: str, scenario: Scenario) -> None:
        self.events.append(("start_run", run_id))

    def save_baseline(self, run_id: str, result: BenchmarkResult) -> None:
        self.events.append(("save_baseline", result.version))

    def save_trial(self, run_id: str, trial: Trial, bench: BenchmarkResult | None) -> None:
        self.events.append(("save_trial", trial.index))

    def save_version(self, run_id: str, name: str, parent: str, gain_pct: float) -> None:
        self.events.append(("save_version", (name, parent, round(gain_pct, 1))))

    def finish_run(self, run_id: str, summary: RunSummary) -> None:
        self.events.append(("finish_run", summary.stop_reason.value))
        self.summary = summary

    def history(self, run_id: str) -> list[Trial]:
        return []


# Value of a `// NAME=value` marker in the source.
def _marker(text: str, name: str, default: str) -> str:
    match = re.search(rf"//\s*{name}=(\S+)", text)
    return match.group(1) if match else default
