"""Coder: rewrites the component to implement an idea, and fixes failed builds (spec 07.2)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict

from perf_agent.agents.prompting import chat_prompt, tail_lines
from perf_agent.models import Idea
from perf_agent.ports import ModelRegistryPort

# How many lines of a failed build/test log are shown to the Coder.
LOG_TAIL_LINES = 150

# A whole answer wrapped in a ```java ... ``` Markdown fence.
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


# The Coder's answer: the complete new Java file and a short summary of the change.
class CodeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    java_source: str
    summary: str


# Asks the LLM to rewrite the class for an idea, or to fix its previous attempt.
class Coder:
    def __init__(self, registry: ModelRegistryPort) -> None:
        self._registry = registry
        structured = registry.structured(CodeAnswer)
        self._implement = chat_prompt("coder_system", "coder_implement") | structured
        self._fix = chat_prompt("coder_system", "coder_fix") | structured

    # First version of the class implementing `idea`.
    def implement(self, component_source: str, idea: Idea, class_name: str) -> CodeAnswer:
        return self._ask(
            self._implement,
            {
                **_class_values(class_name),
                **_idea_values(idea),
                "component_source": component_source,
            },
        )

    # Corrected version after a failed build or test; only the end of `log` is sent.
    def fix(self, current_source: str, log: str, idea: Idea, class_name: str) -> CodeAnswer:
        return self._ask(
            self._fix,
            {
                **_class_values(class_name),
                **_idea_values(idea),
                "current_source": current_source,
                "log_tail": tail_lines(log, LOG_TAIL_LINES),
            },
        )

    # One LLM call after a budget check; a Markdown fence around the code is removed.
    def _ask(
        self, chain: Runnable[dict[str, Any], CodeAnswer], values: dict[str, str]
    ) -> CodeAnswer:
        self._registry.check_budget()
        answer = chain.invoke(values)
        return answer.model_copy(update={"java_source": strip_code_fence(answer.java_source)})


# Remove a ```java ... ``` fence wrapped around the whole source, if any.
def strip_code_fence(source: str) -> str:
    match = _FENCE.match(source)
    return match.group(1) + "\n" if match else source


# Package and simple class name from a fully qualified name (com.example.Bfs -> com.example, Bfs).
def _class_values(class_name: str) -> dict[str, str]:
    package, _, simple_name = class_name.rpartition(".")
    return {"class_name": simple_name, "package": package or "(default package)"}


# The idea's title and rationale for the prompt.
def _idea_values(idea: Idea) -> dict[str, str]:
    return {"idea_title": idea.title, "idea_rationale": idea.rationale}
