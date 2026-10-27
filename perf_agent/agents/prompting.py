"""Load the agents' Markdown prompts and build their chat templates (spec 07)."""

from __future__ import annotations

from importlib.resources import files

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


# Read `perf_agent/agents/prompts/<name>.md`.
def load_prompt(name: str) -> str:
    return files("perf_agent.agents").joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")


# A fixed system message (role and rules) followed by a task template with {placeholders}.
def chat_prompt(system: str, task: str) -> ChatPromptTemplate:
    # The system text is passed as a message, not a template, so it is never formatted.
    return ChatPromptTemplate.from_messages(
        [SystemMessage(content=load_prompt(system)), ("human", load_prompt(task))]
    )


# Keep only the last `max_lines` lines of `text` (build logs end with the useful errors).
def tail_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return f"[... {len(lines) - max_lines} earlier lines omitted ...]\n" + "\n".join(
        lines[-max_lines:]
    )


# Keep only the first `max_lines` lines of `text`.
def head_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n[... {len(lines) - max_lines} more lines omitted ...]"
