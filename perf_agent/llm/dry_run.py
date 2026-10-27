"""Fake model for `perf-agent run --dry-run`: answers every agent without an API key (spec 06).

On the stub project it makes each trial twice as fast (`// SPEED=<n>` doubled), so the whole
pipeline runs and a new version is promoted. On a real Java project it only adds a comment, so
it exercises build, test, benchmark and storage without optimizing anything.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from perf_agent.agents.coder import CodeAnswer
from perf_agent.llm.fake import ScriptedChatModel
from perf_agent.models import Critique, Idea

_SPEED = re.compile(r"//\s*SPEED=(\d+)")
_JAVA_BLOCK = re.compile(r"```java\n(.*?)\n```", re.DOTALL)


# A chat model whose structured answers are computed from the prompt instead of a script.
class DryRunChatModel(ScriptedChatModel):
    model_name: str = "dry-run"
    ideas_given: int = 0

    # Return a runnable that builds the right answer for `schema` (Idea, CodeAnswer, Critique).
    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        def answer(prompt: LanguageModelInput) -> Any:
            parsed = self._answer(schema, _prompt_text(prompt))
            if not include_raw:
                return parsed
            raw = AIMessage(content=parsed.model_dump_json(), response_metadata={})
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(answer)

    # The fixed idea, the source with its speed marker doubled, or the fixed critique.
    def _answer(self, schema: type[BaseModel], prompt: str) -> BaseModel:
        if schema is Idea:
            self.ideas_given += 1
            return Idea(
                title=f"Dry-run idea {self.ideas_given}",
                rationale="Dry run: doubles the stub's SPEED marker, no real optimization.",
                expected_gain_pct=100,
                risk="low",
            )
        if schema is CodeAnswer:
            blocks = _JAVA_BLOCK.findall(prompt)
            source = blocks[-1] if blocks else ""
            return CodeAnswer(java_source=faster(source), summary="Dry run: doubled SPEED.")
        if schema is Critique:
            return Critique(
                explanation="Dry run: the result comes from the stub's SPEED marker.",
                next_direction="Run without --dry-run to get real optimization ideas.",
            )
        raise TypeError(f"the dry-run model cannot answer {schema.__name__}")


# Double the `// SPEED=<n>` marker, or add `// SPEED=2` when there is none.
def faster(source: str) -> str:
    match = _SPEED.search(source)
    if match is None:
        return "// SPEED=2\n" + source + ("" if source.endswith("\n") else "\n")
    return _SPEED.sub(f"// SPEED={int(match.group(1)) * 2}", source, count=1) + (
        "" if source.endswith("\n") else "\n"
    )


# All the text of a prompt (system and task messages).
def _prompt_text(prompt: LanguageModelInput) -> str:
    if isinstance(prompt, PromptValue):
        return "\n".join(str(m.content) for m in prompt.to_messages())
    return str(prompt)
