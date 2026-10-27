"""A scripted chat model for tests and `--dry-run`: no network, no API key, no cost (spec 06)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap
from pydantic import BaseModel, Field


# Script item: an answer cut off by max_tokens (its message says stop_reason="max_tokens").
@dataclass(frozen=True)
class Truncated:
    content: str = ""


# Chat model that returns its script items in order: str, Pydantic object, Truncated or Exception.
class ScriptedChatModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    usage: tuple[int, int] = (100, 50)
    model_name: str = "scripted"

    # Name LangChain uses for this model type.
    @property
    def _llm_type(self) -> str:
        return "scripted"

    # Record the prompt and answer with the next script item (or raise it if it is an error).
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise IndexError("ScriptedChatModel: the script has no answer left")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        stop_reason = "end_turn"
        if isinstance(item, Truncated):
            content, stop_reason = item.content, "max_tokens"
        elif isinstance(item, BaseModel):
            content = item.model_dump_json()
        else:
            content = str(item)
        input_tokens, output_tokens = self.usage
        message = AIMessage(
            content=content,
            response_metadata={"stop_reason": stop_reason, "model_name": self.model_name},
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    # Like the real providers: parse the answer into `schema`, optionally keeping the raw message.
    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, Any]:
        def parse(message: BaseMessage) -> BaseModel:
            return schema.model_validate_json(str(message.content))

        if not include_raw:
            return self | RunnableLambda(parse)

        def parse_keeping_raw(output: dict[str, Any]) -> dict[str, Any]:
            try:
                return {"raw": output["raw"], "parsed": parse(output["raw"]), "parsing_error": None}
            except ValueError as exc:
                return {"raw": output["raw"], "parsed": None, "parsing_error": exc}

        return RunnableMap(raw=self) | RunnableLambda(parse_keeping_raw)
