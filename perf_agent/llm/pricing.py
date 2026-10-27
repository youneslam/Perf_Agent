"""LLM prices used to track the run's budget (spec 06)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# USD per million tokens (input, output). Add a model here to have its cost tracked.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    # OpenAI, standard tier, prompts up to 272K tokens (developers.openai.com/api/docs/pricing,
    # checked 2026-10-27).
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-pro": (30.00, 180.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}

# Models already warned about, so the "unknown price" warning is logged once per model.
_warned: set[str] = set()


# Cost in USD of one call; local Ollama models are free, unknown models cost 0 with a warning.
def cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float:
    table = PRICES_USD_PER_MTOK if prices is None else prices
    if model not in table:
        if provider != "ollama" and model not in _warned:
            _warned.add(model)
            logger.warning("no price for model %r: its calls are counted as $0", model)
        return 0.0
    input_price, output_price = table[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
