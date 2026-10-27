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
