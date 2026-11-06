"""Negotiate a safe output budget for short generations (L5).

A fixed ``max_tokens=80`` is fine for a plain instruct model but starves
reasoning models: they spend the whole budget on hidden thinking tokens and
return an empty string. That produced a false "Task model returned an empty
response" failure in Settings even though the production title path had a
fallback.

The fix is not to pick a bigger magic number. Both the settings probe and the
production session-title path call :func:`resolve_output_token_budget`, which
gives reasoning models headroom under a hard cap. ``get_token_limit_kwargs``
remains responsible only for mapping the parameter name
(``max_tokens`` vs ``max_completion_tokens``); it does not scale the budget.
"""

from __future__ import annotations

import os

from .capabilities import has_thinking_tags
from .config import uses_max_completion_tokens

DEFAULT_SHORT_OUTPUT_TOKENS = 80
REASONING_SHORT_OUTPUT_TOKENS = 512
MAX_SHORT_OUTPUT_TOKENS = 1024
SHORT_OUTPUT_CAP_ENV = "DEEPTUTOR_SHORT_OUTPUT_TOKEN_CAP"

# Model families that reserve part of the completion budget for reasoning.
# Kept as markers rather than a provider capability lookup: the catalog can
# describe any custom model name, and ``has_thinking_tags`` only covers tags.
_REASONING_MODEL_MARKERS = (
    "reasoner",
    "-reasoning",
    "thinking",
    "qwq",
    "qwen3",
    "qwen-3",
    "deepseek-r1",
    "flashx",
    "glm-z1",
    "gemini-2.5-pro",
    "claude-opus-4",
    "claude-sonnet-4",
    "minimax-m3",
)


def _cap() -> int:
    """Hard cap, optionally tightened by deployment; never widened past 4096."""

    raw = os.getenv(SHORT_OUTPUT_CAP_ENV)
    if raw is None or not raw.strip():
        return MAX_SHORT_OUTPUT_TOKENS
    try:
        requested = int(raw.strip())
    except ValueError:
        return MAX_SHORT_OUTPUT_TOKENS
    if requested <= 0:
        return MAX_SHORT_OUTPUT_TOKENS
    return min(requested, 4096)


def requires_reasoning_headroom(
    model: str | None,
    *,
    binding: str | None = None,
    reasoning_effort: str | None = None,
) -> bool:
    """Whether *model* needs more than the plain short-output budget."""

    effort = str(reasoning_effort or "").strip().lower()
    if effort and effort not in {"none", "off", "minimal", "false"}:
        return True

    name = str(model or "").strip().lower()
    if not name:
        return False
    if uses_max_completion_tokens(name):
        return True
    if any(marker in name for marker in _REASONING_MODEL_MARKERS):
        return True
    if binding and has_thinking_tags(binding, model):
        return True
    return False


def resolve_output_token_budget(
    model: str | None,
    *,
    binding: str | None = None,
    reasoning_effort: str | None = None,
    base: int = DEFAULT_SHORT_OUTPUT_TOKENS,
    reasoning_budget: int = REASONING_SHORT_OUTPUT_TOKENS,
) -> int:
    """Return the output token budget for one short generation.

    Plain models keep *base*. Reasoning models get at least *reasoning_budget*,
    but every result is clamped by the hard cap so a misconfigured deployment
    cannot turn a four-word title into an unbounded request.
    """

    floor = max(1, base)
    cap = min(_cap(), 4096)
    if cap < floor:
        return cap
    if not requires_reasoning_headroom(
        model, binding=binding, reasoning_effort=reasoning_effort
    ):
        return floor
    return min(cap, max(floor, reasoning_budget))


__all__ = [
    "DEFAULT_SHORT_OUTPUT_TOKENS",
    "MAX_SHORT_OUTPUT_TOKENS",
    "REASONING_SHORT_OUTPUT_TOKENS",
    "SHORT_OUTPUT_CAP_ENV",
    "requires_reasoning_headroom",
    "resolve_output_token_budget",
]
