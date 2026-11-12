"""Bounded resource budget for one ``deep_question`` turn (local hardening).

Why this exists
---------------
The agentic loop already has ``max_iterations``.  What was missing is a bound on
the *aggregate* cost of a single user request: thinking output is streamed one
event per provider chunk, so a 30-question request across three phases and
repeated protocol repairs was observed producing ~19,000 persisted
``turn_events`` rows, a 110 MB database and ~300 frontend polling requests in
three minutes, after which the frontend Node process died with
``STATUS_STACK_BUFFER_OVERRUN`` (0xC0000409).

This module supplies that bound as a small, dependency-free counter checked
between questions.  It deliberately does not raise: exceeding the budget
degrades gracefully by keeping the questions already produced and returning a
normal result envelope.  A turn that stops early on a declared budget is a
correct outcome; a turn that consumes unbounded resources is not.

Tuning (environment variables, all optional)::

    DEEPTUTOR_TURN_MAX_QUESTIONS       default 10
    DEEPTUTOR_TURN_MAX_EVENTS          default 4000
    DEEPTUTOR_TURN_MAX_SECONDS         default 420
    DEEPTUTOR_TURN_MAX_THINKING_EVENTS default 2000
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any

DEFAULT_MAX_QUESTIONS = 10
DEFAULT_MAX_EVENTS = 4000
DEFAULT_MAX_SECONDS = 420.0
DEFAULT_MAX_THINKING_EVENTS = 2000

_ENV_PREFIX = "DEEPTUTOR_TURN_"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def thinking_event_limit_from_env() -> int:
    """Cap for ``THINKING`` events on a turn-scoped stream bus.

    Shared by every exploring-loop turn (chat and the deep modes) so a single
    request cannot persist tens of thousands of reasoning chunks.  The cap only
    suppresses THINKING events; content, progress, result and error survive.
    """
    return _env_int("MAX_THINKING_EVENTS", DEFAULT_MAX_THINKING_EVENTS)


def clamp_question_count(raw_num: Any, raw_max: Any = 10) -> tuple[int, int]:
    """Return ``(effective_num, effective_max)``.

    The API used to read ``num_questions`` and ``max_questions`` independently,
    so a request asking for 30 questions silently bypassed the configured
    ceiling of 10.  Invalid, missing or negative values fall back to safe
    defaults instead of raising inside the capability.
    """
    try:
        ceiling = int(raw_max)
    except (TypeError, ValueError):
        ceiling = DEFAULT_MAX_QUESTIONS
    ceiling = max(1, min(ceiling, 100))
    try:
        requested = int(raw_num)
    except (TypeError, ValueError):
        requested = 1
    requested = max(1, requested)
    return min(requested, ceiling), ceiling


@dataclass(slots=True)
class TurnBudget:
    """Aggregate cost ceiling for one user turn.

    ``events_at_start`` lets ``latest_seq`` be used as a free event counter even
    when the stream bus was already carrying events from earlier phases.
    """

    max_questions: int = DEFAULT_MAX_QUESTIONS
    max_events: int = DEFAULT_MAX_EVENTS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_thinking_events: int = DEFAULT_MAX_THINKING_EVENTS
    events_at_start: int = 0
    started_at: float = field(default_factory=time.monotonic)
    questions_done: int = 0

    @classmethod
    def from_env(
        cls,
        *,
        requested_questions: int | None = None,
        events_at_start: int = 0,
    ) -> "TurnBudget":
        """Build a budget; the environment value is a hard ceiling.

        A smaller, already-clamped request stays smaller, so users cannot raise
        their own ceiling by passing a larger ``num_questions``.
        """
        hard_ceiling = _env_int("MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS)
        limit = (
            min(max(1, int(requested_questions)), hard_ceiling)
            if requested_questions is not None
            else hard_ceiling
        )
        return cls(
            max_questions=limit,
            max_events=_env_int("MAX_EVENTS", DEFAULT_MAX_EVENTS),
            max_seconds=_env_float("MAX_SECONDS", DEFAULT_MAX_SECONDS),
            max_thinking_events=_env_int(
                "MAX_THINKING_EVENTS", DEFAULT_MAX_THINKING_EVENTS
            ),
            events_at_start=max(0, int(events_at_start or 0)),
        )

    def elapsed(self, *, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.started_at

    def produced_events(self, latest_seq: int) -> int:
        """Events emitted since this budget was installed.

        ``latest_seq`` is zero on buses created without ``assign_seq``; in that
        case event counting is simply skipped rather than guessed.
        """
        if not latest_seq:
            return 0
        return max(0, int(latest_seq) - self.events_at_start)

    def check(self, *, latest_seq: int = 0, now: float | None = None) -> str | None:
        """Return a stop reason, or ``None`` while the turn may continue.

        Called *before* starting the next question, so a stopped turn always
        contains only fully generated questions.
        """
        if self.questions_done >= self.max_questions:
            return "questions"
        if self.elapsed(now=now) >= self.max_seconds:
            return "time"
        produced = self.produced_events(latest_seq)
        if produced and produced >= self.max_events:
            return "events"
        return None


_STOP_NOTICES: dict[str, dict[str, str]] = {
    "zh": {
        "questions": "已达到本轮题目数上限，已生成的题目会正常返回。",
        "time": "本轮生成已达到时长上限，已生成的题目会正常返回。",
        "events": "本轮事件量已达上限，已生成的题目会正常返回。",
    },
    "en": {
        "questions": "Question limit reached for this turn; generated questions are returned.",
        "time": "Time limit reached for this turn; generated questions are returned.",
        "events": "Event budget reached for this turn; generated questions are returned.",
    },
}


def stop_notice(reason: str, language: str = "en") -> str:
    """User-visible explanation for a budget stop (never an error)."""
    lang = "zh" if str(language).lower().startswith("zh") else "en"
    return _STOP_NOTICES[lang].get(reason, _STOP_NOTICES[lang]["events"])


__all__ = [
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_MAX_QUESTIONS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_THINKING_EVENTS",
    "TurnBudget",
    "clamp_question_count",
    "stop_notice",
    "thinking_event_limit_from_env",
]
