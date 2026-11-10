"""Local-hardening tests for the bounded deep-question turn budget.

These are deliberately pure/unit level: no LLM, no database, no running server.
They pin the two invariants that the production incident violated:

* a user cannot ask for more questions than the configured ceiling;
* one turn cannot emit an unbounded number of streaming events.
"""

from __future__ import annotations

import asyncio

import pytest

from deeptutor.core.stream import StreamEventType
from deeptutor.runtime.agentic.turn_budget import (
    DEFAULT_MAX_QUESTIONS,
    TurnBudget,
    clamp_question_count,
    stop_notice,
)
from deeptutor.runtime.stream_bus import StreamBus


@pytest.mark.parametrize(
    ("raw_num", "raw_max", "expected_num", "expected_max"),
    [
        (30, 10, 10, 10),
        (3, 10, 3, 10),
        (0, 10, 1, 10),
        (-5, 10, 1, 10),
        ("not-a-number", 10, 1, 10),
        (None, "bad", 1, DEFAULT_MAX_QUESTIONS),
        (10, 1000, 10, 100),
        (1, 0, 1, 1),
    ],
)
def test_clamp_question_count(raw_num, raw_max, expected_num, expected_max):
    assert clamp_question_count(raw_num, raw_max) == (expected_num, expected_max)


def test_clamp_defaults():
    assert clamp_question_count(30) == (DEFAULT_MAX_QUESTIONS, DEFAULT_MAX_QUESTIONS)


def test_env_is_hard_ceiling(monkeypatch):
    monkeypatch.setenv("DEEPTUTOR_TURN_MAX_QUESTIONS", "4")
    monkeypatch.setenv("DEEPTUTOR_TURN_MAX_EVENTS", "50")
    monkeypatch.setenv("DEEPTUTOR_TURN_MAX_SECONDS", "12")
    budget = TurnBudget.from_env(requested_questions=30)
    assert budget.max_questions == 4
    assert budget.max_events == 50
    assert budget.max_seconds == 12.0


def test_env_smaller_request_stays_smaller(monkeypatch):
    monkeypatch.setenv("DEEPTUTOR_TURN_MAX_QUESTIONS", "10")
    assert TurnBudget.from_env(requested_questions=3).max_questions == 3


def test_check_is_none_while_within_budget():
    budget = TurnBudget(
        max_questions=10,
        max_events=100,
        max_seconds=100.0,
        started_at=0.0,
        events_at_start=10,
    )
    assert budget.check(latest_seq=20, now=5.0) is None
    assert budget.produced_events(20) == 10


def test_check_stops_on_questions_first():
    budget = TurnBudget(max_questions=2, max_events=5, max_seconds=5.0, started_at=0.0)
    budget.questions_done = 2
    assert budget.check(latest_seq=100, now=100.0) == "questions"


def test_check_stops_on_time():
    budget = TurnBudget(max_questions=10, max_events=100, max_seconds=10.0, started_at=0.0)
    assert budget.check(now=9.9) is None
    assert budget.check(now=10.0) == "time"


def test_check_stops_on_events():
    budget = TurnBudget(
        max_questions=10,
        max_events=50,
        max_seconds=1000.0,
        started_at=0.0,
        events_at_start=100,
    )
    assert budget.check(latest_seq=149, now=1.0) is None
    assert budget.check(latest_seq=150, now=1.0) == "events"


def test_check_skips_event_counting_without_seq():
    budget = TurnBudget(max_questions=10, max_events=1, max_seconds=1000.0, started_at=0.0)
    assert budget.check(latest_seq=0, now=1.0) is None


def test_stop_notice_is_not_an_error():
    assert "上限" in stop_notice("time", "zh")
    assert "limit" in stop_notice("questions", "en")
    assert stop_notice("unknown", "en") == stop_notice("events", "en")


def _types(bus: StreamBus) -> list[str]:
    # The bus keeps a bounded in-memory history; reading it avoids having to
    # stand up a subscriber task just to assert what was emitted.
    return [event.type.value for event in bus._history]


def test_thinking_cap_is_opt_in():
    async def scenario() -> StreamBus:
        bus = StreamBus(assign_seq=True)
        for _ in range(10):
            await bus.thinking("chunk")
        return bus

    bus = asyncio.run(scenario())
    assert _types(bus).count(StreamEventType.THINKING.value) == 10
    assert bus.thinking_dropped == 0


def test_thinking_cap_drops_only_thinking():
    async def scenario() -> StreamBus:
        bus = StreamBus(assign_seq=True)
        bus.set_thinking_limit(2)
        for index in range(5):
            await bus.thinking(f"chunk-{index}")
        await bus.content("final answer")
        await bus.progress("budget notice")
        return bus

    bus = asyncio.run(scenario())
    types = _types(bus)
    assert types.count(StreamEventType.THINKING.value) == 2
    assert StreamEventType.CONTENT.value in types
    assert StreamEventType.PROGRESS.value in types
    assert bus.thinking_dropped == 3
    assert bus.latest_seq == 4


def test_thinking_cap_can_be_disabled():
    async def scenario() -> StreamBus:
        bus = StreamBus(assign_seq=True)
        bus.set_thinking_limit(1)
        await bus.thinking("one")
        await bus.thinking("dropped")
        bus.set_thinking_limit(None)
        await bus.thinking("restored")
        return bus

    bus = asyncio.run(scenario())
    assert _types(bus).count(StreamEventType.THINKING.value) == 2
    assert bus.thinking_dropped == 0
