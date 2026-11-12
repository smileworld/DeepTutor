"""Local hardening: a request that exceeds the window must be made to fit."""

from __future__ import annotations

import copy
import dataclasses
import logging
from types import SimpleNamespace

import pytest

from deeptutor.agents.chat.agentic_pipeline import AgenticChatPipeline
from deeptutor.agents.loop.pipeline import CONTEXT_WINDOW_GUARD_RATIO, AgenticLoopPipeline
from deeptutor.runtime.agentic.context_guard import TRIM_PREFIX, hard_trim_to_budget
from deeptutor.runtime.agentic.turn_budget import (
    DEFAULT_MAX_THINKING_EVENTS,
    thinking_event_limit_from_env,
)
from deeptutor.runtime.stream_bus import StreamBus
from deeptutor.services.session.context_builder import _count_model_context_tokens, count_tokens

WINDOW = 8192
BUDGET = int(WINDOW * CONTEXT_WINDOW_GUARD_RATIO)


def _text(tokens: int) -> str:
    unit = "alpha beta gamma delta "
    per_unit = count_tokens(unit * 100) / 100
    return unit * max(1, round(tokens / per_unit))


def _count(message: dict) -> int:
    return _count_model_context_tokens(message)


def _pipeline() -> AgenticLoopPipeline:
    pipeline = AgenticLoopPipeline(language="en")
    pipeline.llm_config = dataclasses.replace(pipeline.llm_config, context_window=WINDOW)
    return pipeline


def _trim(messages: list[dict], *, budget: int = BUDGET):
    counts = [_count(message) for message in messages]
    return hard_trim_to_budget(
        messages,
        count=_count,
        counts=counts,
        total=sum(counts),
        budget=budget,
    )


def test_hard_trim_drops_oldest_dialogue_and_protects_system_and_current_question():
    messages = [
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": _text(2000)},
        {"role": "assistant", "content": _text(2000)},
        {"role": "user", "content": _text(2000)},
        {"role": "assistant", "content": _text(2000)},
        {"role": "user", "content": "What about now?"},
    ]

    report = _trim(messages)

    assert report.within_budget is True
    assert report.dropped_messages >= 1
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "What about now?"}
    assert sum(_count(message) for message in messages) <= BUDGET


def test_hard_trim_drops_a_tool_round_as_one_unit():
    huge = _text(4000)
    messages = [
        {"role": "system", "content": "You are a tutor."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": huge},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": huge},
        {"role": "user", "content": "Now answer."},
    ]

    report = _trim(messages)

    has_call = any(message.get("tool_calls") for message in messages)
    has_result = any(message.get("role") == "tool" for message in messages)
    assert has_call == has_result, "a tool call and its result must be dropped together"
    assert report.dropped_messages >= 2
    assert messages[-1] == {"role": "user", "content": "Now answer."}
    assert report.within_budget is True


def test_hard_trim_truncates_an_oversized_single_message_and_keeps_the_tail():
    content = _text(BUDGET + 3000)
    tail = content[-80:]
    messages = [{"role": "user", "content": content}]

    report = _trim(messages)

    assert len(messages) == 1
    assert messages[0]["content"].startswith(TRIM_PREFIX)
    assert messages[0]["content"].endswith(tail)
    assert report.dropped_chars > 0
    assert report.truncated_messages == 1
    assert report.within_budget is True
    assert _count(messages[0]) <= BUDGET


def test_hard_trim_is_a_noop_when_the_request_already_fits():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}]
    before = copy.deepcopy(messages)

    report = _trim(messages)

    assert messages == before
    assert report.changed is False
    assert report.within_budget is True


@pytest.mark.asyncio
async def test_guard_hard_trims_instead_of_sending_the_request_as_is(caplog):
    pipeline = _pipeline()
    messages = [{"role": "user", "content": _text(BUDGET + 1000)}]
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="deeptutor.agents.loop.pipeline"):
        await pipeline._guard_context_window(messages, StreamBus())

    assert "sending it as-is" not in caplog.text
    assert "hard-trimmed" in caplog.text
    assert pipeline._estimate_messages_tokens(messages) <= BUDGET


def test_thinking_cap_is_opt_in_by_environment(monkeypatch):
    monkeypatch.delenv("DEEPTUTOR_TURN_MAX_THINKING_EVENTS", raising=False)
    assert thinking_event_limit_from_env() == DEFAULT_MAX_THINKING_EVENTS
    monkeypatch.setenv("DEEPTUTOR_TURN_MAX_THINKING_EVENTS", "37")
    assert thinking_event_limit_from_env() == 37
    monkeypatch.setenv("DEEPTUTOR_TURN_MAX_THINKING_EVENTS", "not-a-number")
    assert thinking_event_limit_from_env() == DEFAULT_MAX_THINKING_EVENTS


@pytest.mark.asyncio
async def test_chat_run_installs_the_thinking_cap(monkeypatch):
    from deeptutor.agents.loop import agent_loop as agent_loop_module

    pipeline = AgenticChatPipeline(language="en")

    async def noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(pipeline, "_prepare_deferred_tools", noop)
    monkeypatch.setattr(pipeline, "_prepare_kb_manifests", noop)
    monkeypatch.setattr(pipeline, "_exec_allowed", noop)
    monkeypatch.setattr(pipeline, "_compose_enabled_tools", lambda _context: [])
    monkeypatch.setattr(pipeline, "_can_use_native_tool_calling", lambda: False)
    monkeypatch.setattr(pipeline, "_build_openai_client", lambda: object())

    class FakeLoop:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self) -> dict:
            return {}

    monkeypatch.setattr(agent_loop_module, "AgentLoop", FakeLoop)
    monkeypatch.delenv("DEEPTUTOR_TURN_MAX_THINKING_EVENTS", raising=False)

    context = SimpleNamespace(
        runtime=SimpleNamespace(workspace=object(), turn_id="t", previous_model_turn=None),
        active_capability="chat",
        session_id="s",
        language="en",
    )
    bus = StreamBus(assign_seq=True)

    await pipeline.run(context, bus)

    assert bus._thinking_limit == DEFAULT_MAX_THINKING_EVENTS
