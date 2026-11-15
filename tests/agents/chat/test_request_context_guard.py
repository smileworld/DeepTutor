"""P0 local hardening: one shared request-level window guard for every capability.

The 2026-11-15 incident was *not* a chat-only problem. ``QuestionPipeline`` and
``ResearchPipeline`` are standalone pipelines whose ``LoopHost`` guard hooks
were no-ops, so a ``deep_question`` turn sent 45,862 prompt tokens against a
16,384-token window. These tests pin the invariants of the shared fix:

* the guard trims an oversized request instead of just reporting it;
* it acts on a per-request copy, so the agentic loop's durable history (which
  later iterations still need) is never mutated;
* ``run_agentic_loop`` forwards the guard into ``run_labeled_step``, which is
  the single exit point every labeled LLM call goes through.
"""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from deeptutor.runtime.agentic.context_guard import (
    DEFAULT_GUARD_RATIO,
    request_guard_callback,
)
from deeptutor.runtime.agentic.labeled_step import run_labeled_step
from deeptutor.runtime.agentic.loop import LabelProtocol, run_agentic_loop
from deeptutor.runtime.stream_bus import StreamBus
from deeptutor.services.session.context_builder import _count_model_context_tokens, count_tokens

WINDOW = 8192
BUDGET = int(WINDOW * DEFAULT_GUARD_RATIO)
LOGGER = logging.getLogger("test.request_guard")


def _text(tokens: int) -> str:
    """Prose that costs about ``tokens`` under the counter the guard uses."""
    unit = "alpha beta gamma delta "
    per_unit = count_tokens(unit * 100) / 100
    return unit * max(1, round(tokens / per_unit))


def _oversized_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": _text(2000)},
        {"role": "assistant", "content": _text(2000)},
        {"role": "user", "content": _text(2000)},
        {"role": "assistant", "content": _text(2000)},
        {"role": "user", "content": "Now answer."},
    ]


def _config(window: int = WINDOW) -> SimpleNamespace:
    return SimpleNamespace(context_window=window, max_tokens=None)


def _guard(window: int = WINDOW):
    return request_guard_callback(
        model="mystery-7b",
        llm_config=_config(window),
        scope="unit-test",
        logger=LOGGER,
    )


class _EmptyStream:
    """A provider stream that ends without yielding a chunk."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self) -> None:
        return None


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return _EmptyStream()


def _client() -> tuple[SimpleNamespace, _Completions]:
    completions = _Completions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


async def _run_step(*, messages, context_guard):
    client, completions = _client()
    await run_labeled_step(
        client=client,
        model="mystery-7b",
        messages=messages,
        completion_kwargs={},
        tool_schemas=None,
        allowed_labels=("FINISH",),
        final_labels=frozenset({"FINISH"}),
        tool_label=None,
        stream=StreamBus(),
        source="unit-test",
        stage="unit-test",
        iter_meta={},
        context_guard=context_guard,
    )
    return completions.calls[0]["messages"]


def test_request_guard_callback_trims_and_logs(caplog) -> None:
    messages = [{"role": "user", "content": _text(BUDGET + 3000)}]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        outcome = _guard()(messages, None)

    assert outcome.changed is True
    assert outcome.hard_trimmed is True
    assert outcome.within_budget is True
    assert sum(_count_model_context_tokens(m) for m in messages) <= BUDGET
    assert any(
        "unit-test" in record.getMessage() and "hard-trimmed" in record.getMessage()
        for record in caplog.records
    )


def test_request_guard_callback_is_a_noop_for_an_in_budget_request(caplog) -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Hi"}]
    before = copy.deepcopy(messages)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        outcome = _guard()(messages, None)

    assert outcome.changed is False
    assert messages == before
    assert not caplog.records


def test_request_guard_respects_the_configured_window() -> None:
    messages = [{"role": "user", "content": _text(4000)}]
    outcome = _guard(window=2048)(messages, None)

    assert outcome.window == 2048
    assert outcome.budget == int(2048 * DEFAULT_GUARD_RATIO)
    assert outcome.changed is True


@pytest.mark.asyncio
async def test_labeled_step_sends_a_trimmed_copy_and_keeps_durable_history() -> None:
    """The core P0 invariant: the provider sees a bounded request, the loop
    keeps its full message history for later iterations."""
    messages = _oversized_messages()
    before = copy.deepcopy(messages)

    sent = await _run_step(messages=messages, context_guard=_guard())

    assert sent is not messages
    assert sum(_count_model_context_tokens(m) for m in sent) <= BUDGET
    assert messages == before, "the durable history must not be trimmed"


@pytest.mark.asyncio
async def test_labeled_step_without_a_guard_keeps_the_exact_same_list() -> None:
    """Capabilities that have not opted in must see no behavioural change."""
    messages = [{"role": "user", "content": "Hi"}]

    sent = await _run_step(messages=messages, context_guard=None)

    assert sent is messages


@pytest.mark.asyncio
async def test_labeled_step_does_not_swap_an_in_budget_request() -> None:
    """An unchanged request keeps the caller's list (no copy on the hot path)."""
    messages = [{"role": "user", "content": "Hi"}]

    sent = await _run_step(messages=messages, context_guard=_guard())

    assert sent is messages


@pytest.mark.asyncio
async def test_agentic_loop_forwards_the_guard_into_the_labeled_step(monkeypatch) -> None:
    """The loop is the other half of the wiring: every round must carry it."""
    from deeptutor.runtime.agentic import loop as loop_module

    captured: dict[str, Any] = {}

    async def fake_step(**kwargs: Any):
        captured.update(kwargs)
        return SimpleNamespace(
            label="FINISH",
            text="done",
            tool_calls=[],
            finish_reason="stop",
            stream_idle_timeout=False,
            reasoning_content="",
            thinking_blocks=(),
            reasoning_only=False,
        )

    monkeypatch.setattr(loop_module, "run_labeled_step", fake_step)

    class Host:
        async def guard_context_window(self, messages) -> None:
            return None

        def build_iteration_trace_meta(self, iteration):
            return ({}, {})

    guard = _guard()
    protocol = LabelProtocol(
        allowed=("FINISH",),
        terminal=frozenset({"FINISH"}),
        intermediate=frozenset(),
        final=frozenset(),  # terminal without body emission: no emit_final needed
        tool_label=None,
    )
    outcome = await run_agentic_loop(
        initial_messages=[{"role": "user", "content": "Hi"}],
        protocol=protocol,
        client=object(),
        model="mystery-7b",
        completion_kwargs={},
        binding=None,
        tool_schemas=None,
        stream=StreamBus(),
        source="unit-test",
        stage="unit-test",
        max_iterations=1,
        host=Host(),
        context_guard=guard,
    )

    assert outcome.completed is True
    assert captured["context_guard"] is guard
