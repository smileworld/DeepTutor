"""P0 wiring tests for the standalone question / research pipelines.

``AgenticLoopPipeline`` already had a hard context-window guard (P2), but
``QuestionPipeline`` and ``ResearchPipeline`` are separate classes with no-op
``LoopHost.guard_context_window`` hooks. This is what let ``deep_question``
send a 45,862-token request on 2026-11-15. These tests verify that every
request exit in both pipelines now carries the shared guard, and that the
incident's model/window combination is bounded.
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from deeptutor.agents.question import pipeline as question_module
from deeptutor.agents.question.pipeline import QuestionPipeline
from deeptutor.agents.research import pipeline as research_module
from deeptutor.agents.research.pipeline import ResearchPipeline
from deeptutor.runtime.agentic.context_guard import DEFAULT_GUARD_RATIO
from deeptutor.runtime.stream_bus import StreamBus
from deeptutor.services.llm.context_window import (
    DEFAULT_CONTEXT_WINDOW_FALLBACK,
    resolve_effective_context_window,
)
from deeptutor.services.session.context_builder import _count_model_context_tokens, count_tokens

WINDOW = 8192
BUDGET = int(WINDOW * DEFAULT_GUARD_RATIO)
LOGGER = logging.getLogger("test.request_guard.wiring")


def _text(tokens: int) -> str:
    unit = "alpha beta gamma delta "
    per_unit = count_tokens(unit * 100) / 100
    return unit * max(1, round(tokens / per_unit))


def _pipe(cls):
    """Build a pipeline without running its config/registry-heavy constructor."""
    pipe = object.__new__(cls)
    pipe.language = "en"
    pipe.model = "spark-x2.6"
    pipe.llm_config = SimpleNamespace(context_window=WINDOW, max_tokens=None)
    pipe.binding = "openai"
    pipe.reasoning_effort = None
    pipe.usage = None
    pipe._budgets = {
        "answering": {"max_tokens": 1024},
        "block": {"max_tokens": 1024},
        "note": {"max_tokens": 512},
        "quiz_finish": {"max_tokens": 1024},
        "tool_summarizer": {"max_tokens": 512},
    }
    pipe._completion_kwargs = lambda *_args, **_kwargs: {}
    return pipe


def _protocol() -> SimpleNamespace:
    return SimpleNamespace(
        allowed=("FINISH",), final=frozenset({"FINISH"}), tool_label=None
    )


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        (question_module, QuestionPipeline),
        (research_module, ResearchPipeline),
    ],
)
@pytest.mark.asyncio
async def test_labeled_step_installs_a_working_request_guard(monkeypatch, module, cls) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_labeled_step(**kwargs: Any):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "run_labeled_step", fake_run_labeled_step)
    pipe = _pipe(cls)

    await pipe._run_labeled_step(
        client=object(),
        messages=[{"role": "user", "content": "Hi"}],
        tool_schemas=None,
        protocol=_protocol(),
        stream=StreamBus(),
        stage="unit-test",
        iter_meta={},
    )

    guard = captured.get("context_guard")
    assert callable(guard), "every one-shot labeled step must carry the guard"
    oversized = [{"role": "user", "content": _text(BUDGET + 3000)}]
    outcome = guard(oversized, None)
    assert outcome.changed is True
    assert sum(_count_model_context_tokens(m) for m in oversized) <= BUDGET


def test_pipeline_guard_uses_the_configured_window(caplog) -> None:
    pipe = _pipe(QuestionPipeline)
    guard = pipe._request_context_guard()
    oversized = [{"role": "user", "content": _text(BUDGET + 1000)}]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="deeptutor.agents.question.pipeline"):
        outcome = guard(oversized, None)

    assert outcome.window == WINDOW
    assert outcome.budget == BUDGET
    assert outcome.changed is True
    assert sum(_count_model_context_tokens(m) for m in oversized) <= BUDGET
    assert any("deep_question" in record.getMessage() for record in caplog.records)


def test_incident_model_falls_back_to_a_bounded_window() -> None:
    """Reproduce 2026-11-15: spark-x2.6 has no metadata, so an oversized
    deep_question request must still be capped at the 16,384 fallback."""
    assert resolve_effective_context_window(model="spark-x2.6") == DEFAULT_CONTEXT_WINDOW_FALLBACK
    budget = int(DEFAULT_CONTEXT_WINDOW_FALLBACK * DEFAULT_GUARD_RATIO)

    pipe = _pipe(QuestionPipeline)
    pipe.llm_config = SimpleNamespace(context_window=None, max_tokens=None)
    oversized = [
        {"role": "system", "content": "You are a tutor."},
        {"role": "user", "content": _text(20_000)},
        {"role": "user", "content": "Give me the questions now."},
    ]

    outcome = pipe._request_context_guard()(oversized, None)

    assert outcome.window == DEFAULT_CONTEXT_WINDOW_FALLBACK
    assert outcome.budget == budget
    assert outcome.changed is True
    assert outcome.within_budget is True
    assert sum(_count_model_context_tokens(m) for m in oversized) <= budget


def test_every_request_exit_is_wired() -> None:
    """Guard against a future call site silently bypassing the shared guard.

    Each pipeline has: two ``run_agentic_loop`` calls, one
    ``_run_labeled_step`` wrapper, and one direct ``completions.create`` in a
    summariser (all of which must install the guard).
    """
    question_src = inspect.getsource(QuestionPipeline)
    research_src = inspect.getsource(ResearchPipeline)

    for source in (question_src, research_src):
        assert source.count("context_guard=self._request_context_guard()") >= 3
        assert source.count("self._request_context_guard()(guarded_messages") >= 1
