"""L5: one negotiated output budget for the task probe and session titles.

The regression this guards against is a hardcoded ``max_tokens=80`` in one
path only: Settings reported "Task model returned an empty response" while
production quietly fell back to a truncated first message. Both call sites
must go through ``resolve_output_token_budget``.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from deeptutor.services.llm.config import get_token_limit_kwargs
from deeptutor.services.llm.token_budget import (
    DEFAULT_SHORT_OUTPUT_TOKENS,
    REASONING_SHORT_OUTPUT_TOKENS,
    SHORT_OUTPUT_CAP_ENV,
    requires_reasoning_headroom,
    resolve_output_token_budget,
)


# --- pure budget negotiation ------------------------------------------------


def test_plain_model_keeps_the_small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SHORT_OUTPUT_CAP_ENV, raising=False)
    assert resolve_output_token_budget("glm-5.3-flash") == DEFAULT_SHORT_OUTPUT_TOKENS
    assert not requires_reasoning_headroom("glm-5.3-flash")


@pytest.mark.parametrize(
    "model",
    [
        "glm-5.3-flashx",
        "deepseek-reasoner",
        "qwen3-32b",
        "qwq-32b",
        "o3-mini",
        "gpt-5.2",
        "gemini-2.5-pro",
        "claude-sonnet-4-5",
    ],
)
def test_reasoning_models_get_headroom(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.delenv(SHORT_OUTPUT_CAP_ENV, raising=False)
    assert requires_reasoning_headroom(model)
    assert resolve_output_token_budget(model) == REASONING_SHORT_OUTPUT_TOKENS


def test_explicit_reasoning_effort_requests_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SHORT_OUTPUT_CAP_ENV, raising=False)
    assert resolve_output_token_budget("plain-model", reasoning_effort="high") == 512
    assert resolve_output_token_budget("plain-model", reasoning_effort="none") == 80


def test_thinking_tags_capability_requests_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    import deeptutor.services.llm.token_budget as budget_module

    monkeypatch.setattr(budget_module, "has_thinking_tags", lambda binding, model: True)
    assert resolve_output_token_budget("plain-model", binding="deepseek") == 512


def test_hard_cap_clamps_but_never_widens_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SHORT_OUTPUT_CAP_ENV, "200")
    assert resolve_output_token_budget("deepseek-reasoner") == 200

    # A wider cap cannot inflate an already negotiated budget...
    monkeypatch.setenv(SHORT_OUTPUT_CAP_ENV, "99999")
    assert resolve_output_token_budget("deepseek-reasoner") == 512
    # ...but a deliberately oversized request is still clamped to 4096.
    assert resolve_output_token_budget("deepseek-reasoner", reasoning_budget=99999) == 4096


def test_cap_below_base_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SHORT_OUTPUT_CAP_ENV, "40")
    assert resolve_output_token_budget("glm-5.3-flash") == 40


@pytest.mark.parametrize("value", ["", "0", "-1", "not-a-number"])
def test_invalid_cap_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SHORT_OUTPUT_CAP_ENV, value)
    assert resolve_output_token_budget("deepseek-reasoner") == 512


def test_parameter_name_mapping_is_still_separate() -> None:
    """Negotiating the budget must not change the max_tokens/max_completion_tokens split."""

    budget = resolve_output_token_budget("o3-mini")
    assert budget == 512
    assert get_token_limit_kwargs("o3-mini", max_tokens=budget) == {
        "max_completion_tokens": 512
    }
    assert get_token_limit_kwargs("glm-5.3-flash", max_tokens=80) == {"max_tokens": 80}


def test_unknown_model_is_safe() -> None:
    assert resolve_output_token_budget(None) == DEFAULT_SHORT_OUTPUT_TOKENS
    assert resolve_output_token_budget("") == DEFAULT_SHORT_OUTPUT_TOKENS


# --- settings task probe really uses the shared function --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected"),
    [("glm-5.3-flash", 80), ("glm-5.3-flashx", 512)],
)
async def test_task_probe_uses_negotiated_budget(
    monkeypatch: pytest.MonkeyPatch, model: str, expected: int
) -> None:
    import deeptutor.services.config.test_runner as test_runner
    import deeptutor.services.llm.factory as factory
    import deeptutor.services.model_selection.runtime as selection_runtime
    import deeptutor.services.model_selection.tasks as tasks
    from deeptutor.services.config.test_runner import ConfigTestRunner, TestRun

    monkeypatch.delenv(SHORT_OUTPUT_CAP_ENV, raising=False)
    monkeypatch.setattr(tasks, "task_service_configured", lambda catalog: True)
    monkeypatch.setattr(
        test_runner, "resolve_llm_runtime_config", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        selection_runtime,
        "llm_config_from_resolved",
        lambda resolved: SimpleNamespace(
            model=model, binding="zhipu", reasoning_effort=None, base_url="http://local"
        ),
    )

    captured: dict[str, object] = {}

    async def fake_complete(config, **kwargs):
        captured.update(kwargs)
        return "OK"

    monkeypatch.setattr(factory, "complete_with_config", fake_complete)

    run = TestRun(id="probe", service="task")
    await ConfigTestRunner()._test_task(run, {})

    assert captured.get("max_tokens") == expected
    assert run.status != "failed"
    assert any("bounded" in event["message"] for event in run.events)


# --- production session-title path uses the same function -------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_key", "expected"),
    [
        ("glm-5.3-flash", "max_tokens", 80),
        ("o3-mini", "max_completion_tokens", 512),
    ],
)
async def test_session_title_uses_negotiated_budget(
    monkeypatch: pytest.MonkeyPatch, model: str, expected_key: str, expected: int
) -> None:
    import deeptutor.services.llm as llm_module
    import deeptutor.services.model_selection.tasks as tasks
    from deeptutor.services.session.turns.title_service import SessionTitleService

    monkeypatch.delenv(SHORT_OUTPUT_CAP_ENV, raising=False)
    monkeypatch.delenv("DEEPTUTOR_CHAT_ENGINE", raising=False)
    monkeypatch.setattr(tasks, "task_llm_scope", lambda kind: nullcontext())
    monkeypatch.setattr(
        llm_module,
        "get_llm_config",
        lambda: SimpleNamespace(model=model, binding="custom", reasoning_effort=None),
    )

    captured: dict[str, object] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        yield "A Short Title"

    monkeypatch.setattr(llm_module, "stream", fake_stream)

    class FakeStore:
        title: str | None = None

        async def get_session(self, session_id: str) -> dict[str, str]:
            return {"title": "New conversation"}

        async def get_messages(self, session_id: str) -> list[dict[str, str]]:
            return [
                {"role": "user", "content": "Explain photosynthesis"},
                {"role": "assistant", "content": "Plants convert light to energy."},
            ]

        async def update_session_title(self, session_id: str, title: str) -> None:
            self.title = title

    store = FakeStore()
    service = SessionTitleService.__new__(SessionTitleService)
    service.store = store

    async def publish(execution: object, event: object) -> dict[str, object]:
        return {}

    service._publish_live_event = publish

    await service._maybe_generate_session_title(
        execution=None, session_id="s1", ui_language="en"
    )

    assert captured.get(expected_key) == expected
    assert store.title == "A Short Title"


def test_no_hardcoded_80_left_in_the_two_call_sites() -> None:
    """Guard against one path silently drifting back to a magic constant."""

    root = Path(__file__).resolve().parents[3]
    for relative in (
        "deeptutor/services/config/test_runner.py",
        "deeptutor/services/session/turns/title_service.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert "resolve_output_token_budget" in source, relative
        assert "max_tokens=80" not in source, relative
