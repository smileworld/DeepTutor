"""L2 — per-book LLM selection scope.

The invariant under test: a book may pin a model without changing the global
active LLM, without leaking that model into another book's compilation, and
without changing the behaviour of any pre-L2 manifest (which has no field).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from deeptutor.api.routers import book as book_router
from deeptutor.book.engine import BookEngine
from deeptutor.book.model_scope import (
    book_llm_scope,
    normalize_model_selection,
)
from deeptutor.book.models import Book, BookStatus, Page
from deeptutor.services.llm.config import _SCOPED_LLM_CONFIG, set_scoped_llm_config


def _install_fake_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make activation observable without touching the real model catalog."""
    from deeptutor.services.model_selection import runtime

    def fake_activate(selection):
        marker = (selection or {}).get("model_id", "unset")
        token = set_scoped_llm_config(f"book-model:{marker}")
        return None, token

    monkeypatch.setattr(runtime, "activate_llm_selection", fake_activate)


# ── Validation / persistence ────────────────────────────────────────────


def test_normalize_model_selection_accepts_ids_and_rejects_partial() -> None:
    selection = {"profile_id": "p1", "model_id": "m1", "reasoning_effort": "high"}
    assert normalize_model_selection(selection) == selection
    assert normalize_model_selection(None) is None
    assert normalize_model_selection({}) is None
    with pytest.raises(ValueError):
        normalize_model_selection({"profile_id": "p1"})
    with pytest.raises(ValueError):
        normalize_model_selection({"profile_id": "p1", "model_id": "m1",
                                   "reasoning_effort": "nonsense"})


def test_legacy_manifest_stays_readable_and_new_field_roundtrips() -> None:
    legacy = Book.model_validate(
        {"id": "bk_legacy", "title": "Old book", "status": "ready"}
    )
    assert legacy.model_selection is None

    pinned = Book(id="bk_new", model_selection={"profile_id": "p", "model_id": "m"})
    dumped = pinned.model_dump(mode="json")
    assert dumped["model_selection"] == {"profile_id": "p", "model_id": "m"}
    assert Book.model_validate(dumped).model_selection == pinned.model_selection


def test_create_request_accepts_optional_selection() -> None:
    plain = book_router.CreateBookRequest(user_intent="teach me")
    assert plain.model_selection is None
    pinned = book_router.CreateBookRequest(
        user_intent="teach me", model_selection={"profile_id": "p", "model_id": "m"}
    )
    assert pinned.model_selection["model_id"] == "m"


# ── Scope semantics ─────────────────────────────────────────────────────


def test_legacy_book_does_not_disturb_an_outer_scope() -> None:
    token = set_scoped_llm_config("outer")
    try:
        with book_llm_scope(Book()):
            assert _SCOPED_LLM_CONFIG.get() == "outer"
        assert _SCOPED_LLM_CONFIG.get() == "outer"
    finally:
        _SCOPED_LLM_CONFIG.reset(token)


def test_clear_inherited_reproduces_the_worker_behaviour() -> None:
    token = set_scoped_llm_config("request-scoped")
    try:
        with book_llm_scope(Book(), clear_inherited=True):
            assert _SCOPED_LLM_CONFIG.get() is None
        assert _SCOPED_LLM_CONFIG.get() == "request-scoped"
    finally:
        _SCOPED_LLM_CONFIG.reset(token)


def test_selection_is_installed_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_activation(monkeypatch)
    token = set_scoped_llm_config("outer")
    try:
        book = Book(model_selection={"profile_id": "p", "model_id": "m1"})
        with book_llm_scope(book):
            assert _SCOPED_LLM_CONFIG.get() == "book-model:m1"
        assert _SCOPED_LLM_CONFIG.get() == "outer"
    finally:
        _SCOPED_LLM_CONFIG.reset(token)


def test_unresolvable_selection_falls_back_instead_of_breaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deeptutor.services.model_selection import runtime

    def broken(_selection):
        raise ValueError("selected profile/model was not found")

    monkeypatch.setattr(runtime, "activate_llm_selection", broken)
    token = set_scoped_llm_config("outer")
    try:
        with book_llm_scope(Book(model_selection={"profile_id": "gone", "model_id": "m"})):
            # Force inheritance rather than leaking the caller's model.
            assert _SCOPED_LLM_CONFIG.get() is None
        assert _SCOPED_LLM_CONFIG.get() == "outer"
    finally:
        _SCOPED_LLM_CONFIG.reset(token)


@pytest.mark.asyncio
async def test_two_books_do_not_cross_talk(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_activation(monkeypatch)
    barrier = asyncio.Event()
    seen: dict[str, object] = {}

    async def compile_one(name: str, book: Book) -> None:
        with book_llm_scope(book):
            await barrier.wait()
            await asyncio.sleep(0)
            seen[name] = _SCOPED_LLM_CONFIG.get()

    books = {
        "a": Book(model_selection={"profile_id": "p", "model_id": "ma"}),
        "b": Book(model_selection={"profile_id": "p", "model_id": "mb"}),
    }
    tasks = [asyncio.create_task(compile_one(name, book)) for name, book in books.items()]
    await asyncio.sleep(0)
    barrier.set()
    await asyncio.gather(*tasks)

    assert seen == {"a": "book-model:ma", "b": "book-model:mb"}
    assert _SCOPED_LLM_CONFIG.get() is None, "scopes must not leak back to the caller"


# ── Engine integration ──────────────────────────────────────────────────


def _engine(book: Book) -> BookEngine:
    engine = BookEngine.__new__(BookEngine)
    engine._global_lock = asyncio.Lock()
    engine._runtimes = {}
    engine.storage = SimpleNamespace(load_book=lambda book_id: book)
    return engine


@pytest.mark.asyncio
async def test_worker_and_direct_compile_both_see_the_book_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decorator sits on ``compile_page``, which both paths converge on."""
    _install_fake_activation(monkeypatch)
    book = Book(
        id="bk_pinned",
        status=BookStatus.COMPILING,
        model_selection={"profile_id": "p", "model_id": "mp"},
    )
    engine = _engine(book)
    seen: list[object] = []

    async def fake_compile_page_now(**_kwargs):
        seen.append(_SCOPED_LLM_CONFIG.get())
        return Page(id="pg_1", book_id="bk_pinned")

    engine._compile_page_now = fake_compile_page_now
    await engine.compile_page(book_id="bk_pinned", page_id="pg_1")

    assert seen == ["book-model:mp"], (
        "the child task created inside compile_page must inherit the book scope"
    )
    assert _SCOPED_LLM_CONFIG.get() is None


# ── Update endpoint ─────────────────────────────────────────────────────


class _Storage:
    def __init__(self, book: Book) -> None:
        self.book = book

    def load_book(self, book_id: str) -> Book | None:
        return self.book if book_id == self.book.id else None

    def save_book(self, book: Book) -> None:
        self.book = book.model_copy(deep=True)


def _patch_router(monkeypatch: pytest.MonkeyPatch, storage: _Storage) -> None:
    engine = SimpleNamespace(
        storage=storage, load_book=lambda book_id: storage.load_book(book_id)
    )
    monkeypatch.setattr(
        book_router,
        "_resolve_book_or_404",
        lambda *_args, **_kwargs: SimpleNamespace(engine=engine, is_shared=False),
    )
    monkeypatch.setattr(
        book_router, "_claim_content_mutation", lambda *_args, **_kwargs: 2
    )


@pytest.mark.asyncio
async def test_set_model_selection_endpoint_pins_and_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _Storage(Book(id="bk_sel"))
    _patch_router(monkeypatch, storage)

    response = await book_router.set_book_model_selection(
        book_router.SetBookModelSelectionRequest(
            book_id="bk_sel", model_selection={"profile_id": "p", "model_id": "m"}
        )
    )
    assert response["model_selection"] == {"profile_id": "p", "model_id": "m"}
    assert storage.book.model_selection == {"profile_id": "p", "model_id": "m"}

    cleared = await book_router.set_book_model_selection(
        book_router.SetBookModelSelectionRequest(book_id="bk_sel", model_selection=None)
    )
    assert cleared["model_selection"] is None
    assert storage.book.model_selection is None


@pytest.mark.asyncio
async def test_set_model_selection_rejects_malformed_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    storage = _Storage(Book(id="bk_sel"))
    _patch_router(monkeypatch, storage)

    with pytest.raises(HTTPException) as caught:
        await book_router.set_book_model_selection(
            book_router.SetBookModelSelectionRequest(
                book_id="bk_sel", model_selection={"profile_id": "only-one"}
            )
        )
    assert caught.value.status_code == 400
