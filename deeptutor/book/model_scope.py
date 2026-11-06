"""Per-book LLM selection scope (myDeeptutor L2).

A book may pin one configured LLM via ``Book.model_selection``. The reference
is ID-only (``profile_id`` / ``model_id``, optionally ``reasoning_effort``),
exactly like the ``LLMSelection`` a chat session carries. No secret is ever
written into ``manifest.json`` — credentials stay in the server-side catalog
and are resolved at runtime by ``activate_llm_selection``.

Backwards compatibility is deliberate: a pre-L2 manifest has no such field.
In that case compilation must behave *exactly* as it did before, i.e. follow
the live global configuration. We therefore do **not** pin a snapshot of
``get_llm_config()`` into the scope when there is no selection; we simply leave
the context alone (or clear it for the worker, which is what the worker already
did). This matters because a long-lived worker would otherwise keep using a
model the user changed mid-book.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import Token
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def normalize_model_selection(value: Any) -> dict[str, str] | None:
    """Validate and canonicalise an ID-only model reference.

    ``None`` / ``{}`` means "inherit the global LLM". A malformed reference
    raises ``ValueError`` so the API can reject it with 4xx instead of
    persisting a manifest that can never resolve.
    """
    from deeptutor.services.model_selection.llm import LLMSelection

    selection = LLMSelection.from_payload(value)
    return selection.to_dict() if selection is not None else None


def activate_book_selection(selection: Any) -> Token[Any] | None:
    """Install *selection* in the current async context.

    Returns the reset token, or ``None`` when the reference cannot be resolved
    (e.g. its profile/model was deleted from the catalog). A stale reference
    must not make a whole book un-compilable; inheriting the live global config
    is strictly better and is what every pre-L2 book already does.
    """
    from deeptutor.services.model_selection.runtime import activate_llm_selection

    try:
        _config, token = activate_llm_selection(selection)
        return token
    except Exception:
        logger.warning(
            "Book model_selection %r could not be resolved; inheriting global LLM",
            selection,
            exc_info=True,
        )
        return None


@contextmanager
def book_llm_scope(
    book_or_selection: Any, *, clear_inherited: bool = False
) -> Iterator[None]:
    """Install a book's ``model_selection`` for the duration of the block.

    ``book_or_selection`` accepts a ``Book``, a raw selection dict (used by
    ``create_book`` before the manifest exists), or ``None``.

    ``clear_inherited=True`` reproduces the worker's original
    ``set_scoped_llm_config(None)`` behaviour: drop a leaked request scope.
    ``False`` leaves an outer scope untouched, so short-lived request paths do
    not silently erase a caller's model pick when the book has no override.
    """
    from deeptutor.services.llm import config as llm_config_module
    from deeptutor.services.model_selection.runtime import reset_llm_selection

    if isinstance(book_or_selection, dict):
        selection = book_or_selection
    else:
        selection = (
            getattr(book_or_selection, "model_selection", None)
            if book_or_selection is not None
            else None
        )

    if selection:
        token = activate_book_selection(selection)
        if token is None:
            # Unusable selection: force inheritance instead of leaking whatever
            # scoped config the caller happened to have installed.
            token = llm_config_module.set_scoped_llm_config(None)
    elif clear_inherited:
        token = llm_config_module.set_scoped_llm_config(None)
    else:
        token = None

    try:
        yield
    finally:
        if token is not None:
            reset_llm_selection(token)


__all__ = [
    "activate_book_selection",
    "book_llm_scope",
    "normalize_model_selection",
]
