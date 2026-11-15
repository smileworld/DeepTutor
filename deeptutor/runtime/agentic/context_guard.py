"""Hard context-window trimming for a single model request (local hardening).

Why this exists
---------------
The base guard only replaces old ``tool`` results with a short marker.  A long
exploring chat turn accumulates *dialogue* as well as tool results: on
2026-11-11 one chat turn reached 45,978 prompt tokens against a 16,384 window,
logged twelve ``Context-window guard`` warnings, and was still sent unchanged
(``no tool results are left to trim; sending it as-is``).  The model then
returned reasoning only, with ``finish_reason=length``.

A context window is a physical limit.  A request that does not fit must be
*made* to fit instead of merely being reported.  This module does that on the
transient message list that is about to be sent; the durable conversation
history is never touched.

Safety rules
------------
* ``system`` messages and the final ``user`` message (the current question)
  are protected from deletion.
* An ``assistant`` message carrying ``tool_calls`` and its ``tool`` results are
  dropped *together*.  Providers reject requests with a dangling tool call.
* Orphan ``tool`` messages that cannot be paired are kept.
* Truncation keeps the *tail* of the text: recent content is what the model is
  being asked about.
* Best effort only.  If the window is still exceeded afterwards the report says
  so explicitly; it never claims the request fits when it does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

TRIM_PREFIX = "[earlier text trimmed to fit the model's context window]\n"


@dataclass(slots=True)
class TrimResult:
    """Outcome of one hard-trim pass (all counts refer to the request copy)."""

    dropped_messages: int = 0
    truncated_messages: int = 0
    dropped_chars: int = 0
    final_tokens: int = 0
    budget: int = 0
    within_budget: bool = True

    @property
    def changed(self) -> bool:
        return bool(self.dropped_messages or self.truncated_messages)


@dataclass(slots=True)
class _Chunk:
    """An indivisible group of messages, in document order."""

    indices: tuple[int, ...]
    kind: str  # system | dialogue | tool_round | orphan_tool
    protected: bool


def _chunk(messages: list[dict[str, Any]]) -> list[_Chunk]:
    """Group messages so tool calls and their results are never split."""
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index

    chunks: list[_Chunk] = []
    index, size = 0, len(messages)
    while index < size:
        message = messages[index]
        role = message.get("role")
        if role == "system":
            chunks.append(_Chunk((index,), "system", True))
            index += 1
            continue
        if role == "assistant" and message.get("tool_calls"):
            group = [index]
            cursor = index + 1
            while cursor < size and messages[cursor].get("role") == "tool":
                group.append(cursor)
                cursor += 1
            chunks.append(_Chunk(tuple(group), "tool_round", False))
            index = cursor
            continue
        if role == "tool":
            # Never paired with an assistant tool_call above: dropping it alone
            # would leave a dangling call, so keep it.
            chunks.append(_Chunk((index,), "orphan_tool", True))
            index += 1
            continue
        protected = index == last_user and role == "user"
        chunks.append(_Chunk((index,), "dialogue", protected))
        index += 1
    return chunks


def _shrink_to_tokens(
    message: dict[str, Any],
    count: Callable[[dict[str, Any]], int],
    target_tokens: int,
) -> int:
    """Binary-search the longest tail of ``content`` that fits ``target_tokens``.

    Returns the number of characters removed, or zero when nothing changed.
    """
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return 0
    if count(message) <= target_tokens:
        return 0

    low, high, best = 0, len(content), 0
    while low <= high:
        middle = (low + high) // 2
        message["content"] = (
            TRIM_PREFIX + content[-middle:] if middle else TRIM_PREFIX
        )
        if count(message) <= target_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    message["content"] = TRIM_PREFIX + content[-best:] if best else TRIM_PREFIX
    return len(content) - best


def hard_trim_to_budget(
    messages: list[dict[str, Any]],
    *,
    count: Callable[[dict[str, Any]], int],
    counts: list[int],
    total: int,
    budget: int,
) -> TrimResult:
    """Mutate ``messages`` in place until the request fits ``budget`` tokens.

    ``counts`` must align with ``messages`` and ``total`` must already include
    any fixed overhead (e.g. tool schemas).  Both come from the caller so the
    guard does not have to re-count the whole request twice per model round.
    """
    overhead = max(0, total - sum(counts))
    current = sum(counts) + overhead
    result = TrimResult(
        final_tokens=current, budget=budget, within_budget=current <= budget
    )
    if current <= budget:
        return result

    chunks = _chunk(messages)
    dropped: set[int] = set()

    # Ordinary dialogue is history; whole tool rounds keep the loop's working
    # state, so drop dialogue first and only then sacrifice tool rounds.
    order = [c for c in chunks if not c.protected and c.kind == "dialogue"]
    order += [c for c in chunks if not c.protected and c.kind == "tool_round"]
    for chunk in order:
        if current <= budget:
            break
        for index in chunk.indices:
            if index not in dropped:
                dropped.add(index)
                current -= counts[index]

    kept_indices = [i for i in range(len(messages)) if i not in dropped]
    messages[:] = [messages[i] for i in kept_indices]
    kept_counts = [counts[i] for i in kept_indices]
    result.dropped_messages = len(dropped)

    if current > budget:
        # Non-system messages first, largest first; system prompts are a last
        # resort because cutting them can remove the loop's instructions.
        candidates = sorted(
            range(len(messages)),
            key=lambda i: (messages[i].get("role") == "system", -kept_counts[i]),
        )
        for index in candidates:
            if current <= budget:
                break
            message = messages[index]
            if not isinstance(message.get("content"), str):
                continue
            before = kept_counts[index]
            target = max(0, before - (current - budget))
            removed = _shrink_to_tokens(message, count, target)
            if removed <= 0:
                continue
            after = count(message)
            current -= before - after
            kept_counts[index] = after
            result.truncated_messages += 1
            result.dropped_chars += removed

    result.final_tokens = current
    result.within_budget = current <= budget
    return result


DEFAULT_GUARD_RATIO = 0.9
DEFAULT_TOOL_RESULT_MARKER = (
    "[earlier tool result snipped to stay within context window; "
    "call the same tool again if the content is still needed]"
)


def default_message_token_count(message: dict[str, Any]) -> int:
    """Count one message the same way history planning counts it."""
    from deeptutor.services.session.context_builder import _count_model_context_tokens

    # ``_context_snapshot`` is bookkeeping stripped before the provider sees it.
    sent = {key: value for key, value in message.items() if key != "_context_snapshot"}
    return _count_model_context_tokens(sent)


@dataclass(slots=True)
class GuardOutcome:
    """What one request-level guard pass did, for logging and progress notices."""

    window: int = 0
    budget: int = 0
    before_tokens: int = 0
    snipped: int = 0
    report: TrimResult = field(default_factory=TrimResult)

    @property
    def hard_trimmed(self) -> bool:
        return self.report.changed

    @property
    def changed(self) -> bool:
        return self.snipped > 0 or self.hard_trimmed

    @property
    def final_tokens(self) -> int:
        return self.report.final_tokens

    @property
    def within_budget(self) -> bool:
        return self.report.within_budget


def apply_request_guard(
    messages: list[dict[str, Any]],
    *,
    model: str,
    context_window: Any = None,
    max_tokens: Any = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    ratio: float = DEFAULT_GUARD_RATIO,
    snip_marker: str | None = None,
    count: Callable[[dict[str, Any]], int] | None = None,
) -> GuardOutcome:
    """Make ``messages`` fit the model window, in place, before it is sent.

    This is the single shared implementation used by the chat agent loop and
    by the standalone question / research pipelines.  ``messages`` must be the
    transient request copy — durable conversation history is never mutated.

    Phases: snip old tool results, drop whole historical chunks, then shorten
    oversized text.  Tool schemas are counted as fixed overhead so the *whole*
    request, not just its message body, is bounded.
    """
    from deeptutor.services.llm.context_window import resolve_effective_context_window
    from deeptutor.services.session.context_builder import _count_model_context_tokens

    counter = count or default_message_token_count
    try:
        window = resolve_effective_context_window(
            context_window=context_window, model=model, max_tokens=max_tokens
        )
    except Exception:
        window = 0
    if not window or window <= 0:
        return GuardOutcome(report=TrimResult(within_budget=True))

    budget = max(1, int(window * ratio))
    overhead = _count_model_context_tokens(tool_schemas) if tool_schemas else 0
    counts = [counter(message) for message in messages]
    total = sum(counts) + overhead
    outcome = GuardOutcome(
        window=window,
        budget=budget,
        before_tokens=total,
        report=TrimResult(
            final_tokens=total, budget=budget, within_budget=total <= budget
        ),
    )
    if total <= budget:
        return outcome

    marker = snip_marker or DEFAULT_TOOL_RESULT_MARKER
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or message.get("content") == marker:
            continue
        message["content"] = marker
        outcome.snipped += 1
        after = counter(message)
        total += after - counts[index]
        counts[index] = after
        if total <= budget:
            break

    if total > budget:
        outcome.report = hard_trim_to_budget(
            messages,
            count=counter,
            counts=counts,
            total=total,
            budget=budget,
        )
    else:
        outcome.report = TrimResult(
            final_tokens=total, budget=budget, within_budget=True
        )
    return outcome


def log_request_guard(outcome: GuardOutcome, *, scope: str, logger: Any) -> None:
    """Emit one consistent warning for capabilities without a progress bus."""
    if not outcome.changed:
        return
    report = outcome.report
    logger.warning(
        "Context-window guard [%s]: hard-trimmed %d message(s), shortened %d "
        "(%d chars), snipped %d tool result(s); request is now about %d tokens "
        "against a %d-token budget (window %d, before %d)%s",
        scope,
        report.dropped_messages,
        report.truncated_messages,
        report.dropped_chars,
        outcome.snipped,
        outcome.final_tokens,
        outcome.budget,
        outcome.window,
        outcome.before_tokens,
        "" if outcome.within_budget else "; still over budget after best effort",
    )


def request_guard_callback(
    *,
    model: str,
    llm_config: Any = None,
    scope: str,
    logger: Any,
    ratio: float = DEFAULT_GUARD_RATIO,
    snip_marker: str | None = None,
    count: Callable[[dict[str, Any]], int] | None = None,
) -> Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], GuardOutcome]:
    """Build the ``context_guard`` callback accepted by ``run_labeled_step``.

    ``run_labeled_step`` invokes it on a per-request copy of the messages just
    before the provider call, so the capability's durable history (which the
    agentic loop keeps appending to) is never trimmed. The returned callback
    reports its :class:`GuardOutcome` so the caller can decide whether the
    copy actually needs to replace the outgoing request.
    """
    context_window = getattr(llm_config, "context_window", None)
    max_tokens = getattr(llm_config, "max_tokens", None)

    def _guard(
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None,
    ) -> GuardOutcome:
        outcome = apply_request_guard(
            messages,
            model=model,
            context_window=context_window,
            max_tokens=max_tokens,
            tool_schemas=tool_schemas,
            ratio=ratio,
            snip_marker=snip_marker,
            count=count,
        )
        log_request_guard(outcome, scope=scope, logger=logger)
        return outcome

    return _guard


__all__ = [
    "DEFAULT_GUARD_RATIO",
    "DEFAULT_TOOL_RESULT_MARKER",
    "TRIM_PREFIX",
    "GuardOutcome",
    "TrimResult",
    "apply_request_guard",
    "default_message_token_count",
    "hard_trim_to_budget",
    "log_request_guard",
    "request_guard_callback",
]
