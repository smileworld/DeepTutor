"""Block capability probing + configurable downgrade policy (myDeeptutor L3).

Two complementary mechanisms:

* **Capability probing** (plan time). Generators with heavy local dependencies
  declare ``probe()``; unavailable types are kept out of the page plan entirely.
  Results are cached in-process with an environment fingerprint and a TTL, and
  a readable snapshot is written to ``Book.metadata["capabilities"]`` so a
  diagnosis can explain *why* a block type disappeared.

* **Runtime downgrade** (generation time). A block that fails anyway can be
  converted to a cheaper/more robust type, with ``original_type`` and the
  failure reason recorded in ``block.metadata["downgrade"]`` for traceability
  and one-click reversal. The policy is data, not code: defaults live here and
  a book can override them via ``Book.metadata["block_downgrades"]``.

The downgrade table mirrors the four manual ``change-block-type`` repairs that
were needed to get book ``bk_1d69e3e22a`` to ``ready`` in this session.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from typing import Any

from ..models import BlockType
from .base import BlockGeneratorRegistry, ProbeResult, get_block_registry

logger = logging.getLogger(__name__)


DEFAULT_DOWNGRADE_MAP: dict[BlockType, BlockType] = {
    BlockType.ANIMATION: BlockType.SECTION,
    BlockType.INTERACTIVE: BlockType.SECTION,
    BlockType.QUIZ: BlockType.FLASH_CARDS,
}

CAPABILITIES_META_KEY = "capabilities"
DOWNGRADE_META_KEY = "block_downgrades"
AUTO_DOWNGRADE_META_KEY = "auto_downgrade"
CAPABILITIES_TTL_SECONDS = 6 * 60 * 60

# fingerprint -> (probed_at, results). Probing is pure and synchronous-cheap;
# the cache only avoids re-reading module manifests on every page of a book.
_PROBE_CACHE: dict[str, tuple[float, dict[BlockType, ProbeResult]]] = {}


def downgrade_map_for_book(book: Any) -> dict[BlockType, BlockType]:
    """Resolve the effective downgrade policy for *book*.

    ``Book.metadata["auto_downgrade"] = False`` disables it for one book.
    ``Book.metadata["block_downgrades"]`` is a ``{source: target|null}`` map;
    a null/false target removes that default. Unknown enum values are ignored
    rather than raising, because metadata is user-editable JSON.
    """
    metadata = getattr(book, "metadata", None) or {}
    if not isinstance(metadata, dict) or metadata.get(AUTO_DOWNGRADE_META_KEY) is False:
        return {}

    result = dict(DEFAULT_DOWNGRADE_MAP)
    overrides = metadata.get(DOWNGRADE_META_KEY)
    if not isinstance(overrides, dict):
        return result

    for raw_from, raw_to in overrides.items():
        try:
            source = BlockType(str(raw_from).strip().lower())
        except ValueError:
            continue
        if raw_to is None or raw_to is False or str(raw_to).strip().lower() in (
            "",
            "none",
            "off",
            "false",
        ):
            result.pop(source, None)
            continue
        try:
            result[source] = BlockType(str(raw_to).strip().lower())
        except ValueError:
            continue
    return result


def capability_fingerprint(registry: BlockGeneratorRegistry) -> str:
    """Identify the environment whose capability results may be reused.

    The data root and ``sys.prefix`` are part of the key because optional
    extras and enabled modules can differ between installations/venvs.
    """
    try:
        from deeptutor.runtime.home import get_runtime_data_root

        data_root = str(get_runtime_data_root())
    except Exception:  # noqa: BLE001
        data_root = "unknown"
    parts = (
        sys.prefix,
        data_root,
        ",".join(sorted(block_type.value for block_type in registry.types())),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


async def probe_registry(
    registry: BlockGeneratorRegistry | None = None,
    *,
    ttl_seconds: float = CAPABILITIES_TTL_SECONDS,
    refresh: bool = False,
) -> dict[BlockType, ProbeResult]:
    """Probe all generators, reusing a short-lived fingerprinted cache."""
    target = registry or get_block_registry()
    key = capability_fingerprint(target)
    now = time.time()
    cached = _PROBE_CACHE.get(key)
    if not refresh and cached is not None and now - cached[0] < ttl_seconds:
        return dict(cached[1])
    results = await target.probe_all()
    _PROBE_CACHE[key] = (now, dict(results))
    return results


async def available_block_types(
    registry: BlockGeneratorRegistry | None = None,
    *,
    refresh: bool = False,
) -> set[BlockType]:
    probed = await probe_registry(registry, refresh=refresh)
    return {bt for bt, result in probed.items() if result.available}


def capabilities_snapshot(
    results: dict[BlockType, ProbeResult],
    registry: BlockGeneratorRegistry,
) -> dict[str, Any]:
    """A JSON-safe diagnostic view stored on ``Book.metadata``.

    ``probed_at`` comes from the probe cache, not from ``time.time()`` here, so
    the snapshot is stable across pages and ``record_capabilities`` does not
    rewrite the manifest on every compile.
    """
    fingerprint = capability_fingerprint(registry)
    cached = _PROBE_CACHE.get(fingerprint)
    return {
        "fingerprint": fingerprint,
        "probed_at": cached[0] if cached is not None else time.time(),
        "available": sorted(bt.value for bt, r in results.items() if r.available),
        "unavailable": {
            bt.value: (r.reason or "unavailable")
            for bt, r in sorted(results.items(), key=lambda pair: pair[0].value)
            if not r.available
        },
    }


def record_capabilities(book: Any, storage: Any, snapshot: dict[str, Any]) -> None:
    """Best-effort persist a capability snapshot for later diagnosis.

    Never raises: a read-only or embedded storage must not break compilation
    just because we wanted to leave an audit trail.
    """
    if book is None or not snapshot:
        return
    try:
        # Re-read the manifest immediately before writing. The compiler may
        # have loaded its copy a while ago, and writing that stale object back
        # would clobber a concurrent editor's revision bump.
        loader = getattr(storage, "load_book", None)
        target = loader(getattr(book, "id", "")) if loader is not None else None
        target = target or book
        metadata = dict(getattr(target, "metadata", None) or {})
        if metadata.get(CAPABILITIES_META_KEY) == snapshot:
            return
        metadata[CAPABILITIES_META_KEY] = snapshot
        target.metadata = metadata
        saver = getattr(storage, "save_book", None)
        if saver is not None:
            saver(target)
    except Exception:  # noqa: BLE001
        logger.debug("Could not persist capability snapshot", exc_info=True)


def clear_probe_cache() -> None:
    """Drop cached probe results (tests, or after installing extras)."""
    _PROBE_CACHE.clear()


__all__ = [
    "AUTO_DOWNGRADE_META_KEY",
    "CAPABILITIES_META_KEY",
    "CAPABILITIES_TTL_SECONDS",
    "DEFAULT_DOWNGRADE_MAP",
    "DOWNGRADE_META_KEY",
    "available_block_types",
    "capabilities_snapshot",
    "capability_fingerprint",
    "clear_probe_cache",
    "downgrade_map_for_book",
    "probe_registry",
    "record_capabilities",
]
