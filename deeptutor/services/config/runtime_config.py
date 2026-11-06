"""One coherent view of runtime configuration, and one invalidation seam (L4b).

Historically each configuration store was reached through its own singleton
and had its own ad-hoc cache reset. After a runtime-home switch (or an admin
catalog apply) those caches could disagree with each other and with the files
on disk: the CLI reported "No books yet" while the API had books, and the
diagnostic endpoint (``config_sources``) could only *describe* that drift.

``get_runtime_config()`` resolves every location from a single home on every
call. It deliberately reads the files directly: a read must never trigger a
catalog migration write, install a scoped config, or clear a cache. Secrets
may be present in ``RuntimeConfig.catalog`` — callers must not echo it to an
API response.

``invalidate_runtime_config()`` is the matching write-side seam: one call
broadcasts invalidation to every configuration cache, reporting failures per
cache instead of aborting after the first one. Adoption of this broadcast on
the settings-apply path is gated by ``DEEPTUTOR_RUNTIME_SSOT`` (default off),
so enabling L4b is a deliberate, reversible deployment decision.

Nothing here resolves a home at import time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deeptutor.runtime.home import (
    DEEPTUTOR_HOME_ENV,
    get_runtime_data_root,
    get_runtime_home,
    runtime_flag,
)

logger = logging.getLogger(__name__)

SSOT_FLAG = "ssot"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """A read-only snapshot of where this process reads its configuration."""

    home: Path
    data_root: Path
    settings_dir: Path
    catalog_path: Path
    main_config_path: Path
    catalog: dict[str, Any]
    main_config: dict[str, Any]
    sources: dict[str, str]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # noqa: BLE001 - a snapshot read must not break a request
        logger.debug("Could not read %s for the runtime config snapshot", path, exc_info=True)
        return {}


def _read_yaml_object(path: Path) -> dict[str, Any]:
    try:
        import yaml

        if not path.exists():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # noqa: BLE001 - a snapshot read must not break a request
        logger.debug("Could not read %s for the runtime config snapshot", path, exc_info=True)
        return {}


def _home_source(home: str | Path | None) -> str:
    if home is not None:
        return "explicit argument"
    if os.getenv(DEEPTUTOR_HOME_ENV):
        return DEEPTUTOR_HOME_ENV
    return "cwd"


def get_runtime_config(*, home: str | Path | None = None) -> RuntimeConfig:
    """Resolve all configuration locations from one runtime home, right now.

    With no *home* argument this follows production semantics: the singleton
    ``PathService`` (which may be scoped to a multi-user directory) and the
    current process environment. Passing *home* builds a detached PathService
    for that home, which is what CLI/bootstrap and multi-home tests need.
    """

    from deeptutor.services.path_service import PathService, get_path_service

    resolved_home = get_runtime_home(home)
    data_root = get_runtime_data_root(resolved_home)
    path_service = (
        PathService(workspace_root=data_root) if home is not None else get_path_service()
    )
    settings_dir = path_service.get_settings_dir()
    catalog_path = path_service.get_settings_file("model_catalog")
    main_config_path = path_service.get_runtime_config_file("main")

    return RuntimeConfig(
        home=resolved_home,
        data_root=data_root,
        settings_dir=settings_dir,
        catalog_path=catalog_path,
        main_config_path=main_config_path,
        catalog=_read_json_object(catalog_path),
        main_config=_read_yaml_object(main_config_path),
        sources={
            "home": _home_source(home),
            "settings_dir": (
                "PathService(explicit home)" if home is not None else "PathService singleton"
            ),
            "catalog": "read-only file snapshot",
            "main_config": "read-only file snapshot",
        },
    )


def invalidate_runtime_config(*, reason: str = "runtime config invalidation") -> dict[str, bool]:
    """Broadcast one invalidation to every configuration cache.

    Returns a per-cache success report so an operator can see a partial
    failure instead of assuming the whole runtime was refreshed. Failure of
    one cache never skips the others.
    """

    from deeptutor.runtime.cache_reset import reset_runtime_singletons
    from deeptutor.services.config.model_catalog import ModelCatalogService
    from deeptutor.services.config.runtime_settings import RuntimeSettingsService
    from deeptutor.services.llm.config import clear_llm_config_cache
    from deeptutor.utils.config_manager import ConfigManager

    report: dict[str, bool] = {}

    def _run(name: str, action: Any) -> None:
        try:
            action()
            report[name] = True
        except Exception:  # noqa: BLE001 - keep invalidating the remaining caches
            report[name] = False
            logger.warning("Runtime config invalidation failed for %s", name, exc_info=True)

    logger.info("Invalidating runtime configuration caches (%s)", reason)
    _run("runtime_singletons", reset_runtime_singletons)
    _run("catalog_instances", lambda: ModelCatalogService._instances.clear())
    _run("runtime_settings_instances", lambda: RuntimeSettingsService._instances.clear())
    _run("config_manager", ConfigManager.reset_for_tests)
    _run("llm_config_cache", clear_llm_config_cache)

    # The client singletons are reset separately: their constructors also
    # depend on the LLM config cache, so ordering matters here.
    def _reset_clients() -> None:
        from deeptutor.services.embedding.client import reset_embedding_client
        from deeptutor.services.llm.client import reset_llm_client

        reset_llm_client()
        reset_embedding_client()

    _run("llm_and_embedding_clients", _reset_clients)
    return report


def ssot_enabled() -> bool:
    """Whether the L4b/L4c unified-configuration rollout is enabled."""

    return runtime_flag(SSOT_FLAG)


__all__ = [
    "SSOT_FLAG",
    "RuntimeConfig",
    "get_runtime_config",
    "invalidate_runtime_config",
    "ssot_enabled",
]
