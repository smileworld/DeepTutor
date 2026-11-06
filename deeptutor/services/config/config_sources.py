"""Read-only diagnostics for where runtime configuration actually came from.

The settings stack can resolve the same value through several stores and caches:
import-time constants, a per-user ``PathService``, process environment
variables, and the catalog / runtime-settings services (each with its own
instance cache). When two of them disagree, the symptom looks like "the CLI
says there are no books but the web UI has books".

This module only *observes*. It never writes a file, never clears a cache and
never installs a scoped config, so a diagnostic request cannot change the very
configuration it is describing. It also never returns credentials: only paths,
file names, ids and protocol names.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deeptutor.runtime.home import (
    DEEPTUTOR_HOME_ENV,
    get_runtime_data_root,
    get_runtime_home,
    runtime_flag,
)
from deeptutor.services.provider_registry import (
    effective_backend,
    find_by_name,
    normalize_api_format,
    wire_api_from_api_format,
)

_ENV_PREFIXES = ("DEEPTUTOR_", "LLM_", "OPENAI_")


def _safe(fn: Callable[[], Any]) -> Any:
    """A broken probe must not turn a diagnostic request into a 500."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
        return {"error": f"{type(exc).__name__}: {exc}"}


def _file_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "name": path.name,
            "size_bytes": stat.st_size,
            "mtime": round(stat.st_mtime, 3),
        }
    except OSError as exc:
        return {"name": path.name, "error": str(exc)}


def _settings_files(settings_dir: Path) -> list[dict[str, Any]]:
    try:
        paths = sorted(
            path
            for path in settings_dir.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
        )
        return [_file_info(path) for path in paths]
    except OSError as exc:
        return [{"error": str(exc)}]


def _active_profile_fields(service: dict[str, Any]) -> dict[str, Any]:
    active_id = service.get("active_profile_id")
    for profile in service.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        if profile.get("id") == active_id or profile.get("profile_id") == active_id:
            return {
                "profile_name": profile.get("name"),
                "api_format": profile.get("api_format"),
                "wire_api": profile.get("wire_api"),
            }
    return {}


def _service_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    """Non-sensitive summary — profile/model ids and protocol, never keys."""
    services = catalog.get("services") or {}
    summary: dict[str, Any] = {}
    for name, service in services.items():
        if not isinstance(service, dict):
            continue
        entry: dict[str, Any] = {
            "active_profile_id": service.get("active_profile_id"),
            "active_model_id": service.get("active_model_id"),
        }
        if "mode" in service:
            entry["mode"] = service.get("mode")
        entry.update(_active_profile_fields(service))
        summary[name] = entry
    return summary


def _cache_snapshot() -> dict[str, Any]:
    from deeptutor.services.config.model_catalog import ModelCatalogService
    from deeptutor.services.config.runtime_settings import RuntimeSettingsService
    from deeptutor.services.llm import config as llm_config_module
    from deeptutor.utils.config_manager import ConfigManager

    return {
        "llm_config_cached": getattr(llm_config_module, "_LLM_CONFIG_CACHE", None) is not None,
        "catalog_service_instances": len(getattr(ModelCatalogService, "_instances", {}) or {}),
        "runtime_settings_instances": len(
            getattr(RuntimeSettingsService, "_instances", {}) or {}
        ),
        "config_manager_cached": bool(getattr(ConfigManager, "_config_cache", None)),
    }


def _unified_summary() -> dict[str, Any]:
    """Non-sensitive projection of the L4b unified read entry point.

    ``RuntimeConfig.catalog`` may contain provider credentials, so this never
    returns its contents — only the resolved paths, provenance and counts.
    """

    from deeptutor.services.config.runtime_config import get_runtime_config

    snapshot = get_runtime_config()
    return {
        "resolved_home": str(snapshot.home),
        "resolved_data_root": str(snapshot.data_root),
        "settings_dir": str(snapshot.settings_dir),
        "catalog_path": str(snapshot.catalog_path),
        "main_config_path": str(snapshot.main_config_path),
        "sources": dict(snapshot.sources),
        "catalog_connections": len(snapshot.catalog.get("connections") or []),
        "catalog_path_exists": snapshot.catalog_path.exists(),
        "main_config_path_exists": snapshot.main_config_path.exists(),
    }


def collect_config_sources() -> dict[str, Any]:
    """Return a snapshot describing the provenance of live configuration."""
    from deeptutor.services.config import get_model_catalog_service
    from deeptutor.services.config.model_catalog import CATALOG_PATH
    from deeptutor.services.config.provider_runtime import resolve_llm_runtime_config
    from deeptutor.services.path_service import get_path_service

    runtime_home = get_runtime_home()
    data_root = get_runtime_data_root()
    home_env = os.getenv(DEEPTUTOR_HOME_ENV)

    settings_dir = _safe(lambda: get_path_service().get_settings_dir())
    runtime_catalog_path = _safe(
        lambda: str(get_path_service().get_settings_file("model_catalog"))
    )
    catalog = _safe(lambda: get_model_catalog_service().load())
    resolved = _safe(lambda: resolve_llm_runtime_config())

    catalog_payload: dict[str, Any] = {
        "import_time_path": str(CATALOG_PATH),
        "runtime_path": runtime_catalog_path,
    }
    if isinstance(catalog, dict):
        catalog_payload["connections"] = len(catalog.get("connections") or [])
        catalog_payload["services"] = _service_summary(catalog)
    else:
        catalog_payload["error"] = catalog
    if isinstance(runtime_catalog_path, str):
        catalog_payload["paths_agree"] = (
            str(CATALOG_PATH) == runtime_catalog_path
        )

    llm_runtime: dict[str, Any] = {}
    if not isinstance(resolved, dict) or "error" not in resolved:
        api_format = getattr(resolved, "api_format", "auto")
        wire_api = getattr(resolved, "wire_api", "auto")
        provider_name = getattr(resolved, "provider_name", None)
        spec = _safe(lambda: find_by_name(provider_name)) if provider_name else None
        llm_runtime = {
            "model": getattr(resolved, "model", None),
            "provider_name": provider_name,
            "binding": getattr(resolved, "binding", None),
            "api_format": api_format,
            "wire_api": wire_api,
            "normalized_api_format": _safe(lambda: normalize_api_format(api_format)),
            "implied_wire_api": _safe(lambda: wire_api_from_api_format(api_format)),
            "effective_backend": _safe(
                lambda: effective_backend(spec if not isinstance(spec, dict) else None, api_format)
            ),
        }
    else:
        llm_runtime = {"error": resolved}

    warnings: list[str] = []
    if home_env is None:
        warnings.append("DEEPTUTOR_HOME is unset; the runtime home fell back to cwd.")
    if isinstance(runtime_catalog_path, str) and str(CATALOG_PATH) != runtime_catalog_path:
        warnings.append(
            "Import-time CATALOG_PATH differs from the current user's catalog path; "
            "a stale module-level constant may be in use."
        )
    if not data_root.exists():
        warnings.append(f"Resolved data root does not exist: {data_root}")

    return {
        "generated_at": round(time.time(), 3),
        "runtime": {
            "cwd": str(Path.cwd().resolve()),
            "home_env": home_env,
            "resolved_home": str(runtime_home),
            "resolved_data_root": str(data_root),
        },
        "settings_dir": (
            {"path": str(settings_dir), "files": _settings_files(settings_dir)}
            if isinstance(settings_dir, Path)
            else settings_dir
        ),
        "catalog": catalog_payload,
        "llm_runtime": llm_runtime,
        "unified": _safe(_unified_summary),
        "runtime_flags": {"ssot": runtime_flag("ssot")},
        "caches": _safe(_cache_snapshot),
        "env_keys": sorted(
            key for key in os.environ if key.startswith(_ENV_PREFIXES)
        ),
        "warnings": warnings,
    }


__all__ = ["collect_config_sources"]
