"""Shared cache reset after selecting a runtime home.

Service imports remain lazy so CLI/bootstrap can select DEEPTUTOR_HOME first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

from deeptutor.runtime.home import (
    DEEPTUTOR_HOME_ENV,
    get_runtime_home,
    runtime_flag,
)

logger = logging.getLogger(__name__)


def rebind_import_time_paths(
    *, enabled: bool | None = None, home: Path | None = None
) -> list[str]:
    """Refresh module-level paths that were frozen at import time (L4c).

    ``loader.PROJECT_ROOT``, ``launch_settings.PROJECT_ROOT`` and
    ``model_catalog.CATALOG_PATH`` are computed when their modules are first
    imported. If the runtime home is selected later (launcher ``--home``,
    CLI setup), those constants keep pointing at the old tree. This returns
    the names of attributes actually rebound.

    Only already-imported modules are touched: a reset must not pull the whole
    settings stack in just to rebind a constant. Gated by
    ``DEEPTUTOR_RUNTIME_SSOT`` (default off) because changing these globals
    affects every caller in the process.
    """

    if enabled is None:
        enabled = runtime_flag("ssot")
    if not enabled:
        return []

    from deeptutor.services.path_service import get_path_service

    resolved_home = home or get_runtime_home()
    catalog_path = get_path_service().get_settings_file("model_catalog")
    rebound: list[str] = []

    for module_name in (
        "deeptutor.services.config.loader",
        "deeptutor.services.config.launch_settings",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "PROJECT_ROOT"):
            module.PROJECT_ROOT = resolved_home
            rebound.append(f"{module_name}.PROJECT_ROOT")

    catalog_module = sys.modules.get("deeptutor.services.config.model_catalog")
    if catalog_module is not None:
        catalog_module.CATALOG_PATH = catalog_path
        rebound.append("deeptutor.services.config.model_catalog.CATALOG_PATH")

    # __init__.py re-exports loader.PROJECT_ROOT as a package-level constant.
    package = sys.modules.get("deeptutor.services.config")
    if package is not None and hasattr(package, "PROJECT_ROOT"):
        package.PROJECT_ROOT = resolved_home
        rebound.append("deeptutor.services.config.PROJECT_ROOT")

    return rebound


def reset_runtime_singletons() -> None:
    """Refresh home-dependent caches, reporting failures without skipping peers."""
    home = get_runtime_home()
    logger.info(
        "Runtime home resolved: %s (data root: %s; %s=%s)",
        home,
        home / "data",
        DEEPTUTOR_HOME_ENV,
        os.getenv(DEEPTUTOR_HOME_ENV) or "<unset>",
    )
    try:
        from deeptutor.services.path_service import PathService

        PathService.reset_instance()
    except Exception:
        logger.exception(
            "PathService reset failed; the runtime may keep writing to the "
            "previous DEEPTUTOR_HOME instead of the selected one"
        )
    try:
        from deeptutor.services.config.runtime_settings import RuntimeSettingsService

        RuntimeSettingsService._instances.clear()
    except Exception:
        logger.warning(
            "RuntimeSettings cache reset failed; settings cached for the "
            "previous DEEPTUTOR_HOME may be reused",
            exc_info=True,
        )
    try:
        from deeptutor.services.config.model_catalog import ModelCatalogService

        ModelCatalogService._instances.clear()
    except Exception:
        logger.warning(
            "ModelCatalog cache reset failed; model catalog cached for the "
            "previous DEEPTUTOR_HOME may be reused",
            exc_info=True,
        )
    # CLI setup can select --home after multi-user paths were imported.
    # Keep the shared reset seam responsible for every cached home boundary.
    if sys.modules.get("deeptutor.multi_user.paths") is not None:
        try:
            from deeptutor.multi_user import paths

            paths.PROJECT_ROOT = home
            paths.ADMIN_WORKSPACE_ROOT = paths.PROJECT_ROOT / "data"
            paths.USERS_ROOT = paths.ADMIN_WORKSPACE_ROOT / "users"
            paths.SYSTEM_ROOT = paths.ADMIN_WORKSPACE_ROOT / "system"
            paths.LEGACY_MULTI_USER_ROOT = paths.PROJECT_ROOT / "multi-user"
            paths._path_services.clear()
            paths._legacy_migration_done = False
        except Exception:
            logger.warning(
                "Multi-user path cache reset failed; owner paths may retain the previous runtime home",
                exc_info=True,
            )

    try:
        rebound = rebind_import_time_paths(home=home)
        if rebound:
            logger.info("Rebound import-time config paths: %s", ", ".join(rebound))
    except Exception:
        logger.warning(
            "Import-time config path rebind failed; frozen paths may still point "
            "at the previous runtime home",
            exc_info=True,
        )
