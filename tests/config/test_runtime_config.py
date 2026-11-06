"""Regression coverage for the L4b/L4c configuration SSOT work.

The plan requires ``pytest tests/config -q`` to cover multiple data roots,
cache invalidation, environment precedence and the gradual-rollout flag.
These tests avoid mutating process-wide singletons without restoring them:
``monkeypatch`` records the original module constants before a rebind, so a
test cannot leak a temporary DEEPTUTOR_HOME into the rest of the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeptutor.runtime.home import runtime_flag
from deeptutor.services.config.config_sources import collect_config_sources
from deeptutor.services.config.runtime_config import (
    get_runtime_config,
    invalidate_runtime_config,
    ssot_enabled,
)


def _settings_dir(home: Path) -> Path:
    return home / "data" / "user" / "settings"


# --- runtime_flag (import-time safe, dark by default) -----------------------


def test_runtime_flag_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPTUTOR_RUNTIME_SSOT", raising=False)
    assert runtime_flag("ssot") is False
    assert ssot_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", " on ", "True"])
def test_runtime_flag_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DEEPTUTOR_RUNTIME_SSOT", value)
    assert runtime_flag("ssot") is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "  ", "garbage"])
def test_runtime_flag_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DEEPTUTOR_RUNTIME_SSOT", value)
    assert runtime_flag("ssot") is False


def test_runtime_flag_honours_explicit_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPTUTOR_RUNTIME_SSOT", raising=False)
    assert runtime_flag("ssot", default=True) is True


# --- get_runtime_config: precedence, multi-home, read-only ------------------


def test_explicit_home_wins_over_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_home = tmp_path / "env-home"
    explicit = tmp_path / "explicit-home"
    monkeypatch.setenv("DEEPTUTOR_HOME", str(env_home))

    config = get_runtime_config(home=explicit)

    assert config.home == explicit.resolve()
    assert config.data_root == explicit.resolve() / "data"
    assert config.settings_dir == _settings_dir(explicit.resolve())
    assert config.catalog_path == _settings_dir(explicit.resolve()) / "model_catalog.json"
    assert config.main_config_path == _settings_dir(explicit.resolve()) / "main.yaml"
    assert config.sources["home"] == "explicit argument"
    assert "explicit home" in config.sources["settings_dir"]


def test_environment_home_is_used_when_no_explicit_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_home = tmp_path / "env-home"
    monkeypatch.setenv("DEEPTUTOR_HOME", str(env_home))

    config = get_runtime_config()

    assert config.home == env_home.resolve()
    assert config.data_root == env_home.resolve() / "data"
    assert config.sources["home"] == "DEEPTUTOR_HOME"


def test_cwd_fallback_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPTUTOR_HOME", raising=False)

    config = get_runtime_config()

    assert config.home == Path.cwd().resolve()
    assert config.sources["home"] == "cwd"


def test_two_data_roots_never_share_catalog_or_settings(tmp_path: Path) -> None:
    homes = [tmp_path / "home-a", tmp_path / "home-b"]
    for index, home in enumerate(homes):
        settings = _settings_dir(home)
        settings.mkdir(parents=True)
        (settings / "model_catalog.json").write_text(
            json.dumps({"version": 1, "connections": [{"id": f"conn-{index}"}]}),
            encoding="utf-8",
        )
        (settings / "main.yaml").write_text(f"language: lang-{index}\n", encoding="utf-8")

    configs = [get_runtime_config(home=home) for home in homes]

    assert configs[0].catalog_path != configs[1].catalog_path
    assert configs[0].settings_dir != configs[1].settings_dir
    assert configs[0].catalog["connections"][0]["id"] == "conn-0"
    assert configs[1].catalog["connections"][0]["id"] == "conn-1"
    assert configs[0].main_config["language"] == "lang-0"
    assert configs[1].main_config["language"] == "lang-1"


def test_snapshot_is_read_only_and_tolerates_missing_files(tmp_path: Path) -> None:
    home = tmp_path / "empty-home"

    config = get_runtime_config(home=home)

    assert config.catalog == {}
    assert config.main_config == {}
    assert not config.settings_dir.exists(), "a read must never create a settings tree"
    assert config.sources["catalog"] == "read-only file snapshot"


def test_malformed_files_do_not_break_the_snapshot(tmp_path: Path) -> None:
    home = tmp_path / "broken-home"
    settings = _settings_dir(home)
    settings.mkdir(parents=True)
    (settings / "model_catalog.json").write_text("{not json", encoding="utf-8")
    (settings / "main.yaml").write_text("{{bad yaml", encoding="utf-8")

    config = get_runtime_config(home=home)

    assert config.catalog == {}
    assert config.main_config == {}
    assert config.catalog_path == settings / "model_catalog.json"


# --- L4c: frozen import-time paths ------------------------------------------


def test_rebind_is_a_noop_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import deeptutor.services.config as config_package
    from deeptutor.runtime import cache_reset
    from deeptutor.services.config import loader, launch_settings, model_catalog

    for module, name in (
        (loader, "PROJECT_ROOT"),
        (launch_settings, "PROJECT_ROOT"),
        (config_package, "PROJECT_ROOT"),
        (model_catalog, "CATALOG_PATH"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))

    original = loader.PROJECT_ROOT
    assert cache_reset.rebind_import_time_paths(enabled=False, home=tmp_path) == []
    assert loader.PROJECT_ROOT == original
    assert launch_settings.PROJECT_ROOT == original


def test_rebind_follows_the_new_home_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import deeptutor.services.config as config_package
    import deeptutor.services.path_service as path_module
    from deeptutor.runtime import cache_reset
    from deeptutor.services.config import loader, launch_settings, model_catalog
    from deeptutor.services.path_service import PathService

    for module, name in (
        (loader, "PROJECT_ROOT"),
        (launch_settings, "PROJECT_ROOT"),
        (config_package, "PROJECT_ROOT"),
        (model_catalog, "CATALOG_PATH"),
    ):
        monkeypatch.setattr(module, name, getattr(module, name))

    home = tmp_path / "new-home"
    path_service = PathService(workspace_root=home / "data")
    monkeypatch.setattr(path_module, "get_path_service", lambda: path_service)

    rebound = cache_reset.rebind_import_time_paths(enabled=True, home=home)

    assert len(rebound) == 4
    assert any("loader.PROJECT_ROOT" in name for name in rebound)
    assert any("model_catalog.CATALOG_PATH" in name for name in rebound)
    assert loader.PROJECT_ROOT == home
    assert launch_settings.PROJECT_ROOT == home
    assert config_package.PROJECT_ROOT == home
    assert model_catalog.CATALOG_PATH == path_service.get_settings_file("model_catalog")


# --- invalidate_runtime_config: one broadcast, failure isolated -------------


def _spy_invalidation(monkeypatch: pytest.MonkeyPatch, *, fail_on: str | None = None) -> list[str]:
    """Replace every real reset with a spy so tests never clear live caches."""

    import deeptutor.runtime.cache_reset as cache_reset
    import deeptutor.services.config.model_catalog as model_catalog
    import deeptutor.services.config.runtime_settings as runtime_settings
    import deeptutor.services.embedding.client as embedding_client
    import deeptutor.services.llm.client as llm_client
    import deeptutor.services.llm.config as llm_config
    import deeptutor.utils.config_manager as config_manager

    calls: list[str] = []

    def spy(name: str):
        def inner() -> None:
            calls.append(name)
            if name == fail_on:
                raise RuntimeError(f"simulated failure in {name}")

        return inner

    monkeypatch.setattr(cache_reset, "reset_runtime_singletons", spy("runtime_singletons"))
    monkeypatch.setattr(model_catalog.ModelCatalogService, "_instances", {"stale": object()})
    monkeypatch.setattr(runtime_settings.RuntimeSettingsService, "_instances", {"stale": object()})
    monkeypatch.setattr(config_manager.ConfigManager, "reset_for_tests", spy("config_manager"))
    monkeypatch.setattr(llm_config, "clear_llm_config_cache", spy("llm_config_cache"))
    monkeypatch.setattr(llm_client, "reset_llm_client", spy("llm_client"))
    monkeypatch.setattr(embedding_client, "reset_embedding_client", spy("embedding_client"))
    return calls


def test_invalidation_reaches_every_config_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_invalidation(monkeypatch)

    report = invalidate_runtime_config(reason="unit test")

    assert report == {
        "runtime_singletons": True,
        "catalog_instances": True,
        "runtime_settings_instances": True,
        "config_manager": True,
        "llm_config_cache": True,
        "llm_and_embedding_clients": True,
    }
    assert "runtime_singletons" in calls
    assert "config_manager" in calls
    assert "llm_client" in calls
    assert "embedding_client" in calls


def test_one_failing_cache_does_not_skip_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_invalidation(monkeypatch, fail_on="config_manager")

    report = invalidate_runtime_config()

    assert report["config_manager"] is False
    assert report["llm_config_cache"] is True
    assert report["llm_and_embedding_clients"] is True
    assert "embedding_client" in calls, "invalidation must continue past a failure"


# --- diagnostics stay non-sensitive -----------------------------------------


def test_diagnostic_exposes_unified_view_and_flag_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPTUTOR_RUNTIME_SSOT", raising=False)

    snapshot = collect_config_sources()

    unified = snapshot["unified"]
    assert {"resolved_home", "catalog_path", "main_config_path", "sources"} <= set(unified)
    assert snapshot["runtime_flags"]["ssot"] is False
    assert "sk-" not in json.dumps(snapshot)
    assert "connections" not in unified, "diagnostics must not echo catalog contents"
