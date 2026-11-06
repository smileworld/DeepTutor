"""Tests for the read-only configuration provenance snapshot (L4a)."""

from __future__ import annotations

import json

from deeptutor.services.config.config_sources import (
    _service_summary,
    collect_config_sources,
)


def test_snapshot_has_expected_sections() -> None:
    snapshot = collect_config_sources()
    for key in (
        "generated_at",
        "runtime",
        "settings_dir",
        "catalog",
        "llm_runtime",
        "caches",
        "env_keys",
        "warnings",
    ):
        assert key in snapshot
    assert snapshot["runtime"]["resolved_data_root"]
    assert isinstance(snapshot["warnings"], list)
    assert isinstance(snapshot["env_keys"], list)


def test_snapshot_reports_both_catalog_paths() -> None:
    snapshot = collect_config_sources()
    catalog = snapshot["catalog"]
    assert "import_time_path" in catalog
    assert "runtime_path" in catalog
    if isinstance(catalog["runtime_path"], str):
        assert isinstance(catalog["paths_agree"], bool)


def test_snapshot_never_exposes_credentials() -> None:
    snapshot = collect_config_sources()
    for service in snapshot["catalog"].get("services", {}).values():
        assert "api_key" not in service
        assert "profiles" not in service
    assert "sk-" not in json.dumps(snapshot)


def test_service_summary_keeps_protocol_but_drops_profile_body() -> None:
    summary = _service_summary(
        {
            "services": {
                "llm": {
                    "active_profile_id": "p1",
                    "active_model_id": "m1",
                    "profiles": [
                        {
                            "id": "p1",
                            "name": "XF",
                            "api_format": "openai_responses",
                            "wire_api": "responses",
                            "api_key": "secret",
                        }
                    ],
                },
                "task": {"mode": "reference", "active_profile_id": None},
            }
        }
    )
    assert summary["llm"]["api_format"] == "openai_responses"
    assert summary["llm"]["wire_api"] == "responses"
    assert summary["task"]["mode"] == "reference"
    assert "api_key" not in summary["llm"]
    assert "profiles" not in summary["llm"]
