"""Tests for fail-fast deployment configuration."""

from __future__ import annotations

import pytest

from cortex_common.config import ConfigurationError, Settings


def test_settings_require_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
        Settings.from_environment()


def test_settings_reject_inverted_distance_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RECALL_MAX_DISTANCE", "0.2")
    monkeypatch.setenv("DEDUP_MAX_DISTANCE", "0.3")

    with pytest.raises(ConfigurationError, match="DEDUP_MAX_DISTANCE"):
        Settings.from_environment()


def test_settings_reject_unknown_web_search_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEB_SEARCH_AGENTS", "responder,unknown")

    with pytest.raises(ConfigurationError, match="unknown WEB_SEARCH_AGENTS"):
        Settings.from_environment()
