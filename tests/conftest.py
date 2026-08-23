"""Shared isolated fixtures for Cortex MCP unit tests."""

from __future__ import annotations

import pytest

from cortex_common.config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Return valid production-like settings without a real secret."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REQUIRE_USER_ID_HEADER", "true")
    return Settings.from_environment()
