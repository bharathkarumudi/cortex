"""Tests for Chroma record coercion that do not require a running server."""

from __future__ import annotations

from typing import Any

from cortex_common.chroma_memory import ChromaMemoryRepository


def _metadata(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "user_id": "user-a",
        "content_type": "fact",
        "source": "direct",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "access_count": 0,
        "deleted": False,
    }
    values.update(overrides)
    return values


def test_record_without_use_case_defaults_to_chat() -> None:
    record = ChromaMemoryRepository._record_from_values("id-1", "text", _metadata())

    assert record.use_case == "chat"
    assert record.schema_version == 2


def test_record_with_unknown_use_case_falls_back_to_chat() -> None:
    record = ChromaMemoryRepository._record_from_values("id-1", "text", _metadata(use_case="bogus"))

    assert record.use_case == "chat"


def test_record_with_valid_use_case_is_preserved() -> None:
    record = ChromaMemoryRepository._record_from_values(
        "id-1", "text", _metadata(use_case="digital_twin")
    )

    assert record.use_case == "digital_twin"
