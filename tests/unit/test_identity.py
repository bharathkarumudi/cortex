"""Tests for trusted request-header identity extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cortex_common.identity import IdentityError, get_user_id


def test_get_user_id_reads_configured_header(settings) -> None:
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            request=SimpleNamespace(headers={"X-OpenWebUI-User-Id": "user-a"})
        )
    )

    assert get_user_id(context, settings) == "user-a"


def test_get_user_id_rejects_missing_header_in_production(settings) -> None:
    context = SimpleNamespace(request_context=SimpleNamespace(request=SimpleNamespace(headers={})))

    with pytest.raises(IdentityError, match="X-OpenWebUI-User-Id"):
        get_user_id(context, settings)
