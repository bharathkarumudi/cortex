"""Trusted Open WebUI identity extraction for MCP tool adapters."""

from __future__ import annotations

from typing import Any

from cortex_common.config import Settings


class IdentityError(PermissionError):
    """Raised when an MCP request does not contain a valid caller identity."""


def get_user_id(context: Any, settings: Settings) -> str:
    """Get the caller identity from the configured Open WebUI request header.

    Tool schemas intentionally do not expose a user identifier. This keeps the
    user boundary in server-controlled request metadata rather than caller data.
    """
    request = getattr(getattr(context, "request_context", None), "request", None)
    headers = getattr(request, "headers", None)
    user_id = headers.get(settings.user_id_header, "").strip() if headers else ""
    if user_id:
        return user_id
    if settings.require_user_id_header:
        raise IdentityError(f"missing required identity header: {settings.user_id_header}")
    return settings.development_user_id
