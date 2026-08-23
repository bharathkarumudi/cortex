"""Tests for request telemetry and logging configuration."""

from __future__ import annotations

import logging

from cortex_common.observability import configure_logging


def test_configure_logging_suppresses_routine_dependency_logs() -> None:
    logger = configure_logging("INFO")

    assert logger.name == "cortex"
    for logger_name in (
        "httpcore",
        "httpx",
        "mcp.server.lowlevel.server",
        "mcp.server.streamable_http",
        "mcp.server.streamable_http_manager",
        "uvicorn.access",
    ):
        assert logging.getLogger(logger_name).level == logging.WARNING
