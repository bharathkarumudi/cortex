"""Tests that both MCP applications can be assembled without network access."""

from __future__ import annotations

import asyncio

from memory_mcp.app import create_app as create_memory_app
from quality_mcp.app import create_app as create_quality_app


def test_quality_app_factory_does_not_connect_to_chroma(settings) -> None:
    app = create_quality_app(settings)

    assert app.name == "quality-mcp"


def test_memory_app_factory_does_not_connect_to_chroma(settings) -> None:
    app = create_memory_app(settings)

    assert app.name == "memory-mcp"


def test_mcp_context_is_not_exposed_as_a_tool_argument(settings) -> None:
    quality_tools = asyncio.run(create_quality_app(settings).list_tools())
    memory_tools = asyncio.run(create_memory_app(settings).list_tools())

    for tool in [*quality_tools, *memory_tools]:
        assert "ctx" not in tool.inputSchema["properties"]
        assert "ctx" not in tool.inputSchema.get("required", [])
