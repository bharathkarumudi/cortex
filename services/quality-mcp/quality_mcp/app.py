"""Application factory and executable entry point for quality-mcp."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from cortex_common.chroma_memory import ChromaMemoryRepository
from cortex_common.config import Settings
from cortex_common.memory_service import MemoryService
from cortex_common.observability import configure_logging
from cortex_common.openrouter import OpenRouterGateway
from quality_mcp.agents import QualityAgents
from quality_mcp.pipeline import QualityPipeline
from quality_mcp.tools import register_tools


def create_app(settings: Settings | None = None) -> FastMCP:
    """Build a quality MCP application without connecting to Chroma at import time."""
    resolved_settings = settings or Settings.from_environment()
    logger = configure_logging(resolved_settings.log_level)
    gateway = OpenRouterGateway(resolved_settings, logger)
    repository = ChromaMemoryRepository(resolved_settings, logger)
    memory = MemoryService(
        repository=repository,
        embeddings=gateway,
        settings=resolved_settings,
    )
    agents = QualityAgents(gateway=gateway, settings=resolved_settings, logger=logger)
    pipeline = QualityPipeline(
        agents=agents,
        memory=memory,
        settings=resolved_settings,
        logger=logger,
    )
    mcp = FastMCP(
        name="quality-mcp",
        host="0.0.0.0",
        port=resolved_settings.quality_mcp_port,
    )
    register_tools(
        mcp=mcp,
        pipeline=pipeline,
        settings=resolved_settings,
        logger=logger,
    )
    return mcp


def main() -> None:
    """Run the Streamable HTTP MCP server."""
    create_app().run(transport="streamable-http")


if __name__ == "__main__":
    main()
