"""FastMCP transport adapters for the quality service."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import Context, FastMCP

from cortex_common.config import Settings
from cortex_common.identity import IdentityError, get_user_id
from cortex_common.observability import timed_request_scope
from quality_mcp.pipeline import QualityPipeline


def register_tools(
    *,
    mcp: FastMCP,
    pipeline: QualityPipeline,
    settings: Settings,
    logger: logging.Logger,
) -> None:
    """Register the small public quality tool surface."""

    @mcp.tool()
    def quality_answer(
        question: str,
        ctx: Context,
        max_iterations: int | None = None,
    ) -> str:
        """Generate a reviewed answer using user-scoped long-term memory."""
        with timed_request_scope(logger):
            try:
                user_id = get_user_id(ctx, settings)
            except IdentityError as error:
                logger.warning("quality request rejected: %s", error)
                return "A verified Open WebUI user identity is required."
            try:
                return pipeline.answer(
                    question=question,
                    user_id=user_id,
                    max_iterations=max_iterations,
                )
            except ValueError as error:
                return f"Invalid request: {error}"

    @mcp.tool()
    def quality_status() -> str:
        """Report quality service policy and configured models without exposing secrets."""
        with timed_request_scope(logger):
            return pipeline.status()
