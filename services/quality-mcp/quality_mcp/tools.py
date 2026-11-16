"""FastMCP transport adapters for the quality service."""

from __future__ import annotations

import asyncio
import logging
import time

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
    async def quality_answer(
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
            task = asyncio.create_task(
                asyncio.to_thread(
                    pipeline.answer,
                    question=question,
                    user_id=user_id,
                    max_iterations=max_iterations,
                )
            )
            try:
                await _await_with_heartbeats(task=task, ctx=ctx, settings=settings, logger=logger)
                return task.result()
            except ValueError as error:
                return f"Invalid request: {error}"
            except asyncio.CancelledError:
                task.cancel()
                raise

    @mcp.tool()
    def quality_status() -> str:
        """Report quality service policy and configured models without exposing secrets."""
        with timed_request_scope(logger):
            return pipeline.status()


async def _await_with_heartbeats(
    *,
    task: asyncio.Task[str],
    ctx: Context,
    settings: Settings,
    logger: logging.Logger,
) -> None:
    """Await a long-running pipeline task while keeping the MCP session stream active.

    The quality loop can run for minutes while no other traffic crosses the
    session's SSE stream. Without periodic server-initiated messages, idle
    read timeouts in intermediate clients or proxies sever the connection and
    the finished answer is lost. Log notifications do not require a progress
    token from the client, so they are the reliable keep-alive primitive.
    """
    started_at = time.monotonic()
    while True:
        done, _ = await asyncio.wait({task}, timeout=settings.heartbeat_interval)
        if done:
            return
        elapsed = int(time.monotonic() - started_at)
        try:
            await ctx.info(
                f"Quality pipeline still working ({elapsed}s elapsed); "
                "this can take several minutes."
            )
            logger.debug("sent heartbeat to MCP client (%ds elapsed)", elapsed)
        except Exception:
            logger.debug("heartbeat notification skipped", exc_info=True)
