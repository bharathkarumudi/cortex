"""Tests for quality-tool transport behavior: SSE keep-alive heartbeats."""

from __future__ import annotations

import asyncio
import dataclasses
import logging

from quality_mcp.tools import _await_with_heartbeats


class RecordingContext:
    """Minimal stand-in for FastMCP's Context: records info notifications."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def info(self, message: str, **extra: object) -> None:
        del extra
        self.messages.append(message)


class FaultyContext:
    """Context whose notifications fail, simulating a severed session."""

    async def info(self, message: str, **extra: object) -> None:
        del message, extra
        raise RuntimeError("session gone")


def test_heartbeats_stream_while_task_runs(settings) -> None:
    fast_settings = dataclasses.replace(settings, heartbeat_interval=0.05)
    ctx = RecordingContext()
    logger = logging.getLogger("test-quality-heartbeat")

    async def scenario() -> str:
        async def work() -> str:
            await asyncio.sleep(0.25)
            return "done"

        task = asyncio.create_task(work())
        await _await_with_heartbeats(  # type: ignore[arg-type]
            task=task, ctx=ctx, settings=fast_settings, logger=logger
        )
        return task.result()

    assert asyncio.run(scenario()) == "done"
    assert len(ctx.messages) >= 3
    assert all("still working" in message for message in ctx.messages)


def test_heartbeats_stop_immediately_when_task_finishes_fast(settings) -> None:
    slow_settings = dataclasses.replace(settings, heartbeat_interval=0.5)
    ctx = RecordingContext()
    logger = logging.getLogger("test-quality-heartbeat-fast")

    async def scenario() -> str:
        async def work() -> str:
            return "instant"

        task = asyncio.create_task(work())
        await _await_with_heartbeats(  # type: ignore[arg-type]
            task=task, ctx=ctx, settings=slow_settings, logger=logger
        )
        return task.result()

    assert asyncio.run(scenario()) == "instant"
    assert ctx.messages == []


def test_heartbeat_failures_do_not_break_the_tool(settings) -> None:
    fast_settings = dataclasses.replace(settings, heartbeat_interval=0.05)
    logger = logging.getLogger("test-quality-heartbeat-fault")

    async def scenario() -> str:
        async def work() -> str:
            await asyncio.sleep(0.12)
            return "survived"

        task = asyncio.create_task(work())
        await _await_with_heartbeats(  # type: ignore[arg-type]
            task=task, ctx=FaultyContext(), settings=fast_settings, logger=logger
        )
        return task.result()

    assert asyncio.run(scenario()) == "survived"
