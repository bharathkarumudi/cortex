"""Request-scoped correlation and model-cost accounting."""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import uuid4


@dataclass(slots=True)
class CostTotals:
    """Mutable usage totals that exist only within one request context."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    web_searches: int = 0


_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_cost_totals: contextvars.ContextVar[CostTotals | None] = contextvars.ContextVar(
    "cost_totals", default=None
)

_ROUTINE_LOGGERS = (
    "httpcore",
    "httpx",
    "mcp.server.lowlevel.server",
    "mcp.server.streamable_http",
    "mcp.server.streamable_http_manager",
    "uvicorn.access",
)


class RequestIdFilter(logging.Filter):
    """Add the current request ID to records produced by this application's loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def configure_logging(level: str) -> logging.Logger:
    """Configure and return the Cortex application logger once per process."""
    logger = logging.getLogger("cortex")
    logger.setLevel(level)
    logger.propagate = False
    for logger_name in _ROUTINE_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s.%(funcName)s "
            "[%(request_id)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    return logger


@contextmanager
def request_scope() -> Iterator[CostTotals]:
    """Set and reliably reset request-specific state around one MCP invocation."""
    request_token = _request_id.set(uuid4().hex[:8])
    totals = CostTotals()
    totals_token = _cost_totals.set(totals)
    try:
        yield totals
    finally:
        _cost_totals.reset(totals_token)
        _request_id.reset(request_token)


def add_usage(*, tokens_in: int, tokens_out: int, cost: float, web_searches: int) -> None:
    """Accumulate provider usage only when executing inside a request scope."""
    totals = _cost_totals.get()
    if totals is None:
        return
    totals.tokens_in += tokens_in
    totals.tokens_out += tokens_out
    totals.cost += cost
    totals.web_searches += web_searches


def log_cost_summary(logger: logging.Logger, elapsed_seconds: float) -> None:
    """Log request-level model cost without recording prompt or answer content."""
    totals = _cost_totals.get()
    if totals is None:
        return
    logger.info(
        "request completed in %.1fs: tokens_in=%d tokens_out=%d cost=$%.5f web_searches=%d",
        elapsed_seconds,
        totals.tokens_in,
        totals.tokens_out,
        totals.cost,
        totals.web_searches,
    )


@contextmanager
def timed_request_scope(logger: logging.Logger) -> Iterator[CostTotals]:
    """Convenience request scope that emits a cost summary on completion."""
    started_at = time.perf_counter()
    with request_scope() as totals:
        try:
            yield totals
        finally:
            log_cost_summary(logger, time.perf_counter() - started_at)
