"""FastMCP transport adapters for user-scoped memory operations."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import Context, FastMCP

from cortex_common.config import Settings
from cortex_common.identity import IdentityError, get_user_id
from cortex_common.memory_service import MemoryNotFoundError, MemoryService
from cortex_common.models import MemoryRecord
from cortex_common.observability import timed_request_scope


def register_tools(
    *,
    mcp: FastMCP,
    memory: MemoryService,
    settings: Settings,
    logger: logging.Logger,
) -> None:
    """Register bounded memory tools that always derive identity from MCP context."""

    @mcp.tool()
    def recall_memory(
        query: str, ctx: Context, limit: int | None = None, use_case: str | None = None
    ) -> str:
        """Search the caller's active long-term memory using semantic similarity.

        Omit use_case to search across all use case buckets, or pass one (for
        example "chat") to restrict results to that bucket.
        """
        with timed_request_scope(logger):
            user_id = _user_id_or_error(ctx, settings, logger)
            if user_id is None:
                return "A verified Open WebUI user identity is required."
            try:
                hits = memory.recall(query=query, user_id=user_id, limit=limit, use_case=use_case)
            except ValueError as error:
                return f"Invalid request: {error}"
            except Exception:
                logger.exception("memory recall failed")
                return "Memory is temporarily unavailable."
            if not hits:
                return "No relevant memories found."
            return "\n".join(
                f"- id={hit.record.memory_id} "
                f"[{hit.record.use_case}/{hit.record.content_type}] {hit.record.text}"
                for hit in hits
            )

    @mcp.tool()
    def save_memory(
        text: str, ctx: Context, content_type: str = "fact", use_case: str = "chat"
    ) -> str:
        """Save an idempotent durable memory for the caller only, tagged with a use case bucket."""
        with timed_request_scope(logger):
            user_id = _user_id_or_error(ctx, settings, logger)
            if user_id is None:
                return "A verified Open WebUI user identity is required."
            try:
                result = memory.save_direct(
                    text=text,
                    user_id=user_id,
                    content_type=content_type,
                    use_case=use_case,
                )
            except ValueError as error:
                return f"Invalid request: {error}"
            except Exception:
                logger.exception("memory save failed")
                return "Memory could not be saved."
            if result.duplicate:
                return f"Memory already exists. id={result.memory_id}"
            return f"Memory saved. id={result.memory_id}"

    @mcp.tool()
    def list_memories(
        ctx: Context,
        limit: int | None = None,
        content_type: str | None = None,
        include_deleted: bool = False,
        use_case: str | None = None,
    ) -> str:
        """List a bounded set of the caller's own memory records and their IDs."""
        with timed_request_scope(logger):
            user_id = _user_id_or_error(ctx, settings, logger)
            if user_id is None:
                return "A verified Open WebUI user identity is required."
            try:
                records = memory.list_memories(
                    user_id=user_id,
                    limit=limit,
                    content_type=content_type,
                    include_deleted=include_deleted,
                    use_case=use_case,
                )
            except ValueError as error:
                return f"Invalid request: {error}"
            except Exception:
                logger.exception("memory list failed")
                return "Memory is temporarily unavailable."
            if not records:
                return "No memories found."
            return "\n".join(_format_record(record) for record in records)

    @mcp.tool()
    def delete_memory(memory_id: str, ctx: Context, reason: str = "") -> str:
        """Soft-delete one caller-owned memory; it remains restorable for the retention period."""
        with timed_request_scope(logger):
            user_id = _user_id_or_error(ctx, settings, logger)
            if user_id is None:
                return "A verified Open WebUI user identity is required."
            try:
                memory.delete(memory_id=memory_id, user_id=user_id, reason=reason)
            except MemoryNotFoundError:
                return "Memory not found."
            except ValueError as error:
                return f"Invalid request: {error}"
            except Exception:
                logger.exception("memory delete failed")
                return "Memory could not be deleted."
            return f"Memory soft-deleted. id={memory_id}"

    @mcp.tool()
    def restore_memory(memory_id: str, ctx: Context) -> str:
        """Restore one caller-owned memory that was previously soft-deleted."""
        with timed_request_scope(logger):
            user_id = _user_id_or_error(ctx, settings, logger)
            if user_id is None:
                return "A verified Open WebUI user identity is required."
            try:
                memory.restore(memory_id=memory_id, user_id=user_id)
            except MemoryNotFoundError:
                return "Memory not found."
            except ValueError as error:
                return f"Invalid request: {error}"
            except Exception:
                logger.exception("memory restore failed")
                return "Memory could not be restored."
            return f"Memory restored. id={memory_id}"

    @mcp.tool()
    def memory_status(ctx: Context) -> str:
        """Report memory health and counts scoped only to the caller."""
        with timed_request_scope(logger):
            user_id = _user_id_or_error(ctx, settings, logger)
            if user_id is None:
                return "A verified Open WebUI user identity is required."
            try:
                status = memory.status(user_id=user_id)
            except Exception:
                logger.exception("memory status failed")
                return "Memory is temporarily unavailable."
            by_type = (
                ", ".join(
                    f"{content_type}={count}"
                    for content_type, count in status.by_content_type.items()
                )
                or "none"
            )
            by_use_case = (
                ", ".join(f"{use_case}={count}" for use_case, count in status.by_use_case.items())
                or "none"
            )
            return "\n".join(
                [
                    "memory-mcp: up",
                    f"active memories: {status.active_count}",
                    f"deleted memories: {status.deleted_count}",
                    f"active by content type: {by_type}",
                    f"active by use case: {by_use_case}",
                ]
            )


def _user_id_or_error(context: Context, settings: Settings, logger: logging.Logger) -> str | None:
    try:
        return get_user_id(context, settings)
    except IdentityError as error:
        logger.warning("memory request rejected: %s", error)
        return None


def _format_record(record: MemoryRecord) -> str:
    deletion_state = "deleted" if record.deleted else "active"
    return (
        f"- id={record.memory_id} [{deletion_state}/{record.use_case}/{record.content_type}] "
        f"created={record.created_at} text={record.text}"
    )
