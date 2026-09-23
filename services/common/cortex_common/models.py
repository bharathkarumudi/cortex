"""Domain models shared by the quality and memory MCP services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

MemoryContentType = Literal["decision", "fact", "general_chat", "note"]
PipelineContentType = Literal["decision", "fact", "general_chat"]
MemoryUseCase = Literal["chat", "digital_twin"]
DEFAULT_USE_CASE: MemoryUseCase = "chat"


def memory_use_case(value: object) -> MemoryUseCase:
    """Narrow a stored metadata value to a known use case, defaulting to chat."""
    if value == "digital_twin":
        return "digital_twin"
    return "chat"


@dataclass(frozen=True, slots=True)
class Verdict:
    """A review outcome for one generated answer."""

    agree: bool
    score: int
    feedback: str
    corrected_answer: str
    content_type: PipelineContentType


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A durable, user-owned memory record."""

    memory_id: str
    text: str
    user_id: str
    content_type: MemoryContentType
    source: str
    created_at: str
    updated_at: str
    access_count: int
    last_accessed: str | None
    deleted: bool
    deleted_at: str | None
    delete_reason: str | None
    schema_version: int = 2
    use_case: MemoryUseCase = DEFAULT_USE_CASE

    @classmethod
    def create(
        cls,
        *,
        memory_id: str,
        text: str,
        user_id: str,
        content_type: MemoryContentType,
        source: str,
        now: datetime,
        use_case: MemoryUseCase = DEFAULT_USE_CASE,
    ) -> MemoryRecord:
        """Create a new active record using UTC ISO timestamps."""
        timestamp = now.isoformat()
        return cls(
            memory_id=memory_id,
            text=text,
            user_id=user_id,
            content_type=content_type,
            source=source,
            created_at=timestamp,
            updated_at=timestamp,
            access_count=0,
            last_accessed=None,
            deleted=False,
            deleted_at=None,
            delete_reason=None,
            schema_version=3,
            use_case=use_case,
        )

    def metadata(self) -> dict[str, bool | int | str]:
        """Return only Chroma-compatible scalar metadata."""
        metadata: dict[str, bool | int | str] = {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "use_case": self.use_case,
            "content_type": self.content_type,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "deleted": self.deleted,
        }
        if self.last_accessed is not None:
            metadata["last_accessed"] = self.last_accessed
        if self.deleted_at is not None:
            metadata["deleted_at"] = self.deleted_at
        if self.delete_reason is not None:
            metadata["delete_reason"] = self.delete_reason
        return metadata


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    """One semantic-memory search result."""

    record: MemoryRecord
    distance: float


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Outcome of an idempotent direct or pipeline memory write."""

    memory_id: str
    created: bool
    duplicate: bool


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """Caller-scoped memory health and record counts."""

    active_count: int
    deleted_count: int
    by_content_type: dict[str, int]
    by_use_case: dict[str, int]


@dataclass(frozen=True, slots=True)
class ChatUsage:
    """Token and provider-reported cost information for one model call."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    web_searches: int = 0
