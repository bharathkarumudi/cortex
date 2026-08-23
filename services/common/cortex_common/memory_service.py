"""Memory policy: isolation, idempotent writes, and recoverable deletion."""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from datetime import UTC, datetime
from typing import ClassVar, Protocol

from cortex_common.chroma_memory import ChromaMemoryRepository
from cortex_common.config import Settings
from cortex_common.models import (
    MemoryContentType,
    MemoryRecord,
    MemorySearchHit,
    MemoryStatus,
    MemoryWriteResult,
)


class EmbeddingProvider(Protocol):
    """Minimal capability required by durable memory operations."""

    def embed_one(self, text: str) -> list[float]:
        """Return a vector for one text value."""


class MemoryNotFoundError(LookupError):
    """Raised when a user tries to mutate an unknown or foreign memory ID."""


class UserWriteLockPool:
    """Bounded striped locks that serialize mutations for an individual user.

    One service replica uses these locks to make dedup, write, delete, and
    restore operations race-safe. Hash collisions are harmless: they only
    serialize otherwise unrelated writes briefly.
    """

    def __init__(self, stripes: int = 64) -> None:
        self._locks = [threading.Lock() for _ in range(stripes)]

    def for_user(self, user_id: str) -> threading.Lock:
        return self._locks[hash(user_id) % len(self._locks)]


class MemoryService:
    """Domain service shared by the quality pipeline and memory MCP tools."""

    _DIRECT_TYPES: ClassVar[set[str]] = {"decision", "fact", "note"}

    def __init__(
        self,
        *,
        repository: ChromaMemoryRepository,
        embeddings: EmbeddingProvider,
        settings: Settings,
        write_locks: UserWriteLockPool | None = None,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._settings = settings
        self._write_locks = write_locks or UserWriteLockPool()

    def initialize(self) -> None:
        """Verify the fixed collection contract before accepting requests."""
        self._repository.initialize()

    def recall(
        self,
        *,
        query: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[MemorySearchHit]:
        """Return relevant active memories and update only their caller-owned access metadata."""
        normalized_query = self._require_text(query, "query")
        requested_limit = self._bounded_limit(limit, self._settings.recall_results)
        vector = self._embeddings.embed_one(normalized_query)
        hits = self._repository.search(
            vector=vector,
            user_id=user_id,
            limit=requested_limit,
            include_deleted=False,
        )
        accepted_hits = [hit for hit in hits if hit.distance <= self._settings.recall_max_distance]
        if not accepted_hits:
            return []

        accessed_at = self._now().isoformat()
        with self._write_locks.for_user(user_id):
            for hit in accepted_hits:
                record = hit.record
                self._repository.update(
                    self._replace(
                        record,
                        access_count=record.access_count + 1,
                        last_accessed=accessed_at,
                        updated_at=accessed_at,
                    )
                )
        return accepted_hits

    def save_direct(
        self,
        *,
        text: str,
        user_id: str,
        content_type: str = "fact",
    ) -> MemoryWriteResult:
        """Validate and idempotently store caller-supplied durable context."""
        if content_type not in self._DIRECT_TYPES:
            allowed = ", ".join(sorted(self._DIRECT_TYPES))
            raise ValueError(f"content_type must be one of: {allowed}")
        return self._store(
            text=text,
            user_id=user_id,
            content_type=content_type,
            source="direct",
        )

    def save_quality(
        self,
        *,
        summary: str,
        user_id: str,
        content_type: MemoryContentType,
        score: int,
    ) -> MemoryWriteResult | None:
        """Store an evaluated pipeline summary only when its policy gate passes."""
        threshold = (
            self._settings.decision_score_threshold
            if content_type in {"decision", "fact"}
            else self._settings.memory_score_threshold
        )
        if score < threshold:
            return None
        return self._store(
            text=summary,
            user_id=user_id,
            content_type=content_type,
            source="quality_pipeline",
        )

    def list_memories(
        self,
        *,
        user_id: str,
        limit: int | None,
        content_type: str | None,
        include_deleted: bool,
    ) -> list[MemoryRecord]:
        """List only the calling user's records with bounded output."""
        if content_type is not None and content_type not in {*self._DIRECT_TYPES, "general_chat"}:
            raise ValueError("content_type is not supported")
        return self._repository.list_owned(
            user_id=user_id,
            limit=self._bounded_limit(limit, self._settings.memory_list_max_results),
            include_deleted=include_deleted,
            content_type=content_type,
        )

    def delete(self, *, memory_id: str, user_id: str, reason: str = "") -> MemoryRecord:
        """Soft-delete a caller-owned memory; physical purging is operator-only."""
        return self._set_deleted(memory_id=memory_id, user_id=user_id, deleted=True, reason=reason)

    def restore(self, *, memory_id: str, user_id: str) -> MemoryRecord:
        """Restore a caller-owned soft-deleted memory during its retention period."""
        return self._set_deleted(memory_id=memory_id, user_id=user_id, deleted=False, reason="")

    def status(self, *, user_id: str) -> MemoryStatus:
        """Return aggregate status only for the requesting user's partition."""
        records = list(self._repository.counts_for_user(user_id=user_id))
        by_content_type = Counter(record.content_type for record in records if not record.deleted)
        return MemoryStatus(
            active_count=sum(not record.deleted for record in records),
            deleted_count=sum(record.deleted for record in records),
            by_content_type=dict(sorted(by_content_type.items())),
        )

    def _store(
        self,
        *,
        text: str,
        user_id: str,
        content_type: MemoryContentType,
        source: str,
    ) -> MemoryWriteResult:
        normalized_text = self._require_text(text, "text")
        vector = self._embeddings.embed_one(normalized_text)
        memory_id = self._stable_id(user_id, content_type, normalized_text)
        with self._write_locks.for_user(user_id):
            existing = self._repository.get_owned(memory_id=memory_id, user_id=user_id)
            if existing and not existing.deleted:
                return MemoryWriteResult(memory_id=memory_id, created=False, duplicate=True)

            nearest = self._repository.search(
                vector=vector,
                user_id=user_id,
                limit=1,
                include_deleted=False,
            )
            if nearest and nearest[0].distance <= self._settings.dedup_max_distance:
                return MemoryWriteResult(
                    memory_id=nearest[0].record.memory_id,
                    created=False,
                    duplicate=True,
                )

            record = MemoryRecord.create(
                memory_id=memory_id,
                text=normalized_text,
                user_id=user_id,
                content_type=content_type,
                source=source,
                now=self._now(),
            )
            self._repository.upsert(record=record, vector=vector)
            return MemoryWriteResult(memory_id=memory_id, created=True, duplicate=False)

    def _set_deleted(
        self,
        *,
        memory_id: str,
        user_id: str,
        deleted: bool,
        reason: str,
    ) -> MemoryRecord:
        if not memory_id.strip():
            raise ValueError("memory_id cannot be empty")
        with self._write_locks.for_user(user_id):
            record = self._repository.get_owned(memory_id=memory_id, user_id=user_id)
            if record is None:
                raise MemoryNotFoundError("memory not found")
            now = self._now().isoformat()
            updated_record = self._replace(
                record,
                deleted=deleted,
                deleted_at=now if deleted else None,
                delete_reason=self._require_reason(reason) if deleted else None,
                updated_at=now,
            )
            self._repository.update(updated_record)
            return updated_record

    def _bounded_limit(self, requested_limit: int | None, default: int) -> int:
        limit = default if requested_limit is None else requested_limit
        if not 1 <= limit <= self._settings.memory_list_max_results:
            raise ValueError(
                f"limit must be between 1 and {self._settings.memory_list_max_results}"
            )
        return limit

    @staticmethod
    def _require_text(value: str, field: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError(f"{field} cannot be empty")
        if len(normalized) > 10_000:
            raise ValueError(f"{field} must be 10000 characters or fewer")
        return normalized

    @staticmethod
    def _require_reason(value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) > 500:
            raise ValueError("reason must be 500 characters or fewer")
        return normalized

    @staticmethod
    def _replace(record: MemoryRecord, **changes: object) -> MemoryRecord:
        values = {
            "memory_id": record.memory_id,
            "text": record.text,
            "user_id": record.user_id,
            "content_type": record.content_type,
            "source": record.source,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "access_count": record.access_count,
            "last_accessed": record.last_accessed,
            "deleted": record.deleted,
            "deleted_at": record.deleted_at,
            "delete_reason": record.delete_reason,
            "schema_version": record.schema_version,
        }
        values.update(changes)
        return MemoryRecord(**values)

    @staticmethod
    def _stable_id(user_id: str, content_type: MemoryContentType, text: str) -> str:
        payload = f"v2|{user_id}|{content_type}|{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)
