"""Tests for user isolation, idempotency, and recoverable memory mutation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from cortex_common.memory_service import MemoryNotFoundError, MemoryService
from cortex_common.models import MemoryRecord, MemorySearchHit


class FakeEmbeddings:
    """Offline deterministic embedding provider for memory-service tests."""

    def embed_one(self, text: str) -> list[float]:
        return [float(len(text))]


class FakeRepository:
    """In-memory repository with the same user-bound behavior as the real adapter."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self.search_hits: list[MemorySearchHit] = []

    def initialize(self) -> None:
        return None

    def search(
        self,
        *,
        vector: list[float],
        user_id: str,
        limit: int,
        include_deleted: bool,
    ) -> list[MemorySearchHit]:
        del vector
        return [
            hit
            for hit in self.search_hits
            if hit.record.user_id == user_id and (include_deleted or not hit.record.deleted)
        ][:limit]

    def get_owned(self, *, memory_id: str, user_id: str) -> MemoryRecord | None:
        record = self.records.get(memory_id)
        return record if record and record.user_id == user_id else None

    def list_owned(
        self,
        *,
        user_id: str,
        limit: int,
        include_deleted: bool,
        content_type: str | None,
    ) -> list[MemoryRecord]:
        return [
            record
            for record in self.records.values()
            if record.user_id == user_id
            and (include_deleted or not record.deleted)
            and (content_type is None or record.content_type == content_type)
        ][:limit]

    def upsert(self, *, record: MemoryRecord, vector: list[float]) -> None:
        del vector
        self.records[record.memory_id] = record

    def update(self, record: MemoryRecord) -> None:
        self.records[record.memory_id] = record

    def counts_for_user(self, *, user_id: str):
        return self.list_owned(
            user_id=user_id,
            limit=100,
            include_deleted=True,
            content_type=None,
        )


def _service(settings) -> tuple[MemoryService, FakeRepository]:
    repository = FakeRepository()
    return (
        MemoryService(repository=repository, embeddings=FakeEmbeddings(), settings=settings),
        repository,
    )


def test_direct_write_is_idempotent_under_same_user_concurrency(settings) -> None:
    service, repository = _service(settings)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: service.save_direct(
                    text="I prefer concise answers.",
                    user_id="user-a",
                ),
                range(8),
            )
        )

    assert sum(result.created for result in results) == 1
    assert len(repository.records) == 1
    assert {result.memory_id for result in results} == set(repository.records)


def test_users_cannot_list_or_delete_each_others_memories(settings) -> None:
    service, _ = _service(settings)
    saved = service.save_direct(text="User A's preference", user_id="user-a")

    assert (
        service.list_memories(
            user_id="user-b",
            limit=None,
            content_type=None,
            include_deleted=False,
        )
        == []
    )
    with pytest.raises(MemoryNotFoundError):
        service.delete(memory_id=saved.memory_id, user_id="user-b")


def test_soft_delete_hides_memory_and_restore_returns_it(settings) -> None:
    service, _ = _service(settings)
    saved = service.save_direct(
        text="Remember this decision",
        user_id="user-a",
        content_type="decision",
    )

    deleted = service.delete(
        memory_id=saved.memory_id,
        user_id="user-a",
        reason="No longer current",
    )
    assert deleted.deleted is True
    assert (
        service.list_memories(
            user_id="user-a",
            limit=None,
            content_type=None,
            include_deleted=False,
        )
        == []
    )

    restored = service.restore(memory_id=saved.memory_id, user_id="user-a")
    assert restored.deleted is False
    assert [
        record.memory_id
        for record in service.list_memories(
            user_id="user-a",
            limit=None,
            content_type=None,
            include_deleted=False,
        )
    ] == [saved.memory_id]


def test_recall_updates_only_matching_user_access_metadata(settings) -> None:
    service, repository = _service(settings)
    saved = service.save_direct(text="Private memory", user_id="user-a")
    record = repository.records[saved.memory_id]
    repository.search_hits = [MemorySearchHit(record=record, distance=0.1)]

    hits = service.recall(query="private", user_id="user-a")

    assert len(hits) == 1
    assert repository.records[saved.memory_id].access_count == 1
