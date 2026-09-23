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
        use_case: str | None = None,
        include_deleted: bool = False,
    ) -> list[MemorySearchHit]:
        del vector
        return [
            hit
            for hit in self.search_hits
            if hit.record.user_id == user_id
            and (include_deleted or not hit.record.deleted)
            and (use_case is None or hit.record.use_case == use_case)
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
        use_case: str | None = None,
    ) -> list[MemoryRecord]:
        return [
            record
            for record in self.records.values()
            if record.user_id == user_id
            and (include_deleted or not record.deleted)
            and (content_type is None or record.content_type == content_type)
            and (use_case is None or record.use_case == use_case)
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


def test_same_text_in_different_use_cases_creates_distinct_records(settings) -> None:
    service, repository = _service(settings)

    chat_result = service.save_direct(text="I value privacy.", user_id="user-a")
    twin_result = service.save_direct(
        text="I value privacy.", user_id="user-a", use_case="digital_twin"
    )

    assert chat_result.created and twin_result.created
    assert chat_result.memory_id != twin_result.memory_id
    assert len(repository.records) == 2


def test_same_text_in_same_use_case_is_duplicate(settings) -> None:
    service, repository = _service(settings)

    first = service.save_direct(text="I value privacy.", user_id="user-a")
    second = service.save_direct(text="I value privacy.", user_id="user-a")

    assert first.created
    assert second.duplicate and not second.created
    assert second.memory_id == first.memory_id
    assert len(repository.records) == 1


def test_recall_scoped_to_use_case_and_global_by_default(settings) -> None:
    service, repository = _service(settings)
    chat_saved = service.save_direct(text="Chat memory", user_id="user-a")
    twin_saved = service.save_direct(text="Twin memory", user_id="user-a", use_case="digital_twin")
    repository.search_hits = [
        MemorySearchHit(record=repository.records[chat_saved.memory_id], distance=0.1),
        MemorySearchHit(record=repository.records[twin_saved.memory_id], distance=0.1),
    ]

    chat_hits = service.recall(query="memory", user_id="user-a", use_case="chat")
    assert [hit.record.memory_id for hit in chat_hits] == [chat_saved.memory_id]

    all_hits = service.recall(query="memory", user_id="user-a")
    assert {hit.record.memory_id for hit in all_hits} == {
        chat_saved.memory_id,
        twin_saved.memory_id,
    }


def test_recall_preserves_use_case_on_access_update(settings) -> None:
    service, repository = _service(settings)
    saved = service.save_direct(text="Twin memory", user_id="user-a", use_case="digital_twin")
    repository.search_hits = [
        MemorySearchHit(record=repository.records[saved.memory_id], distance=0.1)
    ]

    service.recall(query="twin", user_id="user-a")

    updated = repository.records[saved.memory_id]
    assert updated.use_case == "digital_twin"
    assert updated.access_count == 1


def test_invalid_use_case_is_rejected(settings) -> None:
    service, _ = _service(settings)

    with pytest.raises(ValueError, match="use_case"):
        service.save_direct(text="fact", user_id="user-a", use_case="unknown")
    with pytest.raises(ValueError, match="use_case"):
        service.recall(query="fact", user_id="user-a", use_case="unknown")
    with pytest.raises(ValueError, match="use_case"):
        service.list_memories(
            user_id="user-a",
            limit=None,
            content_type=None,
            include_deleted=False,
            use_case="unknown",
        )


def test_status_aggregates_by_use_case(settings) -> None:
    service, _ = _service(settings)
    service.save_direct(text="Chat fact", user_id="user-a")
    service.save_direct(text="Twin fact", user_id="user-a", use_case="digital_twin")

    status = service.status(user_id="user-a")

    assert status.active_count == 2
    assert status.by_use_case == {"chat": 1, "digital_twin": 1}
