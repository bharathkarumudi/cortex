"""Fixed-schema, user-scoped Chroma repository for durable chat memory."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

import chromadb

from cortex_common.config import ConfigurationError, Settings
from cortex_common.models import MemoryRecord, MemorySearchHit


class ChromaMemoryRepository:
    """Chroma operations restricted to the configured memory collection."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._client: Any | None = None
        self._collection: Any | None = None
        self._initialization_lock = threading.Lock()

    def initialize(self) -> None:
        """Create or validate the fixed v2 collection before serving traffic."""
        collection = self._get_collection()
        metadata = getattr(collection, "metadata", None) or {}
        if metadata.get("hnsw:space") != "cosine":
            raise ConfigurationError(
                f"collection {self._settings.chroma_collection!r} must use hnsw:space=cosine"
            )

    def search(
        self,
        *,
        vector: list[float],
        user_id: str,
        limit: int,
        include_deleted: bool = False,
    ) -> list[MemorySearchHit]:
        """Run a semantic search that cannot leave the caller's user partition."""
        results = self._get_collection().query(
            query_embeddings=[vector],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
            where=self._where(user_id, include_deleted),
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadata_items = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            MemorySearchHit(
                record=self._record_from_values(memory_id, document, metadata),
                distance=float(distance),
            )
            for memory_id, document, metadata, distance in zip(
                ids, documents, metadata_items, distances, strict=True
            )
        ]

    def get_owned(self, *, memory_id: str, user_id: str) -> MemoryRecord | None:
        """Return an exact record only when it belongs to the requesting user."""
        results = self._get_collection().get(
            ids=[memory_id],
            include=["documents", "metadatas"],
            where=self._where(user_id, include_deleted=True),
        )
        ids = results.get("ids", [])
        if not ids:
            return None
        documents = results.get("documents", [])
        metadata_items = results.get("metadatas", [])
        return self._record_from_values(ids[0], documents[0], metadata_items[0])

    def list_owned(
        self,
        *,
        user_id: str,
        limit: int,
        include_deleted: bool,
        content_type: str | None,
    ) -> list[MemoryRecord]:
        """List bounded caller-owned records using only supported internal filters."""
        where = self._where(user_id, include_deleted)
        if content_type:
            where = {"$and": [where, {"content_type": content_type}]}
        results = self._get_collection().get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        return [
            self._record_from_values(memory_id, document, metadata)
            for memory_id, document, metadata in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
                strict=True,
            )
        ]

    def upsert(self, *, record: MemoryRecord, vector: list[float]) -> None:
        """Store one record atomically by its stable identifier."""
        self._get_collection().upsert(
            ids=[record.memory_id],
            documents=[record.text],
            embeddings=[vector],
            metadatas=[record.metadata()],
        )

    def update(self, record: MemoryRecord) -> None:
        """Update scalar metadata while preserving the document and embedding."""
        self._get_collection().update(ids=[record.memory_id], metadatas=[record.metadata()])

    def counts_for_user(self, *, user_id: str) -> Iterable[MemoryRecord]:
        """Return all caller records for an internal bounded status aggregation."""
        return self.list_owned(
            user_id=user_id,
            limit=self._settings.memory_list_max_results,
            include_deleted=True,
            content_type=None,
        )

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        with self._initialization_lock:
            if self._collection is None:
                self._client = chromadb.HttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._settings.chroma_collection,
                    metadata={"hnsw:space": "cosine"},
                )
                self._logger.info(
                    "connected to memory collection=%s", self._settings.chroma_collection
                )
        return self._collection

    @staticmethod
    def _where(user_id: str, include_deleted: bool) -> dict[str, object]:
        if include_deleted:
            return {"user_id": user_id}
        return {"$and": [{"user_id": user_id}, {"deleted": False}]}

    @staticmethod
    def _record_from_values(
        memory_id: str,
        document: str,
        metadata: dict[str, Any] | None,
    ) -> MemoryRecord:
        values = metadata or {}
        return MemoryRecord(
            memory_id=memory_id,
            text=document,
            user_id=str(values["user_id"]),
            content_type=str(values["content_type"]),
            source=str(values["source"]),
            created_at=str(values["created_at"]),
            updated_at=str(values["updated_at"]),
            access_count=int(values.get("access_count", 0)),
            last_accessed=values.get("last_accessed"),
            deleted=bool(values.get("deleted", False)),
            deleted_at=values.get("deleted_at"),
            delete_reason=values.get("delete_reason"),
            schema_version=int(values.get("schema_version", 2)),
        )
