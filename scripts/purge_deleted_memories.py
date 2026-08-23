"""Operator-only permanent purge for soft-deleted records past retention."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import chromadb

from cortex_common.config import Settings
from cortex_common.observability import configure_logging


def main() -> None:
    """Preview or permanently delete records that passed the configured retention period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="perform deletion; without this flag the command is a dry run",
    )
    arguments = parser.parse_args()

    settings = Settings.from_environment()
    logger = configure_logging(settings.log_level)
    cutoff = datetime.now(UTC) - timedelta(days=settings.memory_delete_retention_days)
    collection = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    ).get_collection(settings.chroma_collection)
    results = collection.get(
        where={"deleted": True},
        include=["metadatas"],
    )
    expired_ids = [
        memory_id
        for memory_id, metadata in zip(
            results.get("ids", []),
            results.get("metadatas", []),
            strict=True,
        )
        if _is_expired(metadata or {}, cutoff)
    ]
    if not expired_ids:
        logger.info("no soft-deleted records have reached the retention cutoff")
        return
    if not arguments.confirm:
        logger.warning("dry run: %d records would be permanently deleted", len(expired_ids))
        return
    collection.delete(ids=expired_ids)
    logger.warning("permanently deleted %d records past retention", len(expired_ids))


def _is_expired(metadata: dict[str, object], cutoff: datetime) -> bool:
    raw_deleted_at = metadata.get("deleted_at")
    if not isinstance(raw_deleted_at, str):
        return False
    try:
        deleted_at = datetime.fromisoformat(raw_deleted_at)
    except ValueError:
        return False
    if deleted_at.tzinfo is None:
        return False
    return deleted_at <= cutoff


if __name__ == "__main__":
    main()
