"""Create and validate the clean v2 Chroma collection on the remote host."""

from __future__ import annotations

from cortex_common.chroma_memory import ChromaMemoryRepository
from cortex_common.config import Settings
from cortex_common.observability import configure_logging


def main() -> None:
    """Initialize only the configured v2 collection; legacy collections are never touched."""
    settings = Settings.from_environment()
    logger = configure_logging(settings.log_level)
    repository = ChromaMemoryRepository(settings, logger)
    repository.initialize()
    logger.info("memory collection is ready: %s", settings.chroma_collection)


if __name__ == "__main__":
    main()
