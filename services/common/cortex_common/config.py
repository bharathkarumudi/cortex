"""Validated configuration shared by the Cortex MCP services."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ


class ConfigurationError(ValueError):
    """Raised when deployment configuration is unsafe or inconsistent."""


def _required(name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required")
    return value


def _integer(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw_value = environ.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number") from error
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw_value = environ.get(name, str(default)).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration for one MCP service instance."""

    openrouter_api_key: str
    openrouter_base_url: str
    chroma_host: str
    chroma_port: int
    chroma_collection: str
    quality_mcp_port: int
    memory_mcp_port: int
    user_id_header: str
    require_user_id_header: bool
    development_user_id: str
    embedding_model: str
    llm_timeout_seconds: float
    llm_max_retries: int
    recall_results: int
    recall_max_distance: float
    dedup_max_distance: float
    memory_list_max_results: int
    memory_delete_retention_days: int
    refiner_model: str
    responder_model: str
    evaluator_model: str
    memory_model: str
    max_iterations: int
    max_iterations_cap: int
    agreement_score: int
    memory_score_threshold: int
    decision_score_threshold: int
    max_concurrent_pipelines: int
    web_search_agents: frozenset[str]
    web_search_engine: str
    web_search_max_results: int
    web_search_max_total_results: int
    log_level: str

    @classmethod
    def from_environment(cls) -> Settings:
        """Load and validate settings from the process environment."""
        require_user_id_header = _boolean("REQUIRE_USER_ID_HEADER", True)
        development_user_id = environ.get("DEFAULT_USER_ID", "development-only").strip()
        if not require_user_id_header and not development_user_id:
            raise ConfigurationError(
                "DEFAULT_USER_ID is required when REQUIRE_USER_ID_HEADER is false"
            )

        agreement_score = _integer("AGREEMENT_SCORE", 9, minimum=1)
        if agreement_score > 10:
            raise ConfigurationError("AGREEMENT_SCORE must be at most 10")

        memory_score_threshold = _integer("MEMORY_SCORE_THRESHOLD", 8, minimum=1)
        decision_score_threshold = _integer("DECISION_SCORE_THRESHOLD", 4, minimum=1)
        if memory_score_threshold > 10 or decision_score_threshold > 10:
            raise ConfigurationError("memory score thresholds must be at most 10")

        recall_max_distance = _float("RECALL_MAX_DISTANCE", 0.60)
        dedup_max_distance = _float("DEDUP_MAX_DISTANCE", 0.05)
        if recall_max_distance > 2 or dedup_max_distance > 2:
            raise ConfigurationError("cosine distances must be between 0 and 2")
        if dedup_max_distance > recall_max_distance:
            raise ConfigurationError("DEDUP_MAX_DISTANCE must not exceed RECALL_MAX_DISTANCE")

        max_iterations = _integer("MAX_ITERATIONS", 3)
        max_iterations_cap = _integer("MAX_ITERATIONS_CAP", 6)
        if max_iterations > max_iterations_cap:
            raise ConfigurationError("MAX_ITERATIONS must not exceed MAX_ITERATIONS_CAP")

        known_search_agents = {"responder", "evaluator"}
        configured_search_agents = frozenset(
            agent.strip().lower()
            for agent in environ.get("WEB_SEARCH_AGENTS", "").split(",")
            if agent.strip()
        )
        unknown_search_agents = configured_search_agents - known_search_agents
        if unknown_search_agents:
            names = ", ".join(sorted(unknown_search_agents))
            raise ConfigurationError(f"unknown WEB_SEARCH_AGENTS values: {names}")

        refiner_model = environ.get("REFINER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        responder_model = environ.get("RESPONDER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        evaluator_model = environ.get("EVALUATOR_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        memory_model = environ.get("MEMORY_MODEL", "").strip() or refiner_model

        return cls(
            openrouter_api_key=_required("OPENAI_API_KEY"),
            openrouter_base_url=environ.get(
                "OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            chroma_host=environ.get("CHROMA_HOST", "chroma").strip(),
            chroma_port=_integer("CHROMA_PORT", 8000),
            chroma_collection=environ.get("CHROMA_MEMORY_COLLECTION", "chat_memory_v2").strip(),
            quality_mcp_port=_integer("QUALITY_MCP_PORT", 8100),
            memory_mcp_port=_integer("MEMORY_MCP_PORT", 8101),
            user_id_header=environ.get("USER_ID_HEADER", "X-OpenWebUI-User-Id").strip(),
            require_user_id_header=require_user_id_header,
            development_user_id=development_user_id,
            embedding_model=environ.get(
                "EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"
            ).strip(),
            llm_timeout_seconds=_float("LLM_TIMEOUT", 60.0, minimum=0.1),
            llm_max_retries=_integer("LLM_MAX_RETRIES", 4, minimum=0),
            recall_results=_integer("RECALL_RESULTS", 3),
            recall_max_distance=recall_max_distance,
            dedup_max_distance=dedup_max_distance,
            memory_list_max_results=_integer("MEMORY_LIST_MAX_RESULTS", 50),
            memory_delete_retention_days=_integer("MEMORY_DELETE_RETENTION_DAYS", 30),
            refiner_model=refiner_model,
            responder_model=responder_model,
            evaluator_model=evaluator_model,
            memory_model=memory_model,
            max_iterations=max_iterations,
            max_iterations_cap=max_iterations_cap,
            agreement_score=agreement_score,
            memory_score_threshold=memory_score_threshold,
            decision_score_threshold=decision_score_threshold,
            max_concurrent_pipelines=_integer("MAX_CONCURRENT_PIPELINES", 2),
            web_search_agents=configured_search_agents,
            web_search_engine=environ.get("WEB_SEARCH_ENGINE", "auto").strip(),
            web_search_max_results=_integer("WEB_SEARCH_MAX_RESULTS", 5),
            web_search_max_total_results=_integer("WEB_SEARCH_MAX_TOTAL_RESULTS", 10),
            log_level=environ.get("LOG_LEVEL", "INFO").upper(),
        )
