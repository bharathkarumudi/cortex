"""OpenRouter's OpenAI-compatible chat and embedding gateway."""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import BadRequestError, OpenAI

from cortex_common.config import Settings
from cortex_common.models import ChatUsage
from cortex_common.observability import add_usage


class OpenRouterGateway:
    """One configured OpenRouter client shared within a service process."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float | None = None,
        json_mode: bool = False,
        web_search: bool = False,
    ) -> str:
        """Complete one chat turn and account for provider-reported usage."""
        started_at = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if web_search:
            kwargs["tools"] = self._web_search_tools()

        response: Any = None
        content: str | None = None
        if json_mode:
            try:
                response = self._client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
                content = (response.choices[0].message.content or "").strip()
            except BadRequestError:
                self._logger.debug("json mode rejected by model=%s; retrying without it", model)

        if content is None:
            response = self._client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()

        usage = self._usage(response)
        add_usage(
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost=usage.cost,
            web_searches=usage.web_searches,
        )
        if usage.web_searches:
            self._logger.info(
                "web search: %d query(s) via %s for model=%s",
                usage.web_searches,
                self._settings.web_search_engine,
                model,
            )
        elif web_search:
            self._logger.debug(
                "web search enabled for model=%s but no searches were reported", model
            )
        self._logger.debug(
            "chat model=%s json=%s took %.1fs tokens_in=%d tokens_out=%d cost=$%.5f",
            model,
            json_mode,
            time.perf_counter() - started_at,
            usage.tokens_in,
            usage.tokens_out,
            usage.cost,
        )
        return content

    def embed_one(self, text: str) -> list[float]:
        """Embed one non-empty text value using the configured common model."""
        started_at = time.perf_counter()
        response = self._client.embeddings.create(
            model=self._settings.embedding_model,
            input=[text],
            encoding_format="float",
        )
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        cost = float(getattr(usage, "cost", 0) or 0)
        add_usage(tokens_in=tokens_in, tokens_out=0, cost=cost, web_searches=0)
        self._logger.debug(
            "embed model=%s chars=%d took %.1fs tokens_in=%d cost=$%.5f",
            self._settings.embedding_model,
            len(text),
            time.perf_counter() - started_at,
            tokens_in,
            cost,
        )
        return response.data[0].embedding

    def _web_search_tools(self) -> list[dict[str, object]]:
        return [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": self._settings.web_search_engine,
                    "max_results": self._settings.web_search_max_results,
                    "max_total_results": self._settings.web_search_max_total_results,
                },
            }
        ]

    @staticmethod
    def _usage(response: Any) -> ChatUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return ChatUsage()
        extra = getattr(usage, "model_extra", None) or {}
        server_tool_use = (
            getattr(usage, "server_tool_use_details", None)
            or extra.get("server_tool_use_details")
            or getattr(usage, "server_tool_use", None)
            or extra.get("server_tool_use")
        )
        if isinstance(server_tool_use, dict):
            searches = int(server_tool_use.get("web_search_requests") or 0)
        else:
            searches = int(getattr(server_tool_use, "web_search_requests", 0) or 0)
        return ChatUsage(
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            cost=float(getattr(usage, "cost", 0) or 0),
            web_searches=searches,
        )
