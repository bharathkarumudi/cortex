"""Concurrency-bounded quality-answer orchestration."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from cortex_common.config import Settings
from cortex_common.memory_service import MemoryService
from cortex_common.models import MemorySearchHit, Verdict


class QualityAgentProtocol(Protocol):
    """Agent capabilities consumed by the orchestration loop."""

    def refine(self, question: str) -> str:
        """Return a refined question."""

    def respond(
        self,
        *,
        question: str,
        memory_context: str,
        feedback: str | None,
        reviewer_suggestion: str,
    ) -> str:
        """Return one candidate answer."""

    def evaluate(self, *, question: str, answer: str) -> Verdict:
        """Review one candidate answer."""

    def summarize(self, *, question: str, answer: str) -> str:
        """Produce a memory-ready summary."""


class QualityPipeline:
    """Run independently-scoped answer workflows with bounded provider concurrency."""

    def __init__(
        self,
        *,
        agents: QualityAgentProtocol,
        memory: MemoryService,
        settings: Settings,
        logger: logging.Logger,
    ) -> None:
        self._agents = agents
        self._memory = memory
        self._settings = settings
        self._logger = logger
        self._semaphore = threading.BoundedSemaphore(settings.max_concurrent_pipelines)

    def answer(
        self,
        *,
        question: str,
        user_id: str,
        max_iterations: int | None = None,
    ) -> str:
        """Run one complete quality pipeline without leaking request state."""
        normalized_question = question.strip()
        if not normalized_question:
            return "Please provide a question."
        effective_iterations = self._effective_iterations(max_iterations)
        with self._semaphore:
            return self._run(
                question=normalized_question,
                user_id=user_id,
                max_iterations=effective_iterations,
            )

    def status(self) -> str:
        """Describe the quality policy without exposing secrets or another user's data."""
        return "\n".join(
            [
                "quality-mcp: up",
                f"refiner: {self._settings.refiner_model}",
                f"responder: {self._settings.responder_model}",
                f"evaluator: {self._settings.evaluator_model}",
                f"memory summarizer: {self._settings.memory_model}",
                f"max concurrent pipelines: {self._settings.max_concurrent_pipelines}",
                f"max iterations: {self._settings.max_iterations}",
                f"agreement score: {self._settings.agreement_score}",
            ]
        )

    def _run(self, *, question: str, user_id: str, max_iterations: int) -> str:
        started_at = time.perf_counter()
        self._logger.info(
            "quality start (user=%s, max_iterations=%d): %s",
            user_id,
            max_iterations,
            self._snippet(question),
        )

        stage_started_at = time.perf_counter()
        refined_question = self._refine_or_original(question)
        self._logger.info(
            "refined in %.1fs -> %s",
            time.perf_counter() - stage_started_at,
            self._snippet(refined_question),
        )

        stage_started_at = time.perf_counter()
        memory_context = self._recall_context(refined_question, user_id)
        self._logger.info(
            "recalled %d memory snippet(s) for user=%s in %.1fs",
            len(memory_context.splitlines()) if memory_context else 0,
            user_id,
            time.perf_counter() - stage_started_at,
        )
        feedback: str | None = None
        reviewer_suggestion = ""
        best_answer = ""
        best_verdict: Verdict | None = None
        previous_score: int | None = None

        for attempt in range(1, max_iterations + 1):
            stage_started_at = time.perf_counter()
            try:
                answer = self._agents.respond(
                    question=refined_question,
                    memory_context=memory_context,
                    feedback=feedback,
                    reviewer_suggestion=reviewer_suggestion,
                )
            except Exception:
                self._logger.exception("responder failed on attempt=%d", attempt)
                break
            self._logger.info(
                "attempt %d/%d drafted in %.1fs (responder=%s) -> reviewing (evaluator=%s)",
                attempt,
                max_iterations,
                time.perf_counter() - stage_started_at,
                self._settings.responder_model,
                self._settings.evaluator_model,
            )

            stage_started_at = time.perf_counter()
            try:
                verdict = self._agents.evaluate(question=refined_question, answer=answer)
            except Exception:
                self._logger.exception("evaluator failed on attempt=%d", attempt)
                if not best_answer:
                    self._logger.info(
                        "quality completed in %.1fs: UNREVIEWED draft returned "
                        "(reviewer unavailable)",
                        time.perf_counter() - started_at,
                    )
                    return f"{answer}\n\n[unreviewed -- reviewer was temporarily unavailable]"
                break
            self._logger.info(
                "attempt %d/%d reviewed in %.1fs: agree=%s score=%d/10 (%s)",
                attempt,
                max_iterations,
                time.perf_counter() - stage_started_at,
                verdict.agree,
                verdict.score,
                verdict.content_type,
            )

            if best_verdict is None or verdict.score > best_verdict.score:
                best_answer = answer
                best_verdict = verdict

            if verdict.agree and verdict.score >= self._settings.agreement_score:
                self._remember(refined_question, answer, verdict, user_id)
                self._logger.info(
                    "quality completed in %.1fs: consensus reached (score=%d, type=%s)",
                    time.perf_counter() - started_at,
                    verdict.score,
                    verdict.content_type,
                )
                return answer

            if previous_score is not None and verdict.score <= previous_score:
                self._logger.info(
                    "review score plateau detected: previous=%d current=%d",
                    previous_score,
                    verdict.score,
                )
                break

            previous_score = verdict.score
            feedback = verdict.feedback
            reviewer_suggestion = verdict.corrected_answer

        if not best_answer or best_verdict is None:
            self._logger.warning(
                "quality failed in %.1fs: no answer produced (upstream errors after retries)",
                time.perf_counter() - started_at,
            )
            return "The answer service is temporarily unavailable. Please try again shortly."

        self._remember(refined_question, best_answer, best_verdict, user_id)
        self._logger.info(
            "quality completed in %.1fs: NO consensus within %d rounds (best score=%d)",
            time.perf_counter() - started_at,
            max_iterations,
            best_verdict.score,
        )
        concern = best_verdict.feedback.strip() or "unspecified"
        return (
            f"{best_answer}\n\n[no reviewer consensus (best score: {best_verdict.score}/10) "
            f"-- open concern: {concern}]"
        )

    def _refine_or_original(self, question: str) -> str:
        try:
            refined_question = self._agents.refine(question).strip()
        except Exception:
            self._logger.exception("refiner failed; using original question")
            return question
        return refined_question or question

    def _recall_context(self, question: str, user_id: str) -> str:
        try:
            hits = self._memory.recall(query=question, user_id=user_id)
        except Exception:
            self._logger.exception("memory recall failed; continuing without context")
            return ""
        return self._format_memory_context(hits)

    def _remember(self, question: str, answer: str, verdict: Verdict, user_id: str) -> None:
        started_at = time.perf_counter()
        try:
            summary = self._agents.summarize(question=question, answer=answer)
            result = self._memory.save_quality(
                summary=summary,
                user_id=user_id,
                content_type=verdict.content_type,
                score=verdict.score,
            )
            duration = time.perf_counter() - started_at
            if result is None:
                self._logger.info(
                    "memory skipped by policy (user=%s, type=%s, score=%d) in %.1fs",
                    user_id,
                    verdict.content_type,
                    verdict.score,
                    duration,
                )
            elif result.duplicate:
                self._logger.info(
                    "memory skipped as duplicate (user=%s, type=%s, id=%s) in %.1fs",
                    user_id,
                    verdict.content_type,
                    result.memory_id,
                    duration,
                )
            else:
                self._logger.info(
                    "memory saved (user=%s, type=%s, score=%d, id=%s) in %.1fs: %s",
                    user_id,
                    verdict.content_type,
                    verdict.score,
                    result.memory_id,
                    duration,
                    self._snippet(summary),
                )
        except Exception:
            self._logger.exception("memory persistence failed; returning answer anyway")

    def _effective_iterations(self, requested: int | None) -> int:
        iterations = self._settings.max_iterations if requested is None else requested
        if not isinstance(iterations, int):
            raise ValueError("max_iterations must be an integer")
        return max(1, min(self._settings.max_iterations_cap, iterations))

    @staticmethod
    def _format_memory_context(hits: list[MemorySearchHit]) -> str:
        return "\n".join(f"- [{hit.record.content_type}] {hit.record.text}" for hit in hits)

    @staticmethod
    def _snippet(text: str, limit: int = 120) -> str:
        normalized = " ".join(text.split())
        return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}..."
