"""Tests for quality-pipeline consensus and graceful fallback behavior."""

from __future__ import annotations

import logging

from cortex_common.models import Verdict
from quality_mcp.pipeline import QualityPipeline


class FakeMemory:
    """Offline memory service double for pipeline policy tests."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, int]] = []

    def recall(self, *, query: str, user_id: str):
        del query, user_id
        return []

    def save_quality(self, *, summary: str, user_id: str, content_type: str, score: int) -> None:
        self.saved.append((summary, user_id, content_type, score))


class ApprovingLowScoreAgents:
    """Produces approval before the configured quality threshold, then true consensus."""

    def __init__(self) -> None:
        self.response_calls = 0
        self.verdicts = [
            Verdict(True, 3, "needs more detail", "", "general_chat"),
            Verdict(True, 9, "", "", "general_chat"),
        ]

    def refine(self, question: str) -> str:
        return question

    def respond(
        self,
        *,
        question: str,
        memory_context: str,
        feedback: str | None,
        reviewer_suggestion: str,
    ) -> str:
        del question, memory_context, feedback, reviewer_suggestion
        self.response_calls += 1
        return f"draft-{self.response_calls}"

    def evaluate(self, *, question: str, answer: str) -> Verdict:
        del question, answer
        return self.verdicts.pop(0)

    def summarize(self, *, question: str, answer: str) -> str:
        return f"summary for {question}: {answer}"


def test_pipeline_requires_approval_and_agreement_score(settings) -> None:
    agents = ApprovingLowScoreAgents()
    memory = FakeMemory()
    pipeline = QualityPipeline(
        agents=agents,
        memory=memory,
        settings=settings,
        logger=logging.getLogger("test-quality-pipeline"),
    )

    result = pipeline.answer(question="What changed?", user_id="user-a", max_iterations=2)

    assert result == "draft-2"
    assert agents.response_calls == 2
    assert memory.saved == [("summary for What changed?: draft-2", "user-a", "general_chat", 9)]


def test_pipeline_emits_lifecycle_logs(settings, caplog) -> None:
    logger = logging.getLogger("test-quality-pipeline-logs")
    pipeline = QualityPipeline(
        agents=ApprovingLowScoreAgents(),
        memory=FakeMemory(),
        settings=settings,
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.answer(question="What changed?", user_id="user-a", max_iterations=2)

    messages = "\n".join(caplog.messages)
    assert "quality start (user=user-a, max_iterations=2): What changed?" in messages
    assert "refined in " in messages
    assert "recalled 0 memory snippet(s) for user=user-a in " in messages
    assert "attempt 1/2 drafted in " in messages
    assert "attempt 1/2 reviewed in " in messages
    assert "attempt 2/2 reviewed in " in messages
    assert "memory skipped by policy" in messages
    assert "quality completed in " in messages
