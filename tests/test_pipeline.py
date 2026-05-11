"""Unit tests for analyzer/pipeline.py — mocks all I/O."""
import pytest
from unittest.mock import AsyncMock, patch

from src.analyzer.pipeline import analyze_responses
from src.utils.models import LLMResponse, LLMProvider, Sentiment, PromptConfig


def _make_response(text: str, error: str | None = None) -> LLMResponse:
    return LLMResponse(
        run_id="test-run",
        prompt_id="p1",
        prompt_text="What are the best project management tools?",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        response_text=text,
        latency_ms=100,
        error=error,
    )


PROMPT_CONFIG = PromptConfig(
    prompt_id="p1",
    prompt_text="What are the best project management tools?",
    category="project_management",
    target_brands=["Asana", "Jira", "Linear"],
)


class TestAnalyzeResponses:
    @pytest.fixture
    def mock_judge(self):
        """Patch judge_batch to return positive sentiment for all inputs."""
        with patch(
            "src.analyzer.pipeline.judge_batch",
            new=AsyncMock(return_value=[(Sentiment.POSITIVE, "recommended")]),
        ) as m:
            yield m

    async def test_brand_mention_detected(self, mock_judge):
        mock_judge.return_value = [(Sentiment.POSITIVE, "recommended")]
        responses = [_make_response("Asana is the best tool for teams.")]
        mentions, _ = await analyze_responses(responses, [PROMPT_CONFIG])
        assert len(mentions) == 1
        assert mentions[0].brand == "Asana"

    async def test_sentiment_assigned_from_judge(self, mock_judge):
        mock_judge.return_value = [(Sentiment.NEGATIVE, "criticized")]
        responses = [_make_response("Asana has some issues.")]
        mentions, _ = await analyze_responses(responses, [PROMPT_CONFIG])
        assert mentions[0].sentiment == Sentiment.NEGATIVE

    async def test_error_response_skipped(self, mock_judge):
        responses = [_make_response("", error="API timeout")]
        mentions, citations = await analyze_responses(responses, [PROMPT_CONFIG])
        assert mentions == []
        assert citations == []

    async def test_empty_text_skipped(self, mock_judge):
        responses = [_make_response("")]
        mentions, citations = await analyze_responses(responses, [PROMPT_CONFIG])
        assert mentions == []

    async def test_citation_extracted(self, mock_judge):
        mock_judge.return_value = [(Sentiment.NEUTRAL, "factual")]
        text = "Asana is great. See https://www.g2.com/categories/project-management"
        responses = [_make_response(text)]
        _, citations = await analyze_responses(responses, [PROMPT_CONFIG])
        assert len(citations) == 1
        assert citations[0].domain_type == "review_site"

    async def test_multiple_brands_all_detected(self, mock_judge):
        mock_judge.return_value = [
            (Sentiment.POSITIVE, "good"),
            (Sentiment.NEUTRAL, "ok"),
        ]
        text = "Asana is great. Jira is also popular."
        responses = [_make_response(text)]
        mentions, _ = await analyze_responses(responses, [PROMPT_CONFIG])
        brands = {m.brand for m in mentions}
        assert "Asana" in brands
        assert "Jira" in brands

    async def test_position_reflects_order(self, mock_judge):
        mock_judge.return_value = [
            (Sentiment.POSITIVE, "good"),
            (Sentiment.POSITIVE, "good"),
        ]
        text = "First use Jira, then consider Asana."
        responses = [_make_response(text)]
        mentions, _ = await analyze_responses(responses, [PROMPT_CONFIG])
        by_brand = {m.brand: m.position for m in mentions}
        assert by_brand["Jira"] < by_brand["Asana"]

    async def test_no_brands_in_text(self, mock_judge):
        responses = [_make_response("Nothing useful here.")]
        mentions, _ = await analyze_responses(responses, [PROMPT_CONFIG])
        assert mentions == []
        mock_judge.assert_not_called()
