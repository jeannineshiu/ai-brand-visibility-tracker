"""Unit tests for prompt_runner/vote.py"""
import pytest
from src.prompt_runner.vote import majority_vote, VotedMention
from src.utils.models import LLMResponse, LLMProvider, PromptConfig


PROMPT_CONFIG = PromptConfig(
    prompt_id="p1",
    prompt_text="What are the best project management tools?",
    category="project_management",
    target_brands=["Asana", "Jira", "Linear"],
)


def _resp(text: str, trial: int = 0) -> LLMResponse:
    return LLMResponse(
        run_id=f"base-t{trial}",
        prompt_id="p1",
        prompt_text="What are the best project management tools?",
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        response_text=text,
        latency_ms=100,
    )


class TestMajorityVote:
    def test_brand_in_all_runs_kept(self):
        responses = [
            _resp("Asana is the best tool.", trial=0),
            _resp("Asana is a great tool.", trial=1),
            _resp("Asana is highly recommended.", trial=2),
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        assert len(result) == 1
        _, mentions = result[0]
        brands = {m.brand for m in mentions}
        assert "Asana" in brands

    def test_brand_in_minority_filtered_out(self):
        # Asana appears in 1/3 runs → should be filtered (threshold >50%)
        responses = [
            _resp("Asana is great.", trial=0),
            _resp("Nothing useful here.", trial=1),
            _resp("Nothing useful here.", trial=2),
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        _, mentions = result[0]
        brands = {m.brand for m in mentions}
        assert "Asana" not in brands

    def test_brand_in_majority_kept(self):
        # Asana in 2/3 → kept; Linear in 1/3 → filtered
        responses = [
            _resp("Asana is great. Linear is fast.", trial=0),
            _resp("Asana is recommended.", trial=1),
            _resp("No tools mentioned.", trial=2),
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        _, mentions = result[0]
        brands = {m.brand for m in mentions}
        assert "Asana" in brands
        assert "Linear" not in brands

    def test_appearance_rate_correct(self):
        # Asana in 2/3 runs → appearance_rate = 0.67
        responses = [
            _resp("Asana is great.", trial=0),
            _resp("Asana is good.", trial=1),
            _resp("No tools mentioned.", trial=2),
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        _, mentions = result[0]
        asana = next(m for m in mentions if m.brand == "Asana")
        assert abs(asana.appearance_rate - 2 / 3) < 0.01

    def test_position_is_median(self):
        # Asana: position 1 in run 0, position 2 in run 1 → median = 1 or 2
        responses = [
            _resp("Asana first, Jira second.", trial=0),   # Asana pos=1
            _resp("Jira first, Asana second.", trial=1),   # Asana pos=2
            _resp("Asana is the leader.", trial=2),        # Asana pos=1
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        _, mentions = result[0]
        asana = next(m for m in mentions if m.brand == "Asana")
        assert asana.position in (1, 2)

    def test_positions_are_1based_consecutive(self):
        responses = [
            _resp("Jira is good. Asana is great.", trial=0),
            _resp("Jira is good. Asana is great.", trial=1),
            _resp("Jira is good. Asana is great.", trial=2),
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        _, mentions = result[0]
        positions = sorted(m.position for m in mentions)
        assert positions == list(range(1, len(mentions) + 1))

    def test_representative_response_is_middle_trial(self):
        responses = [
            _resp("Asana text.", trial=0),
            _resp("Asana text.", trial=1),
            _resp("Asana text.", trial=2),
        ]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        rep, _ = result[0]
        assert rep.run_id == "base-t1"  # middle of 3

    def test_error_responses_excluded(self):
        ok = _resp("Asana is great.", trial=0)
        ok2 = _resp("Asana is great.", trial=1)
        err = LLMResponse(
            run_id="base-t2", prompt_id="p1",
            prompt_text="...", provider=LLMProvider.OPENAI,
            model="gpt-4o-mini", response_text="", latency_ms=0, error="timeout",
        )
        # Only 2 valid runs → need 1 out of 2 to pass threshold
        result = majority_vote([ok, ok2, err], [PROMPT_CONFIG], n_runs=3)
        assert len(result) == 1

    def test_returns_voted_mention_type(self):
        responses = [_resp("Asana is great.", trial=i) for i in range(3)]
        result = majority_vote(responses, [PROMPT_CONFIG], n_runs=3)
        _, mentions = result[0]
        assert all(isinstance(m, VotedMention) for m in mentions)

    def test_empty_responses_returns_empty(self):
        result = majority_vote([], [PROMPT_CONFIG], n_runs=3)
        assert result == []
