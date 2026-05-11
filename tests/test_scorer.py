"""Unit tests for recommender/scorer.py"""
import pytest
import pandas as pd
from src.recommender.scorer import (
    compute_opportunity_scores,
    DOMAIN_TYPE_WEIGHTS,
    Recommendation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source_features(rows: list[dict]) -> pd.DataFrame:
    cols = ["domain_type", "brand", "cooccurrence_count",
            "avg_position_with_source", "citation_count", "unique_domains"]
    return pd.DataFrame(rows, columns=cols)


def _make_brand_features(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fallback (no citation data)
# ---------------------------------------------------------------------------

class TestFallbackRecommendations:
    def test_returns_all_domain_types(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        returned_types = {r.domain_type for r in recs}
        assert returned_types == set(DOMAIN_TYPE_WEIGHTS.keys())

    def test_sorted_by_score_descending(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        scores = [r.opportunity_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_review_site_highest(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        assert recs[0].domain_type == "review_site"

    def test_priority_high_for_top_results(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        top = next(r for r in recs if r.domain_type == "review_site")
        assert top.priority == "HIGH"

    def test_priority_low_for_bottom_results(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        bottom = next(r for r in recs if r.domain_type == "other")
        assert bottom.priority == "LOW"


# ---------------------------------------------------------------------------
# Score calculation with data
# ---------------------------------------------------------------------------

class TestScoreCalculation:
    def _base_features(self):
        return _make_source_features([
            {"domain_type": "review_site", "brand": "Asana",
             "cooccurrence_count": 1, "avg_position_with_source": 2.0,
             "citation_count": 5, "unique_domains": 3},
            {"domain_type": "review_site", "brand": "Jira",
             "cooccurrence_count": 5, "avg_position_with_source": 1.5,
             "citation_count": 10, "unique_domains": 5},
        ])

    def test_competitor_gap_raises_score(self):
        # Jira appears 5× in review_site, Asana only 1× → high gap → high score
        features = self._base_features()
        recs = compute_opportunity_scores(features, pd.DataFrame(), "Asana", ["Jira"])
        review_rec = next(r for r in recs if r.domain_type == "review_site")
        # Base = 1.0 * 40 = 40, gap = 5-1=4 → comp_bonus = min(30, 40) = 30 → at least 70
        assert review_rec.opportunity_score >= 70

    def test_no_gap_lower_score(self):
        # Asana and Jira equal co-occurrence → no gap bonus
        features = _make_source_features([
            {"domain_type": "review_site", "brand": "Asana",
             "cooccurrence_count": 5, "avg_position_with_source": 2.0,
             "citation_count": 5, "unique_domains": 3},
            {"domain_type": "review_site", "brand": "Jira",
             "cooccurrence_count": 5, "avg_position_with_source": 1.5,
             "citation_count": 5, "unique_domains": 3},
        ])
        recs = compute_opportunity_scores(features, pd.DataFrame(), "Asana", ["Jira"])
        review_rec = next(r for r in recs if r.domain_type == "review_site")
        # No gap bonus → score ≤ 40 + 0 + freq_bonus + pos_bonus
        assert review_rec.opportunity_score < 70

    def test_score_capped_at_100(self):
        features = _make_source_features([
            {"domain_type": "review_site", "brand": "Asana",
             "cooccurrence_count": 0, "avg_position_with_source": 1.0,
             "citation_count": 100, "unique_domains": 50},
            {"domain_type": "review_site", "brand": "Jira",
             "cooccurrence_count": 100, "avg_position_with_source": 1.0,
             "citation_count": 100, "unique_domains": 50},
        ])
        recs = compute_opportunity_scores(features, pd.DataFrame(), "Asana", ["Jira"])
        for r in recs:
            assert r.opportunity_score <= 100

    def test_output_is_recommendation_objects(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_action_not_empty(self):
        recs = compute_opportunity_scores(
            pd.DataFrame(), pd.DataFrame(), "Asana", ["Jira"]
        )
        assert all(len(r.action) > 0 for r in recs)
