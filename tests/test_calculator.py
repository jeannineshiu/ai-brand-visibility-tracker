"""Unit tests for metrics/calculator.py — mocks query_df so no DB required."""
import pandas as pd
import pytest
from unittest.mock import patch


PATCH = "src.metrics.calculator.query_df"


def _df(**kwargs) -> pd.DataFrame:
    lengths = [len(v) for v in kwargs.values()]
    assert len(set(lengths)) == 1, "all columns must have equal length"
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# visibility_summary
# ---------------------------------------------------------------------------

class TestVisibilitySummary:
    def test_visibility_pct_calculated(self):
        brand_rows = _df(
            brand=["Asana", "Jira"],
            mentions=[10, 5],
            avg_position=[2.0, 3.0],
            pos_rate=[60.0, 40.0],
            neg_rate=[10.0, 20.0],
        )
        total_rows = _df(**{"COUNT(DISTINCT run_id || prompt_id || provider)": [100]})

        with patch(PATCH, side_effect=[brand_rows, total_rows]):
            from src.metrics.calculator import visibility_summary
            df = visibility_summary()

        assert "visibility_pct" in df.columns
        assert abs(df.loc[df["brand"] == "Asana", "visibility_pct"].iloc[0] - 10.0) < 0.01
        assert abs(df.loc[df["brand"] == "Jira", "visibility_pct"].iloc[0] - 5.0) < 0.01

    def test_brand_filter_passes_through(self):
        brand_rows = _df(
            brand=["Asana"],
            mentions=[3],
            avg_position=[1.0],
            pos_rate=[100.0],
            neg_rate=[0.0],
        )
        total_rows = _df(**{"COUNT(DISTINCT run_id || prompt_id || provider)": [10]})

        calls = []
        def capture(sql):
            calls.append(sql)
            return brand_rows if calls else total_rows

        with patch(PATCH, side_effect=[brand_rows, total_rows]):
            from src.metrics.calculator import visibility_summary
            df = visibility_summary(brands=["Asana"])

        assert len(df) == 1

    def test_empty_result_returns_empty_df(self):
        empty = pd.DataFrame(columns=["brand", "mentions", "avg_position", "pos_rate", "neg_rate"])
        total_rows = _df(**{"COUNT(DISTINCT run_id || prompt_id || provider)": [0]})

        with patch(PATCH, side_effect=[empty, total_rows]):
            from src.metrics.calculator import visibility_summary
            df = visibility_summary()

        assert df.empty


# ---------------------------------------------------------------------------
# position_trend
# ---------------------------------------------------------------------------

class TestPositionTrend:
    def test_returns_expected_columns(self):
        rows = _df(
            date=["2026-01-01", "2026-01-02"],
            avg_position=[2.5, 1.8],
            mention_count=[4, 6],
        )
        with patch(PATCH, return_value=rows):
            from src.metrics.calculator import position_trend
            df = position_trend("Asana")

        assert list(df.columns) == ["date", "avg_position", "mention_count"]
        assert len(df) == 2

    def test_empty_brand_returns_empty(self):
        empty = pd.DataFrame(columns=["date", "avg_position", "mention_count"])
        with patch(PATCH, return_value=empty):
            from src.metrics.calculator import position_trend
            df = position_trend("UnknownBrand")

        assert df.empty


# ---------------------------------------------------------------------------
# sentiment_trend
# ---------------------------------------------------------------------------

class TestSentimentTrend:
    def test_rolling_sentiment_column_added(self):
        rows = _df(
            date=["2026-01-01", "2026-01-02", "2026-01-03"],
            sentiment_score=[1.0, -1.0, 1.0],
            mentions=[2, 2, 2],
        )
        with patch(PATCH, return_value=rows):
            from src.metrics.calculator import sentiment_trend
            df = sentiment_trend("Asana", window=3)

        assert "rolling_sentiment" in df.columns
        assert len(df) == 3

    def test_empty_df_skips_rolling(self):
        empty = pd.DataFrame(columns=["date", "sentiment_score", "mentions"])
        with patch(PATCH, return_value=empty):
            from src.metrics.calculator import sentiment_trend
            df = sentiment_trend("Asana")

        assert df.empty
        assert "rolling_sentiment" not in df.columns


# ---------------------------------------------------------------------------
# competitor_gap
# ---------------------------------------------------------------------------

class TestCompetitorGap:
    def test_returns_dataframe(self):
        rows = _df(
            brand=["Notion", "Monday"],
            appears_when_target_absent=[8, 5],
            avg_position=[2.0, 3.5],
        )
        with patch(PATCH, return_value=rows):
            from src.metrics.calculator import competitor_gap
            df = competitor_gap("Asana")

        assert list(df.columns) == ["brand", "appears_when_target_absent", "avg_position"]
        assert df.iloc[0]["brand"] == "Notion"


# ---------------------------------------------------------------------------
# citation_type_breakdown
# ---------------------------------------------------------------------------

class TestCitationTypeBreakdown:
    def test_returns_dataframe(self):
        rows = _df(
            domain_type=["review_site", "tech_media"],
            count=[20, 12],
            unique_domains=[3, 5],
        )
        with patch(PATCH, return_value=rows):
            from src.metrics.calculator import citation_type_breakdown
            df = citation_type_breakdown()

        assert list(df.columns) == ["domain_type", "count", "unique_domains"]
        assert df.iloc[0]["domain_type"] == "review_site"

    def test_empty_citations_returns_empty(self):
        empty = pd.DataFrame(columns=["domain_type", "count", "unique_domains"])
        with patch(PATCH, return_value=empty):
            from src.metrics.calculator import citation_type_breakdown
            df = citation_type_breakdown()

        assert df.empty
