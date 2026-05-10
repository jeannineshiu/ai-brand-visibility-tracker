"""Computes visibility metrics from stored brand mention data."""
import pandas as pd
from src.storage.store import query_df, _USE_BIGQUERY
from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET


def _tbl(name: str) -> str:
    """Return fully-qualified table name for the active backend."""
    if _USE_BIGQUERY:
        return f"`{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{name}`"
    return name


def _sentiment_count(value: str) -> str:
    """SQL fragment for counting a sentiment value — BigQuery vs SQLite differ."""
    if _USE_BIGQUERY:
        return f"COUNTIF(sentiment = '{value}')"
    return f"SUM(sentiment = '{value}')"


def _distinct_concat(col_a: str, col_b: str) -> str:
    """COUNT DISTINCT of two columns concatenated."""
    if _USE_BIGQUERY:
        return f"COUNT(DISTINCT CONCAT({col_a}, '|', {col_b}))"
    return f"COUNT(DISTINCT {col_a} || '|' || {col_b})"


def visibility_summary(brands: list[str] | None = None) -> pd.DataFrame:
    """
    Per-brand visibility across all runs.
    Columns: brand, mentions, avg_position, pos_rate, neg_rate, visibility_pct
    """
    brand_filter = ""
    if brands:
        quoted = ", ".join(f"'{b}'" for b in brands)
        brand_filter = f"WHERE brand IN ({quoted})"

    bm = _tbl("brand_mentions")
    sql = f"""
    SELECT
        brand,
        COUNT(*)                                     AS mentions,
        AVG(position)                                AS avg_position,
        ROUND(100.0 * {_sentiment_count('positive')} / COUNT(*), 1) AS pos_rate,
        ROUND(100.0 * {_sentiment_count('negative')} / COUNT(*), 1) AS neg_rate
    FROM {bm}
    {brand_filter}
    GROUP BY brand
    ORDER BY mentions DESC
    """
    df = query_df(sql)

    # Denominator = total unique (run_id, prompt_id, provider) slots queried
    # Using llm_responses avoids inflation when brand_mentions has duplicates across runs
    lr = _tbl("llm_responses")
    if _USE_BIGQUERY:
        total_sql = f"SELECT COUNT(DISTINCT CONCAT(run_id, prompt_id, provider)) FROM {lr}"
    else:
        total_sql = f"SELECT COUNT(DISTINCT run_id || prompt_id || provider) FROM {lr}"
    total_df = query_df(total_sql)
    total = int(total_df.iloc[0, 0]) if not total_df.empty else 1

    df["visibility_pct"] = (df["mentions"] / total * 100).round(1)
    return df


def position_trend(brand: str) -> pd.DataFrame:
    """Daily average position for a brand over time."""
    bm = _tbl("brand_mentions")
    sql = f"""
    SELECT
        DATE(created_at)  AS date,
        AVG(position)     AS avg_position,
        COUNT(*)          AS mention_count
    FROM {bm}
    WHERE brand = '{brand}'
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    return query_df(sql)


def sentiment_trend(brand: str, window: int = 7) -> pd.DataFrame:
    """Daily sentiment score with rolling average. Score: +1/0/-1"""
    bm = _tbl("brand_mentions")
    sql = f"""
    SELECT
        DATE(created_at) AS date,
        AVG(CASE sentiment
            WHEN 'positive' THEN 1.0
            WHEN 'negative' THEN -1.0
            ELSE 0.0
        END) AS sentiment_score,
        COUNT(*) AS mentions
    FROM {bm}
    WHERE brand = '{brand}'
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    df = query_df(sql)
    if not df.empty:
        df["rolling_sentiment"] = df["sentiment_score"].rolling(window, min_periods=1).mean()
    return df


def competitor_gap(target_brand: str) -> pd.DataFrame:
    """Which competitors appear in prompts where target brand is absent."""
    bm = _tbl("brand_mentions")
    sql = f"""
    WITH target_prompts AS (
        SELECT DISTINCT run_id, prompt_id, provider
        FROM {bm}
        WHERE brand = '{target_brand}'
    ),
    all_prompts AS (
        SELECT DISTINCT run_id, prompt_id, provider
        FROM {bm}
    ),
    missing_prompts AS (
        SELECT a.run_id, a.prompt_id, a.provider
        FROM all_prompts a
        LEFT JOIN target_prompts t
          ON a.run_id=t.run_id AND a.prompt_id=t.prompt_id AND a.provider=t.provider
        WHERE t.run_id IS NULL
    )
    SELECT
        bm.brand,
        COUNT(*) AS appears_when_target_absent,
        AVG(bm.position) AS avg_position
    FROM {bm} bm
    JOIN missing_prompts mp
      ON bm.run_id=mp.run_id AND bm.prompt_id=mp.prompt_id AND bm.provider=mp.provider
    WHERE bm.brand != '{target_brand}'
    GROUP BY bm.brand
    ORDER BY appears_when_target_absent DESC
    """
    return query_df(sql)


def citation_type_breakdown() -> pd.DataFrame:
    """Count citations by domain_type."""
    cs = _tbl("citation_sources")
    sql = f"""
    SELECT domain_type, COUNT(*) AS count, COUNT(DISTINCT domain) AS unique_domains
    FROM {cs}
    WHERE domain_type IS NOT NULL
    GROUP BY domain_type
    ORDER BY count DESC
    """
    return query_df(sql)
