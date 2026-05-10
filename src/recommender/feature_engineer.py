"""Builds feature vectors from brand_mentions + citation_sources for the ranking model."""
import pandas as pd
from src.storage.store import query_df, _USE_BIGQUERY
from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET


def _tbl(name: str) -> str:
    if _USE_BIGQUERY:
        return f"`{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{name}`"
    return name


def build_source_features() -> pd.DataFrame:
    """
    One row per (domain_type, brand) combination.
    Features used by the ranking model:
      - citation_count:        how often this domain type is cited
      - unique_domains:        breadth of sources in this type
      - brand_mention_rate:    % of prompts where brand is mentioned alongside this source type
      - avg_position:          avg brand position when this source type is present
      - competitor_presence:   how many competing brands appear in same prompts
      - opportunity_gap:       difference between competitor avg citation and brand citation
    """
    cs = _tbl("citation_sources")
    bm = _tbl("brand_mentions")

    # Citation frequency per domain_type
    citation_sql = f"""
    SELECT
        domain_type,
        COUNT(*)                  AS citation_count,
        COUNT(DISTINCT domain)    AS unique_domains,
        COUNT(DISTINCT prompt_id) AS prompts_with_citations
    FROM {cs}
    WHERE domain_type IS NOT NULL
    GROUP BY domain_type
    """

    # Brand mention rate per domain_type (join on prompt_id + provider)
    cooccurrence_sql = f"""
    SELECT
        c.domain_type,
        b.brand,
        COUNT(DISTINCT b.prompt_id || b.provider) AS cooccurrence_count,
        AVG(b.position)                            AS avg_position_with_source
    FROM {cs} c
    JOIN {bm} b
      ON c.prompt_id = b.prompt_id AND c.provider = b.provider
    WHERE c.domain_type IS NOT NULL
    GROUP BY c.domain_type, b.brand
    """ if not _USE_BIGQUERY else f"""
    SELECT
        c.domain_type,
        b.brand,
        COUNT(DISTINCT CONCAT(b.prompt_id, b.provider)) AS cooccurrence_count,
        AVG(b.position)                                  AS avg_position_with_source
    FROM {cs} c
    JOIN {bm} b
      ON c.prompt_id = b.prompt_id AND c.provider = b.provider
    WHERE c.domain_type IS NOT NULL
    GROUP BY c.domain_type, b.brand
    """

    try:
        citation_df = query_df(citation_sql)
        cooc_df = query_df(cooccurrence_sql)
    except Exception:
        # No citation data yet — return empty frame with correct columns
        return pd.DataFrame(columns=[
            "domain_type", "brand", "citation_count", "unique_domains",
            "cooccurrence_count", "avg_position_with_source", "opportunity_score",
        ])

    if citation_df.empty or cooc_df.empty:
        return pd.DataFrame(columns=[
            "domain_type", "brand", "citation_count", "unique_domains",
            "cooccurrence_count", "avg_position_with_source", "opportunity_score",
        ])

    merged = cooc_df.merge(citation_df, on="domain_type", how="left")
    return merged


def build_brand_features(target_brand: str) -> pd.DataFrame:
    """
    Features for the target brand vs competitors, grouped by domain_type.
    Used to compute opportunity_score in the scorer.
    """
    bm = _tbl("brand_mentions")

    brand_sql = f"""
    SELECT
        brand,
        COUNT(*)         AS total_mentions,
        AVG(position)    AS avg_position,
        COUNT(DISTINCT prompt_id) AS prompts_mentioned
    FROM {bm}
    GROUP BY brand
    """
    try:
        df = query_df(brand_sql)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    target_row = df[df["brand"] == target_brand]
    if target_row.empty:
        return df

    target_mentions = target_row["total_mentions"].values[0]
    df["gap_vs_target"] = df["total_mentions"] - target_mentions
    return df
