"""Computes visibility metrics from stored brand mention data."""
import json
import pandas as pd
from src.storage.store import query_df, _USE_BIGQUERY
from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET


def _tbl(name: str) -> str:
    """Return fully-qualified table name for the active backend."""
    if _USE_BIGQUERY:
        return f"`{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{name}`"
    return name


def _safe(value: str) -> str:
    """Escape single quotes to prevent SQL injection in brand/label strings."""
    return value.replace("'", "''")


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
    if df.empty:
        df["visibility_pct"] = pd.Series(dtype=float)
        return df

    total = int(total_df.iloc[0, 0]) if not total_df.empty else 1
    total = max(total, 1)
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
    WHERE brand = '{_safe(brand)}'
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
    WHERE brand = '{_safe(brand)}'
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    df = query_df(sql)
    if not df.empty:
        df["rolling_sentiment"] = df["sentiment_score"].rolling(window, min_periods=1).mean()
    return df


def competitor_gap(target_brand: str, category: str | None = None) -> pd.DataFrame:
    """Which competitors appear in prompts where target brand is absent.
    When category is provided, only considers prompts from that category so
    cross-category brands (e.g. CRM tools appearing in PM gap) are excluded.
    """
    bm = _tbl("brand_mentions")
    pr = _tbl("prompts")

    if category:
        cat_filter = f"JOIN {pr} p ON bm.prompt_id = p.prompt_id WHERE p.category = '{_safe(category)}'"
        cat_and    = f"AND p.category = '{_safe(category)}'"
        target_cte = f"""
        SELECT DISTINCT bm.run_id, bm.prompt_id, bm.provider
        FROM {bm} bm
        JOIN {pr} p ON bm.prompt_id = p.prompt_id
        WHERE bm.brand = '{_safe(target_brand)}' AND p.category = '{_safe(category)}'"""
        all_cte = f"""
        SELECT DISTINCT bm.run_id, bm.prompt_id, bm.provider
        FROM {bm} bm
        JOIN {pr} p ON bm.prompt_id = p.prompt_id
        WHERE p.category = '{_safe(category)}'"""
        final_join = f"""
        JOIN {pr} p ON bm.prompt_id = p.prompt_id
        WHERE bm.brand != '{_safe(target_brand)}' AND p.category = '{_safe(category)}'"""
    else:
        target_cte = f"""
        SELECT DISTINCT run_id, prompt_id, provider
        FROM {bm} WHERE brand = '{_safe(target_brand)}'"""
        all_cte = f"""
        SELECT DISTINCT run_id, prompt_id, provider FROM {bm}"""
        final_join = f"WHERE bm.brand != '{_safe(target_brand)}'"

    sql = f"""
    WITH target_prompts AS ({target_cte}),
    all_prompts AS ({all_cte}),
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
    {final_join}
    GROUP BY bm.brand
    ORDER BY appears_when_target_absent DESC
    """
    return query_df(sql)


def visibility_summary_by_category(category: str, brands: list[str]) -> pd.DataFrame:
    """
    Per-brand visibility scoped to a single category.
    Denominator = unique (run_id, prompt_id, provider) combos whose prompt
    belongs to that category (not all-time total).
    Columns: brand, mentions, avg_position, pos_rate, neg_rate, visibility_pct
    """
    bm = _tbl("brand_mentions")
    pr = _tbl("prompts")
    lr = _tbl("llm_responses")

    if not brands:
        return pd.DataFrame(columns=["brand", "mentions", "avg_position",
                                     "pos_rate", "neg_rate", "visibility_pct"])

    quoted = ", ".join(f"'{_safe(b)}'" for b in brands)
    sql = f"""
    SELECT
        bm.brand,
        COUNT(*)                                              AS mentions,
        AVG(bm.position)                                     AS avg_position,
        ROUND(100.0 * {_sentiment_count('positive')} / COUNT(*), 1) AS pos_rate,
        ROUND(100.0 * {_sentiment_count('negative')} / COUNT(*), 1) AS neg_rate
    FROM {bm} bm
    JOIN {pr} p ON bm.prompt_id = p.prompt_id
    WHERE p.category = '{_safe(category)}'
      AND bm.brand IN ({quoted})
    GROUP BY bm.brand
    ORDER BY mentions DESC
    """
    df = query_df(sql)

    if _USE_BIGQUERY:
        total_sql = f"""
        SELECT COUNT(DISTINCT CONCAT(lr.run_id, lr.prompt_id, lr.provider))
        FROM {lr} lr
        JOIN {pr} p ON lr.prompt_id = p.prompt_id
        WHERE p.category = '{_safe(category)}'
        """
    else:
        total_sql = f"""
        SELECT COUNT(DISTINCT lr.run_id || lr.prompt_id || lr.provider)
        FROM {lr} lr
        JOIN {pr} p ON lr.prompt_id = p.prompt_id
        WHERE p.category = '{_safe(category)}'
        """

    if df.empty:
        df["visibility_pct"] = pd.Series(dtype=float)
        return df

    total_df = query_df(total_sql)
    total = max(int(total_df.iloc[0, 0]) if not total_df.empty else 1, 1)
    df["visibility_pct"] = (df["mentions"] / total * 100).round(1)
    return df


def category_stats(category: str) -> dict:
    """Brand Mentions / LLM Responses / Brands Tracked / Date Range for one category."""
    bm = _tbl("brand_mentions")
    pr = _tbl("prompts")
    lr = _tbl("llm_responses")
    cat = f"p.category = '{_safe(category)}'"

    mentions_df  = query_df(f"SELECT COUNT(*) AS n FROM {bm} bm JOIN {pr} p ON bm.prompt_id=p.prompt_id WHERE {cat}")
    responses_df = query_df(f"SELECT COUNT(*) AS n FROM {lr} lr JOIN {pr} p ON lr.prompt_id=p.prompt_id WHERE {cat}")
    brands_df    = query_df(f"SELECT COUNT(DISTINCT bm.brand) AS n FROM {bm} bm JOIN {pr} p ON bm.prompt_id=p.prompt_id WHERE {cat}")
    dates_df     = query_df(f"SELECT MIN(DATE(lr.created_at)) AS d0, MAX(DATE(lr.created_at)) AS d1 FROM {lr} lr JOIN {pr} p ON lr.prompt_id=p.prompt_id WHERE {cat}")

    d0 = str(dates_df["d0"].iloc[0]) if not dates_df.empty and pd.notna(dates_df["d0"].iloc[0]) else "—"
    d1 = str(dates_df["d1"].iloc[0]) if not dates_df.empty and pd.notna(dates_df["d1"].iloc[0]) else "—"
    return {
        "mentions":  int(mentions_df.iloc[0, 0])  if not mentions_df.empty  else 0,
        "responses": int(responses_df.iloc[0, 0]) if not responses_df.empty else 0,
        "brands":    int(brands_df.iloc[0, 0])    if not brands_df.empty    else 0,
        "d0": d0, "d1": d1,
    }


def list_tracked_brands() -> list[str]:
    """Return all distinct brands in the database, sorted alphabetically."""
    bm = _tbl("brand_mentions")
    df = query_df(f"SELECT DISTINCT brand FROM {bm} ORDER BY brand")
    return df["brand"].tolist() if not df.empty else []


def provider_breakdown(brand: str) -> pd.DataFrame:
    """Mentions by provider × sentiment for a brand."""
    bm = _tbl("brand_mentions")
    sql = f"""
    SELECT provider, sentiment, COUNT(*) AS count
    FROM {bm}
    WHERE brand = '{_safe(brand)}'
    GROUP BY provider, sentiment
    ORDER BY provider, sentiment
    """
    return query_df(sql)


def brands_by_category() -> dict[str, list[str]]:
    """Return {category: sorted brand list} from the prompts table target_brands JSON."""
    pr = _tbl("prompts")
    df = query_df(f"SELECT category, target_brands FROM {pr}")
    mapping: dict[str, set] = {}
    for _, row in df.iterrows():
        try:
            brands = json.loads(row["target_brands"])
        except Exception:
            continue
        mapping.setdefault(row["category"], set()).update(brands)
    return {k: sorted(v) for k, v in sorted(mapping.items())}


def prompts_by_category(category: str, limit: int = 5) -> list[str]:
    """Return sample prompt texts for a given category."""
    pr = _tbl("prompts")
    sql = (
        f"SELECT prompt_text FROM {pr} "
        f"WHERE category = '{_safe(category)}' "
        f"LIMIT {limit}"
    )
    df = query_df(sql)
    return df["prompt_text"].tolist() if not df.empty else []


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


def llm_disagreement(category: str) -> pd.DataFrame:
    """
    Per-brand visibility % broken down by LLM provider for a category.
    Returns wide DataFrame: brand | openai | anthropic | gemini (visibility %)
    Sorted by average visibility descending.
    """
    bm = _tbl("brand_mentions")
    pr = _tbl("prompts")
    lr = _tbl("llm_responses")

    sql = f"""
    SELECT bm.brand, bm.provider, COUNT(*) AS mentions
    FROM {bm} bm
    JOIN {pr} p ON bm.prompt_id = p.prompt_id
    WHERE p.category = '{_safe(category)}'
    GROUP BY bm.brand, bm.provider
    """
    df = query_df(sql)
    if df.empty:
        return pd.DataFrame(columns=["brand", "openai", "anthropic", "gemini"])

    if _USE_BIGQUERY:
        denom_sql = f"""
        SELECT lr.provider, COUNT(DISTINCT CONCAT(lr.run_id, lr.prompt_id)) AS total
        FROM {lr} lr
        JOIN {pr} p ON lr.prompt_id = p.prompt_id
        WHERE p.category = '{_safe(category)}'
        GROUP BY lr.provider
        """
    else:
        denom_sql = f"""
        SELECT lr.provider, COUNT(DISTINCT lr.run_id || lr.prompt_id) AS total
        FROM {lr} lr
        JOIN {pr} p ON lr.prompt_id = p.prompt_id
        WHERE p.category = '{_safe(category)}'
        GROUP BY lr.provider
        """
    denom_df = query_df(denom_sql)
    denom = dict(zip(denom_df["provider"], denom_df["total"])) if not denom_df.empty else {}

    df["total"] = df["provider"].map(denom).fillna(1)
    df["visibility_pct"] = (df["mentions"] / df["total"] * 100).round(1)

    pivot = df.pivot_table(
        index="brand", columns="provider", values="visibility_pct", fill_value=0
    ).reset_index()
    pivot.columns.name = None
    for prov in ["openai", "anthropic", "gemini"]:
        if prov not in pivot.columns:
            pivot[prov] = 0.0

    pivot["_avg"] = pivot[["openai", "anthropic", "gemini"]].mean(axis=1)
    return pivot.sort_values("_avg", ascending=False).drop(columns="_avg").reset_index(drop=True)


def citation_by_provider(category: str | None = None) -> pd.DataFrame:
    """
    Citation counts grouped by (provider, domain_type).
    Optionally filtered to a single category via JOIN with prompts table.
    Returns: provider, domain_type, count
    """
    cs = _tbl("citation_sources")
    pr = _tbl("prompts")

    if category:
        sql = f"""
        SELECT cs.provider, cs.domain_type, COUNT(*) AS count
        FROM {cs} cs
        JOIN {pr} p ON cs.prompt_id = p.prompt_id
        WHERE cs.domain_type IS NOT NULL AND cs.provider IS NOT NULL
          AND p.category = '{_safe(category)}'
        GROUP BY cs.provider, cs.domain_type
        ORDER BY cs.provider, count DESC
        """
    else:
        sql = f"""
        SELECT provider, domain_type, COUNT(*) AS count
        FROM {cs}
        WHERE domain_type IS NOT NULL AND provider IS NOT NULL
        GROUP BY provider, domain_type
        ORDER BY provider, count DESC
        """
    return query_df(sql)
