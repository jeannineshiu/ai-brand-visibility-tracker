"""
LightGBM ranking model trainer.

Problem framing:
  One row = (brand, domain_type) pair.
  Target = competitor_avg_cooccurrence - brand_cooccurrence
           (positive = competitors outrank this brand here → high opportunity)
  Prediction = opportunity score for investing in this source type.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

from src.storage.store import query_df, _USE_BIGQUERY
from src.utils.config import GCP_PROJECT_ID, BIGQUERY_DATASET

MODEL_PATH = Path("data/lgbm_opportunity_model.pkl")
ENCODER_PATH = Path("data/lgbm_label_encoder.pkl")

DOMAIN_TYPE_WEIGHTS = {
    "review_site": 1.0, "tech_media": 0.9, "business_media": 0.8,
    "community": 0.7, "developer": 0.7, "reference": 0.6,
    "blog_platform": 0.5, "video": 0.5, "professional_social": 0.4,
    "brand_owned": 0.3, "other": 0.2,
}


def _tbl(name: str) -> str:
    if _USE_BIGQUERY:
        return f"`{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{name}`"
    return name


def build_training_data() -> pd.DataFrame:
    """
    Build feature matrix from brand_mentions + citation_sources.
    One row per (brand, domain_type).
    """
    bm = _tbl("brand_mentions")
    cs = _tbl("citation_sources")

    # Brand-level features: overall visibility stats
    brand_sql = f"""
    SELECT
        brand,
        COUNT(*)         AS total_mentions,
        AVG(position)    AS brand_avg_position,
        COUNT(DISTINCT prompt_id) AS brand_prompt_count
    FROM {bm}
    GROUP BY brand
    """

    # Citation × brand co-occurrence features
    if _USE_BIGQUERY:
        cooc_sql = f"""
        SELECT
            c.domain_type,
            b.brand,
            COUNT(DISTINCT CONCAT(b.run_id, b.prompt_id, b.provider)) AS cooccurrence,
            AVG(b.position)    AS avg_position_with_source,
            COUNT(DISTINCT c.domain) AS unique_domains
        FROM {cs} c
        JOIN {bm} b
          ON c.prompt_id = b.prompt_id AND c.provider = b.provider
        WHERE c.domain_type IS NOT NULL
        GROUP BY c.domain_type, b.brand
        """
        # Competitor average per domain_type
        comp_sql = f"""
        SELECT
            domain_type,
            AVG(cooccurrence) AS competitor_avg_cooccurrence
        FROM (
            SELECT
                c.domain_type,
                b.brand,
                COUNT(DISTINCT CONCAT(b.run_id, b.prompt_id, b.provider)) AS cooccurrence
            FROM {cs} c
            JOIN {bm} b ON c.prompt_id = b.prompt_id AND c.provider = b.provider
            WHERE c.domain_type IS NOT NULL
            GROUP BY c.domain_type, b.brand
        )
        GROUP BY domain_type
        """
    else:
        cooc_sql = f"""
        SELECT
            c.domain_type,
            b.brand,
            COUNT(DISTINCT b.run_id || b.prompt_id || b.provider) AS cooccurrence,
            AVG(b.position)    AS avg_position_with_source,
            COUNT(DISTINCT c.domain) AS unique_domains
        FROM {cs} c
        JOIN {bm} b
          ON c.prompt_id = b.prompt_id AND c.provider = b.provider
        WHERE c.domain_type IS NOT NULL
        GROUP BY c.domain_type, b.brand
        """
        comp_sql = f"""
        SELECT
            domain_type,
            AVG(cooccurrence) AS competitor_avg_cooccurrence
        FROM (
            SELECT
                c.domain_type, b.brand,
                COUNT(DISTINCT b.run_id || b.prompt_id || b.provider) AS cooccurrence
            FROM {cs} c
            JOIN {bm} b ON c.prompt_id = b.prompt_id AND c.provider = b.provider
            WHERE c.domain_type IS NOT NULL
            GROUP BY c.domain_type, b.brand
        )
        GROUP BY domain_type
        """

    brand_df = query_df(brand_sql)
    cooc_df = query_df(cooc_sql)
    comp_df = query_df(comp_sql)

    if cooc_df.empty:
        raise ValueError("No co-occurrence data — run the pipeline first.")

    # Merge all features
    df = cooc_df.merge(comp_df, on="domain_type", how="left")
    df = df.merge(brand_df, on="brand", how="left")

    # Target: gap between competitor average and this brand's cooccurrence
    # Positive = competitors appear more here → high opportunity
    df["opportunity_gap"] = df["competitor_avg_cooccurrence"] - df["cooccurrence"]

    # Additional features
    df["domain_authority_weight"] = df["domain_type"].map(DOMAIN_TYPE_WEIGHTS).fillna(0.2)
    df["position_score"] = 1.0 / df["avg_position_with_source"].clip(lower=0.1)

    return df


def train(df: pd.DataFrame) -> tuple[lgb.LGBMRegressor, LabelEncoder, list[str]]:
    """Train LightGBM on the feature matrix."""
    le = LabelEncoder()
    df = df.copy()
    df["domain_type_enc"] = le.fit_transform(df["domain_type"])

    features = [
        "cooccurrence",
        "avg_position_with_source",
        "unique_domains",
        "competitor_avg_cooccurrence",
        "total_mentions",
        "brand_avg_position",
        "brand_prompt_count",
        "domain_authority_weight",
        "position_score",
        "domain_type_enc",
    ]
    target = "opportunity_gap"

    X = df[features].fillna(0)
    y = df[target].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE: {mae:.4f}   R²: {r2:.4f}   (test size: {len(y_test)})")

    return model, le, features


def save_model(model, le, features):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"  Model saved → {MODEL_PATH}")


def load_model():
    """Load saved model. Returns (model, features, label_encoder) or None."""
    if not MODEL_PATH.exists():
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return obj["model"], obj["features"], le


def predict_opportunities(
    model, features: list[str], le: LabelEncoder,
    target_brand: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Score all domain_types for a target brand using trained model."""
    brand_df = df[df["brand"] == target_brand].copy()
    if brand_df.empty:
        return pd.DataFrame()

    brand_df["domain_type_enc"] = le.transform(
        brand_df["domain_type"].where(
            brand_df["domain_type"].isin(le.classes_), other=le.classes_[0]
        )
    )
    X = brand_df[features].fillna(0)
    brand_df["predicted_opportunity"] = model.predict(X)

    # Normalize to 0–100 scale
    min_val = brand_df["predicted_opportunity"].min()
    max_val = brand_df["predicted_opportunity"].max()
    if max_val > min_val:
        brand_df["opportunity_score"] = (
            (brand_df["predicted_opportunity"] - min_val) / (max_val - min_val) * 100
        ).round(1)
    else:
        brand_df["opportunity_score"] = 50.0

    return brand_df[["domain_type", "opportunity_score", "cooccurrence",
                      "competitor_avg_cooccurrence"]].sort_values(
        "opportunity_score", ascending=False
    )
