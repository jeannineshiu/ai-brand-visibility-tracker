"""FastAPI service exposing recommendation and metrics endpoints."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.recommender.feature_engineer import build_source_features, build_brand_features
from src.recommender.scorer import compute_opportunity_scores, _generate_action
from src.recommender.train_lgbm import (
    build_training_data, train, save_model, load_model, predict_opportunities
)
from src.metrics.calculator import visibility_summary, competitor_gap, citation_type_breakdown

app = FastAPI(
    title="AI Brand Visibility Tracker API",
    version="1.0.0",
    description="Track brand visibility in LLM responses and get source recommendations.",
)

# Load LightGBM model at startup (if available)
_lgbm_model, _lgbm_features, _lgbm_encoder = load_model()
_lgbm_training_df = None  # lazy-loaded


def _get_lgbm_df():
    """Lazy-load training dataframe (BigQuery query, cache in memory)."""
    global _lgbm_training_df
    if _lgbm_training_df is None:
        try:
            _lgbm_training_df = build_training_data()
        except Exception:
            _lgbm_training_df = None
    return _lgbm_training_df


# ── Request / Response models ──────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    target_brand: str
    competitors: list[str]
    top_n: Optional[int] = 5


class RecommendationOut(BaseModel):
    domain_type: str
    opportunity_score: float
    priority: str
    reason: str
    action: str
    model_used: str


class VisibilityOut(BaseModel):
    brand: str
    mentions: int
    avg_position: float
    pos_rate: float
    neg_rate: float
    visibility_pct: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "lgbm_model_loaded": _lgbm_model is not None,
    }


@app.post("/recommendations", response_model=list[RecommendationOut])
def get_recommendations(req: RecommendRequest):
    """
    Return ranked source-type recommendations for a target brand.
    Uses LightGBM model if available, falls back to rule-based scorer.
    Response includes 'model_used' field indicating which engine was used.
    """
    try:
        # ── LightGBM path ──────────────────────────────────────────────────────
        if _lgbm_model is not None:
            df = _get_lgbm_df()
            if df is not None and not df.empty:
                result = predict_opportunities(
                    _lgbm_model, _lgbm_features, _lgbm_encoder,
                    req.target_brand, df,
                )
                if not result.empty:
                    out = []
                    for _, row in result.head(req.top_n).iterrows():
                        score = float(row["opportunity_score"])
                        priority = "HIGH" if score >= 60 else "MEDIUM" if score >= 35 else "LOW"
                        gap = float(row["competitor_avg_cooccurrence"]) - float(row["cooccurrence"])
                        reason = (
                            f"Competitors appear {row['competitor_avg_cooccurrence']:.1f}× "
                            f"vs your {row['cooccurrence']:.0f}× — LightGBM gap signal"
                            if gap > 0 else
                            f"You lead here ({row['cooccurrence']:.0f} vs avg {row['competitor_avg_cooccurrence']:.1f}) — maintain presence"
                        )
                        out.append(RecommendationOut(
                            domain_type=row["domain_type"],
                            opportunity_score=score,
                            priority=priority,
                            reason=reason,
                            action=_generate_action(row["domain_type"], 0, 0),
                            model_used="lightgbm",
                        ))
                    return out

        # ── Rule-based fallback ────────────────────────────────────────────────
        source_features = build_source_features()
        brand_features = build_brand_features(req.target_brand)
        recs = compute_opportunity_scores(
            source_features, brand_features, req.target_brand, req.competitors
        )
        return [
            RecommendationOut(
                domain_type=r.domain_type,
                opportunity_score=r.opportunity_score,
                priority=r.priority,
                reason=r.reason,
                action=r.action,
                model_used="rule_based",
            )
            for r in recs[: req.top_n]
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain_model():
    """Retrain LightGBM on latest BigQuery data and reload into memory."""
    global _lgbm_model, _lgbm_features, _lgbm_encoder, _lgbm_training_df
    try:
        df = build_training_data()
        model, le, features = train(df)
        save_model(model, le, features)
        _lgbm_model, _lgbm_features, _lgbm_encoder = model, features, le
        _lgbm_training_df = df
        return {
            "status": "retrained",
            "rows": len(df),
            "brands": int(df["brand"].nunique()),
            "domain_types": int(df["domain_type"].nunique()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visibility", response_model=list[VisibilityOut])
def get_visibility(brands: Optional[str] = None):
    """
    Return visibility summary. Example: /visibility?brands=Asana,Jira,Linear
    """
    brand_list = [b.strip() for b in brands.split(",")] if brands else None
    try:
        df = visibility_summary(brand_list)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/competitor-gap/{target_brand}")
def get_competitor_gap(target_brand: str):
    """Return competitors that appear when target brand is absent."""
    try:
        df = competitor_gap(target_brand)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/citations/breakdown")
def get_citation_breakdown():
    """Return citation count grouped by domain type."""
    try:
        df = citation_type_breakdown()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
