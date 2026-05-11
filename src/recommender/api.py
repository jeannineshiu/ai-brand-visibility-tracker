"""FastAPI service exposing recommendation and metrics endpoints."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional
import logging

from src.utils.config import RETRAIN_API_KEY
from src.recommender.feature_engineer import build_source_features, build_brand_features
from src.recommender.scorer import compute_opportunity_scores, _generate_action
from src.recommender.train_lgbm import (
    build_training_data, train, save_model, load_model, load_features_df, predict_opportunities
)
from src.metrics.calculator import visibility_summary, competitor_gap, citation_type_breakdown
from src.storage import store

logger = logging.getLogger(__name__)

_lgbm_model = None
_lgbm_features = None
_lgbm_encoder = None
_lgbm_training_df = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lgbm_model, _lgbm_features, _lgbm_encoder, _lgbm_training_df
    store.init()
    _lgbm_model, _lgbm_features, _lgbm_encoder = load_model()
    if _lgbm_model is not None:
        _lgbm_training_df = load_features_df()
        logger.info("LightGBM model loaded from disk (features_df=%s).", _lgbm_training_df is not None)
    else:
        logger.info("No saved model found — attempting auto-train from available data...")
        try:
            df = build_training_data()
            model, le, features = train(df)
            save_model(model, le, features, df)
            _lgbm_model, _lgbm_features, _lgbm_encoder = model, features, le
            _lgbm_training_df = df
            logger.info("Auto-train complete: %d rows, %d brands.", len(df), df["brand"].nunique())
        except Exception as e:
            logger.warning("Auto-train skipped (no data yet): %s", e)
    yield


app = FastAPI(
    title="AI Brand Visibility Tracker API",
    version="1.0.0",
    description="Track brand visibility in LLM responses and get source recommendations.",
    lifespan=lifespan,
)

def _require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if not RETRAIN_API_KEY:
        raise HTTPException(status_code=503, detail="RETRAIN_API_KEY not configured on server")
    if x_api_key != RETRAIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


_lgbm_training_df = None


def _get_lgbm_df():
    """Return feature DataFrame: prefers in-memory cache, then DB query."""
    global _lgbm_training_df
    if _lgbm_training_df is not None:
        return _lgbm_training_df
    try:
        _lgbm_training_df = build_training_data()
    except Exception:
        _lgbm_training_df = None
    return _lgbm_training_df


# ── Request / Response models ──────────────────────────────────────────────────

class PromptIn(BaseModel):
    prompt_id: Optional[str] = None
    prompt_text: str
    category: str
    target_brands: list[str]


class PromptOut(BaseModel):
    prompt_id: str
    prompt_text: str
    category: str
    target_brands: list[str]


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

@app.get("/prompts", response_model=list[PromptOut])
def list_prompts():
    """List all prompts stored in the database."""
    try:
        return [p.model_dump() for p in store.list_prompts()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts", response_model=PromptOut, status_code=201)
def create_prompt(body: PromptIn):
    """
    Add a new prompt to the database.
    If prompt_id is omitted, one is auto-generated.
    """
    import uuid
    from src.utils.models import PromptConfig
    prompt_id = body.prompt_id or f"custom_{uuid.uuid4().hex[:8]}"
    if store.get_prompt(prompt_id) is not None:
        raise HTTPException(status_code=409, detail=f"prompt_id '{prompt_id}' already exists")
    p = PromptConfig(
        prompt_id=prompt_id,
        prompt_text=body.prompt_text,
        category=body.category,
        target_brands=body.target_brands,
    )
    store.save_prompt(p)
    return p.model_dump()


@app.delete("/prompts/{prompt_id}", status_code=204)
def delete_prompt(prompt_id: str):
    """Delete a prompt by its ID."""
    if not store.delete_prompt(prompt_id):
        raise HTTPException(status_code=404, detail=f"prompt_id '{prompt_id}' not found")


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
def retrain_model(_: None = Depends(_require_api_key)):
    """Retrain LightGBM on latest BigQuery data and reload into memory."""
    global _lgbm_model, _lgbm_features, _lgbm_encoder, _lgbm_training_df
    try:
        df = build_training_data()
        model, le, features = train(df)
        save_model(model, le, features, df)
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
