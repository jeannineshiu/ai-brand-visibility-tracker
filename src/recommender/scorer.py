"""
Opportunity scorer for citation source types.

Rule-based scoring (interpretable, works with small data).
LightGBM upgrade path is wired in as a commented-out section.
"""
import pandas as pd
from dataclasses import dataclass

# Domain type priority weights (editorial knowledge)
DOMAIN_TYPE_WEIGHTS = {
    "review_site":       1.0,   # g2, capterra — highest purchase-intent signal
    "tech_media":        0.9,   # techcrunch, zdnet — high authority
    "business_media":    0.8,   # forbes, inc — broad reach
    "community":         0.7,   # reddit — trust signal from users
    "developer":         0.7,   # stackoverflow, dev.to
    "blog_platform":     0.5,   # medium, substack
    "brand_owned":       0.3,   # own site — low independent credibility
    "professional_social": 0.4, # linkedin
    "video":             0.5,   # youtube
    "reference":         0.6,   # wikipedia
    "other":             0.2,
}


@dataclass
class Recommendation:
    domain_type: str
    opportunity_score: float   # 0–100
    reason: str
    action: str
    priority: str              # HIGH / MEDIUM / LOW


def compute_opportunity_scores(
    source_features: pd.DataFrame,
    brand_features: pd.DataFrame,
    target_brand: str,
    competitors: list[str],
) -> list[Recommendation]:
    """
    Compute opportunity score per domain_type using rule-based scoring.

    Score = base_weight × 40
          + competitor_gap_bonus (up to 30)
          + citation_frequency_bonus (up to 20)
          + position_bonus (up to 10)

    Max = 100.
    """
    recommendations: list[Recommendation] = []

    # If no citation data yet, fall back to pure domain-weight recommendations
    if source_features.empty:
        return _fallback_recommendations(brand_features, target_brand, competitors)

    target_df = source_features[source_features["brand"] == target_brand]
    competitor_df = source_features[source_features["brand"].isin(competitors)]

    for domain_type, weight in DOMAIN_TYPE_WEIGHTS.items():
        target_row = target_df[target_df["domain_type"] == domain_type]
        comp_rows = competitor_df[competitor_df["domain_type"] == domain_type]

        target_cooc = target_row["cooccurrence_count"].sum() if not target_row.empty else 0
        comp_avg_cooc = comp_rows["cooccurrence_count"].mean() if not comp_rows.empty else 0

        # Component 1: base authority weight (0–40)
        base = weight * 40

        # Component 2: competitor gap bonus (0–30)
        # High score when competitors appear here but target doesn't
        gap = max(0, comp_avg_cooc - target_cooc)
        comp_bonus = min(30, gap * 10)

        # Component 3: citation frequency bonus (0–20)
        if not target_row.empty and "citation_count" in target_row.columns:
            freq = target_row["citation_count"].sum()
            freq_bonus = min(20, freq * 2)
        else:
            freq_bonus = 0

        # Component 4: position bonus (0–10)
        # Better position = lower number = more bonus
        if not target_row.empty and "avg_position_with_source" in target_row.columns:
            avg_pos = target_row["avg_position_with_source"].mean()
            pos_bonus = max(0, 10 - avg_pos * 2)
        else:
            pos_bonus = 5  # neutral

        score = round(base + comp_bonus + freq_bonus + pos_bonus, 1)
        priority = "HIGH" if score >= 60 else "MEDIUM" if score >= 35 else "LOW"

        action = _generate_action(domain_type, target_cooc, comp_avg_cooc)
        reason = _generate_reason(domain_type, target_cooc, comp_avg_cooc, score)

        recommendations.append(Recommendation(
            domain_type=domain_type,
            opportunity_score=score,
            reason=reason,
            action=action,
            priority=priority,
        ))

    return sorted(recommendations, key=lambda r: r.opportunity_score, reverse=True)


def _generate_action(domain_type: str, target_cooc: float, comp_cooc: float) -> str:
    actions = {
        "review_site":     "Submit/update your G2 and Capterra profiles; ask customers for reviews",
        "tech_media":      "Pitch product stories to TechCrunch, ZDNet, TechRadar",
        "business_media":  "Issue press releases via PRNewswire; pitch Forbes/Inc contributors",
        "community":       "Engage authentically in relevant subreddits; don't spam",
        "developer":       "Publish technical content on dev.to; answer Stack Overflow questions",
        "blog_platform":   "Publish thought leadership posts on Medium/Substack",
        "brand_owned":     "Improve your own site's SEO and domain authority",
        "professional_social": "Increase LinkedIn content output and employee advocacy",
        "video":           "Create YouTube tutorials and comparison videos",
        "reference":       "Ensure your Wikipedia page is accurate and well-cited",
        "other":           "Audit and diversify your external link profile",
    }
    return actions.get(domain_type, "Increase presence in this source type")


def _generate_reason(domain_type: str, target_cooc: float, comp_cooc: float, score: float) -> str:
    if comp_cooc > target_cooc:
        return (f"Competitors appear {comp_cooc:.1f}× in {domain_type} sources "
                f"vs your {target_cooc:.1f}× — high gap opportunity")
    if score >= 60:
        return f"{domain_type} is high-authority and you have room to grow presence"
    return f"{domain_type} sources contribute to AI citation signals in this category"


def _fallback_recommendations(
    brand_features: pd.DataFrame,
    target_brand: str,
    competitors: list[str],
) -> list[Recommendation]:
    """Rank by domain weight alone when no citation data exists."""
    recs = []
    for domain_type, weight in sorted(DOMAIN_TYPE_WEIGHTS.items(), key=lambda x: -x[1]):
        score = round(weight * 100, 1)
        priority = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
        recs.append(Recommendation(
            domain_type=domain_type,
            opportunity_score=score,
            reason=f"High-authority source type for AI citation signals (no citation data yet)",
            action=_generate_action(domain_type, 0, 0),
            priority=priority,
        ))
    return recs

# ── LightGBM upgrade path (uncomment when you have 500+ rows of citation data) ──
# import lightgbm as lgb
# def train_lgbm(features_df: pd.DataFrame, label_col: str = "cooccurrence_count"):
#     X = features_df[["citation_count", "unique_domains", "avg_position_with_source"]]
#     y = features_df[label_col]
#     model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
#     model.fit(X, y)
#     return model
