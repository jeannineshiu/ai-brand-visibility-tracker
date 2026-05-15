"""Plotly Dash dashboard — interactive brand selector, 4 KPI cards, 6 charts."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.metrics.calculator import (
    visibility_summary_by_category, position_trend, competitor_gap,
    provider_breakdown, _tbl,
    brands_by_category, prompts_by_category, category_stats,
    llm_disagreement, citation_by_provider,
)
from src.storage.store import query_df

# ── constants ─────────────────────────────────────────────────────────────────
CAT_LABELS = {
    "project_management": "Project Management",
    "crm":                "CRM",
    "ai_writing":         "AI Writing",
    "developer_tools":    "Developer Tools",
}

# ── design tokens ──────────────────────────────────────────────────────────────
ACCENT  = "#F97316"   # orange — selected brand
MUTED   = "#CBD5E1"   # slate  — other brands
BG      = "#F8FAFC"

CARD_STYLE = {
    "background": "white",
    "border": "1px solid #E2E8F0",
    "borderRadius": "10px",
    "padding": "18px 24px",
    "textAlign": "center",
    "flex": "1",
    "minWidth": "120px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.07)",
}
GRID2 = {
    "display": "grid",
    "gridTemplateColumns": "1fr 1fr",
    "gap": "20px",
    "marginTop": "16px",
}
SECTION_LABEL = {
    "fontSize": "11px", "fontWeight": "600", "color": "#94A3B8",
    "letterSpacing": "0.08em", "textTransform": "uppercase",
    "margin": "28px 0 4px 2px",
}
PLOT_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font={"family": "Inter, -apple-system, sans-serif", "size": 12},
    margin=dict(l=120, r=60, t=50, b=40),
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, color: str = "#0F172A") -> "html.Div":
    from dash import html
    return html.Div([
        html.Div(value, style={"fontSize": "26px", "fontWeight": "700", "color": color}),
        html.Div(label, style={"fontSize": "11px", "color": "#94A3B8", "marginTop": "4px",
                                "textTransform": "uppercase", "letterSpacing": "0.06em"}),
    ], style=CARD_STYLE)


def _empty_fig(msg: str = "No data") -> go.Figure:
    return (
        go.Figure()
        .add_annotation(text=msg, showarrow=False, font={"size": 13, "color": "#94A3B8"})
        .update_layout(**PLOT_LAYOUT, height=280)
    )


def _base(fig: go.Figure, title: str, height: int = 320, **kw) -> go.Figure:
    layout = {**PLOT_LAYOUT, **kw}
    fig.update_layout(title=title, height=height, **layout)
    return fig


# ── chart builders ─────────────────────────────────────────────────────────────

def _visibility_fig(brand: str, df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No visibility data")
    colors = [ACCENT if b == brand else MUTED for b in df["brand"]]
    fig = go.Figure(go.Bar(
        y=df["brand"], x=df["visibility_pct"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["visibility_pct"]],
        textposition="outside",
    ))
    return _base(fig, "Visibility % — All Brands",
                 height=max(300, len(df) * 44),
                 xaxis_title="Visibility %", yaxis_title="",
                 margin=dict(l=120, r=70, t=50, b=40))


def _sentiment_fig(brand: str, df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No sentiment data")
    df = df.copy()
    df["neutral_rate"] = (100 - df["pos_rate"] - df["neg_rate"]).clip(lower=0)
    lw = [2 if b == brand else 0 for b in df["brand"]]
    lc = [ACCENT if b == brand else "rgba(0,0,0,0)" for b in df["brand"]]
    fig = go.Figure(data=[
        go.Bar(name="Positive", y=df["brand"], x=df["pos_rate"],
               orientation="h", marker_color="#4ADE80",
               marker_line=dict(width=lw, color=lc)),
        go.Bar(name="Neutral",  y=df["brand"], x=df["neutral_rate"],
               orientation="h", marker_color="#CBD5E1",
               marker_line=dict(width=lw, color=lc)),
        go.Bar(name="Negative", y=df["brand"], x=df["neg_rate"],
               orientation="h", marker_color="#F87171",
               marker_line=dict(width=lw, color=lc)),
    ])
    fig.update_layout(barmode="stack")
    return _base(fig, "Sentiment — All Brands",
                 height=max(300, len(df) * 44),
                 xaxis_title="% of mentions",
                 margin=dict(l=120, r=40, t=50, b=40))


def _position_trend_fig(brand: str) -> go.Figure:
    df = position_trend(brand)
    if df.empty:
        return _empty_fig(f"No position data for {brand}")
    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["avg_position"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(size=7, color=ACCENT),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
        name="Avg Position",
    ))
    return _base(fig, f"{brand} — Mention Position Over Time",
                 yaxis=dict(autorange="reversed", title="Position  (1 = first mentioned)"),
                 xaxis_title="Date",
                 margin=dict(l=80, r=40, t=50, b=40))


def _competitor_gap_fig(brand: str, category: str | None = None) -> go.Figure:
    df = competitor_gap(brand, category)
    if df.empty:
        return _empty_fig(f"No competitor gap data for {brand}")
    df = df.head(10)
    fig = px.bar(
        df, x="appears_when_target_absent", y="brand",
        orientation="h",
        color="appears_when_target_absent",
        color_continuous_scale=[[0, "#FDE68A"], [1, "#DC2626"]],
        text="appears_when_target_absent",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _base(fig,
                 f"Competitors when '{brand}' is NOT mentioned",
                 height=max(300, len(df) * 44),
                 xaxis_title="Appearances", yaxis_title="",
                 margin=dict(l=120, r=60, t=50, b=40))


def _provider_fig(brand: str) -> go.Figure:
    df = provider_breakdown(brand)
    if df.empty:
        return _empty_fig(f"No provider data for {brand}")
    pivot = df.pivot_table(
        index="provider", columns="sentiment", values="count", fill_value=0
    ).reset_index()
    colors = {"positive": "#4ADE80", "neutral": "#CBD5E1", "negative": "#F87171"}
    fig = go.Figure()
    for s in ["positive", "neutral", "negative"]:
        if s in pivot.columns:
            fig.add_trace(go.Bar(
                name=s.capitalize(),
                x=pivot["provider"],
                y=pivot[s],
                marker_color=colors[s],
                text=pivot[s],
                textposition="outside",
            ))
    fig.update_layout(barmode="group")
    return _base(fig, f"{brand} — Mentions by LLM Provider",
                 xaxis_title="Provider", yaxis_title="Mentions",
                 margin=dict(l=60, r=40, t=50, b=40))


def _opportunity_fig(brand: str) -> go.Figure:
    # Try LightGBM first, fall back to rule-based scorer
    try:
        from src.recommender.train_lgbm import load_model, load_features_df, predict_opportunities
        model, features, le = load_model()
        df = load_features_df()
        if model is not None and df is not None and not df.empty:
            result = predict_opportunities(model, features, le, brand, df)
            if not result.empty:
                colors = [
                    "#DC2626" if s >= 60 else "#F59E0B" if s >= 35 else "#4ADE80"
                    for s in result["opportunity_score"]
                ]
                fig = go.Figure(go.Bar(
                    y=result["domain_type"], x=result["opportunity_score"],
                    orientation="h", marker_color=colors,
                    text=[f"{s:.0f}" for s in result["opportunity_score"]],
                    textposition="outside",
                ))
                fig.update_layout(xaxis=dict(range=[0, 115]))
                return _base(fig,
                             f"{brand} — Opportunity Score  (LightGBM · R²=0.74)",
                             xaxis_title="Score  (0–100)",
                             margin=dict(l=130, r=60, t=50, b=40))
    except Exception:
        pass

    # Rule-based fallback
    try:
        from src.recommender.scorer import compute_opportunity_scores
        from src.recommender.feature_engineer import build_source_features, build_brand_features
        recs = compute_opportunity_scores(
            build_source_features(), build_brand_features(brand), brand, []
        )
        if recs:
            types  = [r.domain_type for r in recs]
            scores = [r.opportunity_score for r in recs]
            colors = ["#DC2626" if s >= 60 else "#F59E0B" if s >= 35 else "#4ADE80" for s in scores]
            fig = go.Figure(go.Bar(
                y=types, x=scores, orientation="h",
                marker_color=colors,
                text=[str(s) for s in scores],
                textposition="outside",
            ))
            fig.update_layout(xaxis=dict(range=[0, 115]))
            return _base(fig,
                         f"{brand} — Opportunity Score  (rule-based)",
                         xaxis_title="Score  (0–100)",
                         margin=dict(l=130, r=60, t=50, b=40))
    except Exception:
        pass

    return _empty_fig("Opportunity data unavailable")


def _llm_disagreement_fig(category: str) -> go.Figure:
    df = llm_disagreement(category)
    if df.empty:
        return _empty_fig("No data for this category")
    provider_colors = {"openai": "#3B82F6", "anthropic": "#F97316", "gemini": "#8B5CF6"}
    provider_labels = {
        "openai":    "OpenAI (GPT-4o-mini)",
        "anthropic": "Anthropic (Claude Haiku)",
        "gemini":    "Gemini (2.5 Flash)",
    }
    fig = go.Figure()
    for prov in ["openai", "anthropic", "gemini"]:
        if prov not in df.columns:
            continue
        fig.add_trace(go.Bar(
            name=provider_labels[prov],
            y=df["brand"],
            x=df[prov],
            orientation="h",
            marker_color=provider_colors[prov],
            text=[f"{v:.1f}%" if v > 0 else "" for v in df[prov]],
            textposition="outside",
        ))
    fig.update_layout(barmode="group")
    return _base(
        fig, "Brand Visibility % — Cross-LLM Comparison",
        height=max(340, len(df) * 70),
        xaxis_title="Visibility %", yaxis_title="",
        margin=dict(l=130, r=80, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def _citation_by_provider_fig(category: str | None = None) -> go.Figure:
    df = citation_by_provider(category)
    if df.empty:
        return _empty_fig("No citation data")
    df = df.copy()
    totals = df.groupby("provider")["count"].transform("sum")
    df["pct"] = (df["count"] / totals * 100).round(1)
    df["provider_label"] = df["provider"].replace(
        {"openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Gemini"}
    )
    domain_colors = {
        "review_site":   "#4ADE80", "tech_media":    "#3B82F6",
        "community":     "#F97316", "official_docs": "#8B5CF6",
        "developer":     "#EC4899", "social_media":  "#EAB308",
        "academic":      "#14B8A6", "news":          "#64748B",
        "brand":         "#CBD5E1", "other":         "#94A3B8",
    }
    df["label"] = df["pct"].apply(lambda x: f"{x:.0f}%" if x >= 5 else "")
    fig = px.bar(
        df.sort_values("domain_type"),
        x="provider_label", y="pct", color="domain_type",
        barmode="stack",
        color_discrete_map=domain_colors,
        category_orders={"provider_label": ["OpenAI", "Anthropic", "Gemini"]},
        labels={"pct": "% of Citations", "provider_label": "", "domain_type": "Source Type"},
        text="label",
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    return _base(
        fig, "Citation Source Mix by LLM  (%)",
        height=400,
        xaxis_title="", yaxis_title="% of Citations",
        margin=dict(l=60, r=40, t=50, b=150),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
                    title_text=""),
        yaxis=dict(range=[0, 105]),
    )


# ── sidebar helpers ────────────────────────────────────────────────────────────

def _sidebar_stats() -> dict:
    """Query DB for sidebar statistics, falling back to BATCH_PROMPTS for categories."""
    try:
        bm = _tbl("brand_mentions")
        lr = _tbl("llm_responses")
        pr = _tbl("prompts")
        mentions  = query_df(f"SELECT COUNT(*) AS n FROM {bm}")
        responses = query_df(f"SELECT COUNT(*) AS n FROM {lr}")
        brands_n  = query_df(f"SELECT COUNT(DISTINCT brand) AS n FROM {bm}")
        providers = query_df(f"SELECT COUNT(DISTINCT provider) AS n FROM {lr}")
        dates     = query_df(f"SELECT MIN(DATE(created_at)) AS d0, MAX(DATE(created_at)) AS d1 FROM {lr}")
        cats      = query_df(f"SELECT category, COUNT(*) AS n FROM {pr} GROUP BY category ORDER BY n DESC")

        # Fall back to BATCH_PROMPTS when prompts table is empty
        if cats.empty:
            from data.prompts_batch import BATCH_PROMPTS
            from collections import Counter
            counts = Counter(p.category for p in BATCH_PROMPTS)
            cats_list = [{"category": k, "n": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
        else:
            cats_list = cats.to_dict("records")

        return {
            "mentions":  int(mentions.iloc[0, 0])  if not mentions.empty  else 0,
            "responses": int(responses.iloc[0, 0]) if not responses.empty else 0,
            "brands":    int(brands_n.iloc[0, 0])  if not brands_n.empty  else 0,
            "providers": int(providers.iloc[0, 0]) if not providers.empty else 0,
            "d0": str(dates["d0"].iloc[0]) if not dates.empty else "—",
            "d1": str(dates["d1"].iloc[0]) if not dates.empty else "—",
            "cats": cats_list,
        }
    except Exception:
        return {"mentions": 0, "responses": 0, "brands": 0, "providers": 0,
                "d0": "—", "d1": "—", "cats": []}


def _build_sidebar(stats: dict) -> "html.Div":
    from dash import html

    side_label = {**SECTION_LABEL, "margin": "20px 0 6px 0"}

    # Static row (plain text value)
    def static_row(label, value):
        return html.Div([
            html.Span(label, style={"fontSize": "12px", "color": "#64748B"}),
            html.Span(value, style={"fontSize": "13px", "fontWeight": "700",
                                    "color": "#0F172A"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "6px 0", "borderBottom": "1px solid #F1F5F9"})

    # Reactive row — value Span has an ID, updated by callback
    def reactive_row(label, comp_id):
        return html.Div([
            html.Span(label, style={"fontSize": "12px", "color": "#64748B"}),
            html.Span("—", id=comp_id,
                      style={"fontSize": "13px", "fontWeight": "700",
                             "color": "#0F172A"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "6px 0", "borderBottom": "1px solid #F1F5F9"})

    total_prompts = sum(c["n"] for c in stats["cats"])
    cat_rows = [
        html.Div([
            html.Span(
                CAT_LABELS.get(c["category"],
                               c["category"].replace("_", " ").title()),
                style={"fontSize": "12px", "color": "#475569"},
            ),
            html.Span(str(c["n"]),
                      style={"fontSize": "12px", "fontWeight": "600",
                             "color": "#0F172A"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "3px 0"})
        for c in stats["cats"]
    ]

    return html.Div([

        html.Div("About", style=side_label),
        html.P(
            "This tool tracks how often your brand appears in AI-generated "
            "answers across OpenAI, Anthropic, and Gemini — the new discovery "
            "surface for B2B software buyers.",
            style={"fontSize": "12px", "color": "#475569",
                   "lineHeight": "1.7", "margin": "0"},
        ),
        html.P(
            "This discipline is called Generative Engine Optimization (GEO). "
            "Traditional SEO tracks Google rankings. This tracks AI mindshare.",
            style={"fontSize": "12px", "color": "#64748B",
                   "lineHeight": "1.7", "marginTop": "8px"},
        ),

        # ── Data Volume (reactive — updates when category changes) ────
        html.Div("Data Volume", style=side_label),
        reactive_row("Brand Mentions", "sb-mentions"),
        reactive_row("LLM Responses",  "sb-responses"),
        reactive_row("Brands Tracked", "sb-brands"),
        static_row("LLM Providers",    str(stats["providers"])),
        reactive_row("Date Range",     "sb-dates"),

        # ── Prompt Coverage (static — same for all categories) ────────
        html.Div("Prompt Coverage", style=side_label),
        html.P(
            f"{total_prompts} prompts across {len(stats['cats'])} categories",
            style={"fontSize": "12px", "color": "#64748B", "margin": "0 0 6px"},
        ),
        *cat_rows,

        html.Div("How to Read", style=side_label),
        html.P(
            "Visibility % = brand mentions ÷ total LLM queries × 100. "
            "Position 1 = brand named first in the response.",
            style={"fontSize": "11px", "color": "#94A3B8",
                   "lineHeight": "1.6", "margin": "0"},
        ),

    ], style={
        "width": "230px",
        "flexShrink": "0",
        "background": "white",
        "border": "1px solid #E2E8F0",
        "borderRadius": "12px",
        "padding": "0 18px 20px",
        "position": "sticky",
        "top": "24px",
        "alignSelf": "flex-start",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.07)",
        "maxHeight": "calc(100vh - 120px)",
        "overflowY": "auto",
    })


# ── app builder ────────────────────────────────────────────────────────────────

def build_dash_app(target_brand=None):
    """
    Build interactive Dash app.
    Brand list is loaded dynamically from the database, grouped by category.
    target_brand: if provided, pre-selects this brand on load.
    """
    from dash import Dash, dcc, html, Input, Output

    # ── load category → brand mapping ─────────────────────────────────
    all_brands_by_cat = brands_by_category()
    if not all_brands_by_cat:
        # fallback: compute from BATCH_PROMPTS if DB prompts table is empty
        from collections import defaultdict
        from data.prompts_batch import BATCH_PROMPTS
        mapping: dict = defaultdict(set)
        for p in BATCH_PROMPTS:
            mapping[p.category].update(p.target_brands)
        all_brands_by_cat = {k: sorted(v) for k, v in sorted(mapping.items())}

    all_cats = sorted(all_brands_by_cat.keys())

    # pick default category — prefer the one containing target_brand
    default_cat = all_cats[0] if all_cats else None
    if target_brand:
        for cat, blist in all_brands_by_cat.items():
            if target_brand in blist:
                default_cat = cat
                break

    initial_brands = all_brands_by_cat.get(default_cat, [])
    default_brand = (
        target_brand if target_brand in initial_brands
        else (initial_brands[0] if initial_brands else None)
    )

    cat_options = [
        {"label": CAT_LABELS.get(c, c.replace("_", " ").title()), "value": c}
        for c in all_cats
    ]
    brand_options = [{"label": b, "value": b} for b in initial_brands]

    app = Dash(__name__)
    sidebar = _build_sidebar(_sidebar_stats())

    _tab_style = {
        "padding": "12px 24px", "fontSize": "13px", "color": "#64748B",
        "fontWeight": "500", "background": "white", "border": "none",
        "borderBottom": "3px solid transparent",
    }
    _tab_selected = {
        **_tab_style, "color": "#0F172A", "fontWeight": "700",
        "borderBottom": f"3px solid {ACCENT}",
    }

    llm_tab = html.Div([
        html.P(
            "Why do ChatGPT, Claude, and Gemini recommend different brands for the same query? "
            "Explore cross-LLM visibility disagreement and citation source preferences.",
            style={"fontSize": "13px", "color": "#64748B", "margin": "0 0 20px",
                   "lineHeight": "1.7"},
        ),
        html.Div([
            html.Label("Category",
                       style={"fontSize": "12px", "fontWeight": "600", "color": "#475569"}),
            dcc.Dropdown(
                id="llm-cat",
                options=cat_options,
                value=default_cat,
                clearable=False,
                style={"marginTop": "6px", "width": "220px"},
            ),
        ], style={"marginBottom": "28px"}),

        html.P("Cross-LLM Visibility Disagreement", style=SECTION_LABEL),
        html.P(
            "Same prompts — how differently do the three LLMs rank brands in this category?",
            style={"fontSize": "12px", "color": "#64748B", "margin": "0 0 12px"},
        ),
        dcc.Graph(id="llm-disagreement-chart"),

        html.P("Citation Source Preferences by LLM",
               style={**SECTION_LABEL, "marginTop": "36px"}),
        html.P(
            "Which types of sources does each LLM tend to cite for queries in this category?",
            style={"fontSize": "12px", "color": "#64748B", "margin": "0 0 12px"},
        ),
        dcc.Graph(id="llm-citation-chart"),

    ], style={"padding": "24px 32px", "maxWidth": "1100px"})

    app.layout = html.Div([

        # ── header ────────────────────────────────────────────────────
        html.Div([
            html.H1("AI Brand Visibility Tracker",
                    style={"margin": "0", "fontSize": "22px", "fontWeight": "700",
                           "color": "#0F172A"}),
            html.P(
                "Track your brand in LLM-generated answers  ·  OpenAI  ·  Anthropic  ·  Gemini",
                style={"margin": "4px 0 0", "color": "#64748B", "fontSize": "13px"},
            ),
        ], style={"padding": "24px 32px 16px", "borderBottom": "1px solid #E2E8F0",
                  "background": "white"}),

        # ── tabs ──────────────────────────────────────────────────────
        dcc.Tabs([

            dcc.Tab(
                label="Brand Dashboard",
                style=_tab_style, selected_style=_tab_selected,
                children=[html.Div([

                    sidebar,

                    html.Div([

                        # ── category + brand selectors (side by side) ─────────
                        html.Div([
                            html.Div([
                                html.Label("Category",
                                           style={"fontSize": "12px", "fontWeight": "600",
                                                  "color": "#475569"}),
                                dcc.Dropdown(
                                    id="category-selector",
                                    options=cat_options,
                                    value=default_cat,
                                    clearable=False,
                                    style={"marginTop": "6px", "width": "220px"},
                                ),
                            ]),
                            html.Div([
                                html.Label("Brand",
                                           style={"fontSize": "12px", "fontWeight": "600",
                                                  "color": "#475569"}),
                                dcc.Dropdown(
                                    id="brand-selector",
                                    options=brand_options,
                                    value=default_brand,
                                    clearable=False,
                                    style={"marginTop": "6px", "width": "260px"},
                                ),
                            ]),
                        ], style={"display": "flex", "gap": "20px",
                                  "alignItems": "flex-end", "marginTop": "4px"}),

                        # ── KPI cards ─────────────────────────────────────────
                        html.Div(id="kpi-cards",
                                 style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                                        "marginTop": "20px"}),

                        # ── sample prompts for selected category ──────────────
                        html.Div(id="prompt-examples"),

                        # ── section: competitive context ──────────────────────
                        html.P("Competitive Context — All Brands", style=SECTION_LABEL),
                        html.Div([
                            dcc.Graph(id="visibility-chart"),
                            dcc.Graph(id="sentiment-chart"),
                        ], style=GRID2),

                        # ── section: brand deep-dive ──────────────────────────
                        html.P(id="deep-dive-label", style=SECTION_LABEL),
                        html.Div([
                            dcc.Graph(id="position-trend-chart"),
                            dcc.Graph(id="competitor-gap-chart"),
                        ], style=GRID2),

                        html.Div([
                            dcc.Graph(id="provider-chart"),
                            dcc.Graph(id="opportunity-chart"),
                        ], style={**GRID2, "marginBottom": "48px"}),

                    ], style={"flex": "1", "minWidth": "0"}),

                ], style={"display": "flex", "gap": "24px",
                          "padding": "24px 32px", "alignItems": "flex-start"})],
            ),

            dcc.Tab(
                label="LLM Behavior Analysis",
                style=_tab_style, selected_style=_tab_selected,
                children=[llm_tab],
            ),

        ], style={"background": "white", "borderBottom": "1px solid #E2E8F0",
                  "padding": "0 8px"}),

    ], style={"maxWidth": "1500px", "margin": "0 auto",
              "fontFamily": "Inter, -apple-system, sans-serif",
              "background": BG, "minHeight": "100vh"})

    # ── callback 1: category → brand options ──────────────────────────
    @app.callback(
        [Output("brand-selector", "options"),
         Output("brand-selector", "value")],
        Input("category-selector", "value"),
    )
    def update_brand_options(category):
        blist = all_brands_by_cat.get(category, [])
        opts = [{"label": b, "value": b} for b in blist]
        return opts, (blist[0] if blist else None)

    # ── callback 2: category → sidebar Data Volume stats ─────────────
    @app.callback(
        [Output("sb-mentions",  "children"),
         Output("sb-responses", "children"),
         Output("sb-brands",    "children"),
         Output("sb-dates",     "children")],
        Input("category-selector", "value"),
    )
    def update_sidebar_stats(category):
        if not category:
            return "—", "—", "—", "—"
        s = category_stats(category)
        date_str = (
            f"{s['d0'][:7]} → {s['d1'][:7]}" if s["d0"] != "—" else "—"
        )
        return f"{s['mentions']:,}", f"{s['responses']:,}", str(s["brands"]), date_str

    # ── callback 3: category → sample prompts ─────────────────────────
    @app.callback(
        Output("prompt-examples", "children"),
        Input("category-selector", "value"),
    )
    def update_prompt_examples(category):
        if not category:
            return []
        texts = prompts_by_category(category, limit=4)
        if not texts:
            return []
        label = CAT_LABELS.get(category, category.replace("_", " ").title())
        items = [
            html.Li(t, style={"fontSize": "13px", "color": "#475569",
                               "marginBottom": "8px", "lineHeight": "1.6"})
            for t in texts
        ]
        return html.Div([
            html.P(f"Sample Prompts — {label}", style=SECTION_LABEL),
            html.Div(
                html.Ul(items, style={"margin": "0", "paddingLeft": "20px"}),
                style={
                    "background": "white",
                    "border": "1px solid #E2E8F0",
                    "borderLeft": f"3px solid {ACCENT}",
                    "borderRadius": "8px",
                    "padding": "14px 18px",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                },
            ),
        ])

    # ── callback 4: brand + category → KPIs + 6 charts ───────────────
    @app.callback(
        [Output("kpi-cards",            "children"),
         Output("visibility-chart",     "figure"),
         Output("sentiment-chart",      "figure"),
         Output("position-trend-chart", "figure"),
         Output("competitor-gap-chart", "figure"),
         Output("provider-chart",       "figure"),
         Output("opportunity-chart",    "figure"),
         Output("deep-dive-label",      "children")],
        [Input("brand-selector",    "value"),
         Input("category-selector", "value")],
    )
    def update_all(brand, category):
        if not brand or not category:
            empty = _empty_fig()
            return [], empty, empty, empty, empty, empty, empty, ""

        cat_brands = all_brands_by_cat.get(category, [])

        # All metrics scoped to this category's brands only
        df_all = visibility_summary_by_category(category, cat_brands)
        row_data = df_all[df_all["brand"] == brand]

        if not row_data.empty:
            r = row_data.iloc[0]
            kpis = [
                _kpi_card("Visibility %",
                          f"{r['visibility_pct']:.1f}%",          "#3B82F6"),
                _kpi_card("Avg Position",
                          f"{r['avg_position']:.1f}",             "#8B5CF6"),
                _kpi_card("Positive Rate",
                          f"{r['pos_rate']:.0f}%",                "#22C55E"),
                _kpi_card("Negative Rate",
                          f"{r['neg_rate']:.0f}%",                "#EF4444"),
            ]
        else:
            kpis = [_kpi_card("No data", "—")]

        return (
            kpis,
            _visibility_fig(brand, df_all),   # df_all = category brands only
            _sentiment_fig(brand, df_all),
            _position_trend_fig(brand),
            _competitor_gap_fig(brand, category),
            _provider_fig(brand),
            _opportunity_fig(brand),
            f"{brand} — Deep-Dive",
        )

    # ── callback 5: llm-cat → cross-LLM disagreement chart ───────────
    @app.callback(
        Output("llm-disagreement-chart", "figure"),
        Input("llm-cat", "value"),
    )
    def update_llm_disagreement(category):
        if not category:
            return _empty_fig()
        return _llm_disagreement_fig(category)

    # ── callback 6: llm-cat → citation by provider chart ─────────────
    @app.callback(
        Output("llm-citation-chart", "figure"),
        Input("llm-cat", "value"),
    )
    def update_llm_citation(category):
        if not category:
            return _empty_fig()
        return _citation_by_provider_fig(category)

    return app
