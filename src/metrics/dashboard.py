"""Plotly Dash dashboard — interactive brand selector, 4 KPI cards, 6 charts."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.metrics.calculator import (
    visibility_summary, position_trend, competitor_gap,
    provider_breakdown, list_tracked_brands,
)

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
    fig.update_layout(title=title, height=height, **PLOT_LAYOUT, **kw)
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
    lc = [ACCENT if b == brand else "transparent" for b in df["brand"]]
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


def _competitor_gap_fig(brand: str) -> go.Figure:
    df = competitor_gap(brand)
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
                             f"{brand} — Opportunity Score  (LightGBM · R²=0.80)",
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


# ── app builder ────────────────────────────────────────────────────────────────

def build_dash_app(brands=None, target_brand=None):
    """
    Build interactive Dash app.
    brands / target_brand are kept for backward compatibility but are ignored;
    brand list is loaded dynamically from the database.
    """
    from dash import Dash, dcc, html, Input, Output

    all_brands = list_tracked_brands()
    if not all_brands:
        all_brands = brands or []
    default = target_brand if target_brand in all_brands else (all_brands[0] if all_brands else None)

    app = Dash(__name__)

    app.layout = html.Div([

        # ── header ────────────────────────────────────────────────────
        html.Div([
            html.H1("AI Brand Visibility Tracker",
                    style={"margin": "0", "fontSize": "22px", "fontWeight": "700",
                           "color": "#0F172A"}),
            html.P("Track your brand in LLM-generated answers  ·  OpenAI  ·  Anthropic  ·  Gemini",
                   style={"margin": "4px 0 0", "color": "#64748B", "fontSize": "13px"}),
        ], style={"padding": "24px 32px 16px", "borderBottom": "1px solid #E2E8F0",
                  "background": "white"}),

        html.Div([

            # ── brand selector ────────────────────────────────────────
            html.Div([
                html.Label("Select Brand",
                           style={"fontSize": "12px", "fontWeight": "600", "color": "#475569"}),
                dcc.Dropdown(
                    id="brand-selector",
                    options=[{"label": b, "value": b} for b in all_brands],
                    value=default,
                    clearable=False,
                    style={"marginTop": "6px", "width": "300px"},
                ),
            ], style={"marginTop": "24px"}),

            # ── KPI cards ─────────────────────────────────────────────
            html.Div(id="kpi-cards",
                     style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                            "marginTop": "20px"}),

            # ── section: competitive context ─────────────────────────
            html.P("Competitive Context — All Brands", style=SECTION_LABEL),
            html.Div([
                dcc.Graph(id="visibility-chart"),
                dcc.Graph(id="sentiment-chart"),
            ], style=GRID2),

            # ── section: brand deep-dive ──────────────────────────────
            html.P(id="deep-dive-label", style=SECTION_LABEL),
            html.Div([
                dcc.Graph(id="position-trend-chart"),
                dcc.Graph(id="competitor-gap-chart"),
            ], style=GRID2),

            html.Div([
                dcc.Graph(id="provider-chart"),
                dcc.Graph(id="opportunity-chart"),
            ], style={**GRID2, "marginBottom": "48px"}),

        ], style={"padding": "0 32px"}),

    ], style={"maxWidth": "1400px", "margin": "0 auto",
              "fontFamily": "Inter, -apple-system, sans-serif",
              "background": BG, "minHeight": "100vh"})

    # ── single callback updates everything ────────────────────────────
    @app.callback(
        [Output("kpi-cards",            "children"),
         Output("visibility-chart",     "figure"),
         Output("sentiment-chart",      "figure"),
         Output("position-trend-chart", "figure"),
         Output("competitor-gap-chart", "figure"),
         Output("provider-chart",       "figure"),
         Output("opportunity-chart",    "figure"),
         Output("deep-dive-label",      "children")],
        Input("brand-selector", "value"),
    )
    def update_all(brand):
        if not brand:
            empty = _empty_fig()
            return [], empty, empty, empty, empty, empty, empty, ""

        df_all = visibility_summary()
        row = df_all[df_all["brand"] == brand]

        if not row.empty:
            r = row.iloc[0]
            kpis = [
                _kpi_card("Mentions",      str(int(r["mentions"])),       ACCENT),
                _kpi_card("Visibility %",  f"{r['visibility_pct']:.1f}%", "#3B82F6"),
                _kpi_card("Avg Position",  f"{r['avg_position']:.1f}",    "#8B5CF6"),
                _kpi_card("Positive Rate", f"{r['pos_rate']:.0f}%",       "#22C55E"),
                _kpi_card("Negative Rate", f"{r['neg_rate']:.0f}%",       "#EF4444"),
            ]
        else:
            kpis = [_kpi_card("No data", "—")]

        return (
            kpis,
            _visibility_fig(brand, df_all),
            _sentiment_fig(brand, df_all),
            _position_trend_fig(brand),
            _competitor_gap_fig(brand),
            _provider_fig(brand),
            _opportunity_fig(brand),
            f"{brand} — Deep-Dive",
        )

    return app
