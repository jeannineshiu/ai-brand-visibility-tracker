"""Plotly Dash dashboard for brand visibility metrics."""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from src.metrics.calculator import visibility_summary, position_trend, sentiment_trend, competitor_gap


def visibility_bar_chart(brands: list[str] | None = None) -> go.Figure:
    """Horizontal bar chart: brand visibility % with sentiment breakdown."""
    df = visibility_summary(brands)
    if df.empty:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["brand"], x=df["visibility_pct"],
        orientation="h", name="Visibility %",
        marker_color=px.colors.qualitative.Set2[:len(df)],
        text=df["visibility_pct"].astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        title="Brand Visibility (% of prompts mentioned)",
        xaxis_title="Visibility %", yaxis_title="",
        height=max(300, len(df) * 50),
        margin=dict(l=120, r=60, t=60, b=40),
    )
    return fig


def sentiment_breakdown_chart(brands: list[str] | None = None) -> go.Figure:
    """Stacked bar: positive / neutral / negative rates per brand."""
    df = visibility_summary(brands)
    if df.empty:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    df["neutral_rate"] = (100 - df["pos_rate"] - df["neg_rate"]).clip(lower=0)

    fig = go.Figure(data=[
        go.Bar(name="Positive", y=df["brand"], x=df["pos_rate"],
               orientation="h", marker_color="#2ECC71"),
        go.Bar(name="Neutral",  y=df["brand"], x=df["neutral_rate"],
               orientation="h", marker_color="#95A5A6"),
        go.Bar(name="Negative", y=df["brand"], x=df["neg_rate"],
               orientation="h", marker_color="#E74C3C"),
    ])
    fig.update_layout(
        barmode="stack",
        title="Sentiment Breakdown by Brand",
        xaxis_title="% of mentions",
        height=max(300, len(df) * 50),
        margin=dict(l=120, r=60, t=60, b=40),
    )
    return fig


def position_trend_chart(brand: str) -> go.Figure:
    """Line chart: average mention position over time for a brand."""
    df = position_trend(brand)
    if df.empty:
        return go.Figure().add_annotation(text=f"No data for {brand}", showarrow=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_position"],
        mode="lines+markers", name="Avg Position",
        line=dict(color="#3498DB", width=2),
    ))
    fig.update_layout(
        title=f"{brand} — Mention Position Trend (lower = mentioned earlier)",
        yaxis=dict(autorange="reversed", title="Position"),
        xaxis_title="Date",
    )
    return fig


def competitor_gap_chart(target_brand: str) -> go.Figure:
    """Bar chart: which competitors appear when target brand is absent."""
    df = competitor_gap(target_brand)
    if df.empty:
        return go.Figure().add_annotation(text="No gap data", showarrow=False)

    fig = px.bar(
        df, x="brand", y="appears_when_target_absent",
        title=f"Competitors appearing when '{target_brand}' is NOT mentioned",
        labels={"appears_when_target_absent": "Count", "brand": "Competitor"},
        color="appears_when_target_absent",
        color_continuous_scale="Reds",
    )
    return fig


def build_dash_app(brands: list[str], target_brand: str):
    """Build and return a Dash app instance."""
    from dash import Dash, dcc, html

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("AI Brand Visibility Tracker", style={"fontFamily": "sans-serif", "padding": "20px"}),

        html.Div([
            dcc.Graph(figure=visibility_bar_chart(brands)),
            dcc.Graph(figure=sentiment_breakdown_chart(brands)),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),

        html.Div([
            dcc.Graph(figure=position_trend_chart(target_brand)),
            dcc.Graph(figure=competitor_gap_chart(target_brand)),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginTop": "20px"}),
    ], style={"maxWidth": "1400px", "margin": "0 auto"})

    return app
