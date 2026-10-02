"""Shared chart styling and small Plotly builders, so pages stay about content."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE, RED, GREY, GREEN, AMBER = "#1F4E79", "#B23A48", "#9AA5B1", "#2E7D5B", "#C98A1B"
LAYOUT = dict(template="simple_white", margin=dict(l=10, r=10, t=40, b=10), height=340)


def bar(df: pd.DataFrame, x: str, y: str, title: str, x_title: str = "", y_title: str = "",
        color: str = BLUE, text_fmt: str = "%{text:.1f}%", hover: list[str] | None = None) -> go.Figure:
    fig = px.bar(df, x=x, y=y, text=y, hover_data=hover, color_discrete_sequence=[color])
    fig.update_traces(texttemplate=text_fmt, textposition="outside", cliponaxis=False)
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, **LAYOUT)
    return fig


def hbar(values: pd.Series, title: str, x_title: str, color: str = BLUE) -> go.Figure:
    fig = px.bar(x=values.values, y=values.index, orientation="h", color_discrete_sequence=[color])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="", **LAYOUT)
    return fig


def predicted_vs_actual(bands: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=bands.risk_band, y=bands.actual_default_rate * 100, name="Actual", marker_color=BLUE)
    fig.add_scatter(x=bands.risk_band, y=bands.avg_pd * 100, name="Predicted (avg PD)",
                    mode="lines+markers", line=dict(color=RED))
    fig.update_layout(title="Predicted vs actual default rate by risk band (1 = safest)",
                      xaxis_title="Risk band", yaxis_title="Default rate (%)",
                      xaxis=dict(dtick=1), **LAYOUT)
    return fig


def roc(roc_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (name, g), col in zip(roc_df.groupby("model", sort=False), [BLUE, RED]):
        fig.add_scatter(x=g.fpr, y=g.tpr, name=name, line=dict(color=col))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="ROC curve (holdout)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", **LAYOUT)
    return fig


def profit_curve(sweep: pd.DataFrame, chosen: float) -> go.Figure:
    # beyond the cut-off where eligibility stops growing the curve is flat; trim it
    grows = sweep.eligible.diff().fillna(1) > 0
    last = sweep.loc[grows, "pd_cutoff"].max() if grows.any() else sweep.pd_cutoff.max()
    sweep = sweep[sweep.pd_cutoff <= max(last + 0.05, chosen + 0.05)]
    fig = go.Figure()
    fig.add_scatter(x=sweep.pd_cutoff, y=sweep.expected_profit, mode="lines+markers",
                    name="Expected profit", line=dict(color=BLUE))
    fig.add_vline(x=chosen, line=dict(color=GREEN, dash="dash"),
                  annotation_text=f"chosen {chosen:.2f}", annotation_position="top")
    fig.update_layout(title="Expected profit by PD cut-off (validation)", xaxis_title="PD cut-off",
                      yaxis_title="Expected 12-mo profit (NT$)", **LAYOUT)
    return fig


def tradeoff(sweep: pd.DataFrame, champ: dict, chall: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=sweep.eligible, y=sweep.default_rate_pct, mode="lines+markers",
                    name="Challenger at each cut-off", line=dict(color=BLUE),
                    customdata=sweep.pd_cutoff,
                    hovertemplate="cut-off %{customdata}<br>%{x:,} accounts<br>%{y:.2f}%")
    fig.add_scatter(x=[champ["Eligible accounts"]], y=[champ["Default rate %"]], mode="markers",
                    name="Champion", marker=dict(color=RED, size=14, symbol="diamond"))
    fig.add_scatter(x=[chall["Eligible accounts"]], y=[chall["Default rate %"]], mode="markers",
                    name="Selected challenger", marker=dict(color=GREEN, size=12))
    fig.update_layout(title="Volume vs risk (holdout)", xaxis_title="Accounts eligible",
                      yaxis_title="Actual default rate (%)", **LAYOUT)
    return fig


def overlaid_hist(a: pd.Series, b: pd.Series, a_name: str, b_name: str, title: str) -> go.Figure:
    fig = go.Figure()
    for s, name, col in [(a, a_name, GREY), (b, b_name, BLUE)]:
        fig.add_histogram(x=s, name=name, histnorm="percent", nbinsx=40, marker_color=col, opacity=0.6)
    fig.update_layout(barmode="overlay", title=title, xaxis_title="Predicted default probability",
                      yaxis_title="% of accounts", **LAYOUT)
    return fig
