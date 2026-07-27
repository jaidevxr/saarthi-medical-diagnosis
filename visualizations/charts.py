"""
Reusable Plotly chart generators styled to the healthcare dashboard theme.
Every function returns a plotly.graph_objects.Figure.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from utils.theme import PALETTE, get_plotly_template

_TPL = None


def _template():
    global _TPL
    if _TPL is None:
        _TPL = get_plotly_template()
    return _TPL


def _base_layout(**overrides):
    """Common layout params for all charts."""
    layout = dict(
        template=_template(),
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor=PALETTE["bg_card"],
        plot_bgcolor=PALETTE["bg_card"],
        margin=dict(l=40, r=20, t=50, b=40),
    )
    layout.update(overrides)
    return layout


# ─────────────────────────────────────────────────
# Distribution Charts
# ─────────────────────────────────────────────────

def plot_disease_distribution(df, target_col="prognosis"):
    """Horizontal bar chart of disease class counts."""
    counts = df[target_col].value_counts().sort_values()
    fig = go.Figure(go.Bar(
        y=counts.index,
        x=counts.values,
        orientation="h",
        marker_color=PALETTE["primary"],
        marker_line=dict(width=0),
        hovertemplate="%{y}: %{x} samples<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title="Disease Distribution",
            xaxis_title="Number of Samples",
            yaxis_title="",
            height=max(450, len(counts) * 22),
        )
    )
    return fig


def plot_symptom_frequency(df, top_n=25):
    """Bar chart of the most frequently present symptoms."""
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    freq = df[symptom_cols].sum().sort_values(ascending=False).head(top_n)
    names = [s.replace("_", " ").title() for s in freq.index]

    fig = go.Figure(go.Bar(
        x=names,
        y=freq.values,
        marker_color=PALETTE["secondary"],
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=f"Top {top_n} Most Frequent Symptoms",
            xaxis_title="",
            yaxis_title="Frequency",
            xaxis_tickangle=-45,
            height=480,
        )
    )
    return fig


def plot_class_pie(df, target_col="prognosis"):
    """Pie chart of disease class proportions."""
    counts = df[target_col].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,
        marker=dict(colors=PALETTE["chart_sequence"] * 5),
        textinfo="label+percent",
        textposition="outside",
        textfont_size=10,
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title="Disease Class Proportions",
            height=600,
            showlegend=False,
        )
    )
    return fig


# ─────────────────────────────────────────────────
# Correlation & Heatmaps
# ─────────────────────────────────────────────────

def plot_correlation_heatmap(df, top_n=30):
    """Heatmap of pairwise correlations for the top-N most variable symptoms."""
    symptom_cols = [c for c in df.columns if c != "prognosis"]
    variances = df[symptom_cols].var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    corr = df[top_cols].corr()

    display_names = [c.replace("_", " ").title()[:18] for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=display_names,
        y=display_names,
        colorscale=[
            [0, PALETTE["primary_dark"]],
            [0.5, "#FFFFFF"],
            [1, PALETTE["danger"]],
        ],
        zmin=-1, zmax=1,
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=f"Symptom Correlation Matrix (Top {top_n})",
            height=650,
            xaxis_tickangle=-45,
        )
    )
    return fig


# ─────────────────────────────────────────────────
# Model Evaluation Charts
# ─────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels):
    """Annotated heatmap confusion matrix."""
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0, "#FFFFFF"],
            [1, PALETTE["primary"]],
        ],
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=True,
    ))
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i][j]
            color = "#FFFFFF" if val > cm.max() * 0.6 else PALETTE["text_primary"]
            fig.add_annotation(
                x=labels[j], y=labels[i],
                text=str(val),
                showarrow=False,
                font=dict(size=8, color=color),
            )
    fig.update_layout(
        **_base_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=max(500, len(labels) * 18),
            yaxis_autorange="reversed",
        )
    )
    return fig


def plot_roc_curves(roc_data):
    """
    Overlaid ROC curves for multiple models.

    Parameters
    ----------
    roc_data : dict
        {model_name: {"fpr": array, "tpr": array, "auc": float}}
    """
    fig = go.Figure()
    colors = PALETTE["chart_sequence"]

    for i, (name, data) in enumerate(roc_data.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=data["fpr"],
            y=data["tpr"],
            mode="lines",
            name=f'{name} (AUC={data["auc"]:.3f})',
            line=dict(color=color, width=2),
        ))

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=PALETTE["text_light"], dash="dash", width=1),
        showlegend=False,
    ))

    fig.update_layout(
        **_base_layout(
            title="ROC Curves — Model Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
            legend=dict(x=0.55, y=0.05),
        )
    )
    return fig


def plot_model_comparison(results_df, metric_cols=None):
    """Grouped bar chart comparing models across metrics."""
    if metric_cols is None:
        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = PALETTE["chart_sequence"]

    fig = go.Figure()
    for i, metric in enumerate(metric_cols):
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df["Algorithm"],
                y=results_df[metric],
                marker_color=colors[i % len(colors)],
                hovertemplate=f"{metric}: " + "%{y:.3f}<extra></extra>",
            ))

    fig.update_layout(
        **_base_layout(
            title="Model Performance Comparison",
            barmode="group",
            xaxis_title="",
            yaxis_title="Score",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    )
    return fig


def plot_feature_importance(importances, names, top_n=20):
    """Horizontal bar chart of feature importances."""
    # Sort and take top N
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        y=[names[i].replace("_", " ").title() for i in idx],
        x=[importances[i] for i in idx],
        orientation="h",
        marker_color=PALETTE["secondary"],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            height=max(400, top_n * 24),
        )
    )
    return fig


# ─────────────────────────────────────────────────
# General Purpose
# ─────────────────────────────────────────────────

def plot_histogram(series, title="", x_label="", bins=30):
    """Simple histogram for a numeric series."""
    fig = go.Figure(go.Histogram(
        x=series,
        nbinsx=bins,
        marker_color=PALETTE["primary"],
        marker_line=dict(color=PALETTE["primary_dark"], width=0.5),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title=x_label, yaxis_title="Count", height=400))
    return fig


def plot_box_plot(df, columns, title=""):
    """Box plot for one or more numeric columns."""
    fig = go.Figure()
    for i, col in enumerate(columns[:10]):
        fig.add_trace(go.Box(
            y=df[col],
            name=col.replace("_", " ").title()[:20],
            marker_color=PALETTE["chart_sequence"][i % len(PALETTE["chart_sequence"])],
        ))
    fig.update_layout(**_base_layout(title=title, height=420))
    return fig


def plot_scatter(df, x_col, y_col, color_col=None, title=""):
    """Scatter plot of two columns, optionally coloured by a third."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        color_discrete_sequence=PALETTE["chart_sequence"],
        template=_template(),
    )
    fig.update_layout(**_base_layout(title=title, height=450))
    return fig


def plot_bar(names, values, title="", x_label="", y_label="", color=None):
    """Simple vertical bar chart."""
    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=color or PALETTE["primary"],
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=420))
    return fig
