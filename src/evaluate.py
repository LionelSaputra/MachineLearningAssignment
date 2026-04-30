"""
evaluate.py
-----------
Helper functions for generating evaluation visualizations used in Streamlit.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def radar_chart_input(symptom_values: dict):
    """
    Create a Plotly radar chart of the user's input symptom intensities.
    """
    labels = list(symptom_values.keys())
    values = list(symptom_values.values())
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(108,99,255,0.2)',
        line=dict(color='#6C63FF', width=2),
        name='Intensitas Gejala',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 3]),
            bgcolor="#1E1E2E",
        ),
        paper_bgcolor="#1E1E2E",
        font=dict(color="#E0E0FF", size=11),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def confidence_bar_chart(class_labels: list, probabilities: list):
    """
    Create a horizontal bar chart of prediction probabilities.
    """
    colors = [
        "#6C63FF", "#FF6584", "#43D9AD", "#FFB347", "#A0C4FF", "#B5EAD7"
    ][:len(class_labels)]

    fig = go.Figure(go.Bar(
        x=probabilities,
        y=class_labels,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#2D2D44"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#E0E0FF", size=11),
        margin=dict(l=10, r=40, t=30, b=30),
        height=280,
    )
    return fig


def metrics_comparison_chart(results: dict):
    """
    Grouped bar chart comparing all model metrics.
    """
    model_names = list(results.keys())
    metrics     = ["accuracy", "precision", "recall", "f1"]
    colors      = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347"]

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        vals = [results[n][metric] for n in model_names]
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=model_names,
            y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        barmode='group',
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#E0E0FF", size=11),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(range=[0, 1.2], gridcolor="#2D2D44"),
        xaxis=dict(gridcolor="#2D2D44"),
        margin=dict(l=20, r=20, t=20, b=60),
        height=380,
    )
    return fig


def confusion_matrix_plotly(conf_matrix: np.ndarray, class_labels: list, model_name: str):
    """
    Interactive Plotly confusion matrix heatmap.
    """
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Prediksi", y="Aktual", color="Count"),
        x=class_labels,
        y=class_labels,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#E0E0FF", size=10),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def feature_importance_plotly(feature_names: list, importances: list, top_n: int = 15):
    """
    Horizontal bar chart for Random Forest feature importances.
    """
    paired   = sorted(zip(importances, feature_names), reverse=True)[:top_n]
    imp_vals = [p[0] for p in reversed(paired)]
    feat_lbs = [p[1].replace("_", " ").title() for p in reversed(paired)]

    fig = go.Figure(go.Bar(
        x=imp_vals,
        y=feat_lbs,
        orientation="h",
        marker=dict(
            color=imp_vals,
            colorscale=[[0, "#3D3D6B"], [1, "#6C63FF"]],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(color="#E0E0FF", size=10),
        xaxis=dict(title="Importance Score", gridcolor="#2D2D44"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=20, t=20, b=20),
        height=420,
    )
    return fig
