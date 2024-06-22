import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import src.globals as g
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Collapse,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    SelectDataset,
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def calibration_curve():
    # Calibration curve (only positive predictions)
    true_probs, pred_probs = g.m_full.calibration_metrics.calibration_curve()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_probs,
            y=true_probs,
            mode="lines+markers",
            name="Calibration plot (Model)",
            line=dict(color="blue"),
            marker=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line=dict(color="orange", dash="dash"),
        )
    )

    fig.update_layout(
        title="Calibration Curve (only positive predictions)",
        xaxis_title="Confidence Score",
        yaxis_title="Fraction of True Positives",
        legend=dict(x=0.6, y=0.1),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700,
        height=500,
    )

    # fig.show()
    return fig


def confidence_score():
    # score_profile = g.m_full.confidence_score_profile()
    # f1_optimal_conf, best_f1 = g.m_full.get_f1_optimal_conf()
    # global df_down

    # df = pd.DataFrame(score_profile)
    # df.columns = ["scores", "Precision", "Recall", "F1"]

    # # downsample
    # if len(df) > 5000:
    #     df_down = df.iloc[:: len(df) // 1000]
    # else:
    #     df_down = df

    color_map = {
        "Precision": "#1f77b4",
        "Recall": "orange",
    }
    fig = px.line(
        g.dfsp_down,
        x="scores",
        y=["Precision", "Recall", "F1"],
        title="Confidence Score Profile",
        labels={"value": "Value", "variable": "Metric", "scores": "Confidence Score"},
        width=None,
        height=500,
        color_discrete_map=color_map,
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

    # Add vertical line for the best threshold
    fig.add_shape(
        type="line",
        x0=g.f1_optimal_conf,
        x1=g.f1_optimal_conf,
        y0=0,
        y1=g.best_f1,
        line=dict(color="gray", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=g.f1_optimal_conf,
        y=g.best_f1 + 0.04,
        text=f"F1-optimal threshold: {g.f1_optimal_conf:.2f}",
        showarrow=False,
    )
    # fig.show()
    return fig


def f1score_at_different_iou():
    # score_profile = g.m_full.confidence_score_profile()
    f1s = g.m_full.score_profile_f1s

    # downsample
    f1s_down = f1s[:, :: f1s.shape[1] // 1000]
    iou_names = list(map(lambda x: str(round(x, 2)), g.m.iouThrs.tolist()))
    df = pd.DataFrame(
        np.concatenate([g.dfsp_down["scores"].values[:, None], f1s_down.T], 1),
        columns=["scores"] + iou_names,
    )

    fig = px.line(
        df,
        x="scores",
        y=iou_names,
        title="F1-Score at different IoU Thresholds",
        labels={"value": "Value", "variable": "IoU threshold", "scores": "Confidence Score"},
        color_discrete_sequence=px.colors.sequential.Viridis,
        width=None,
        height=500,
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

    # add annotations for maximum F1-Score for each IoU threshold
    for i, iou in enumerate(iou_names):
        argmax_f1 = f1s[i].argmax()
        max_f1 = f1s[i][argmax_f1]
        score = g.score_profile["scores"][argmax_f1]
        fig.add_annotation(
            x=score,
            y=max_f1,
            text=f"Best score: {score:.2f}",
            showarrow=True,
            arrowhead=1,
            arrowcolor="black",
            ax=0,
            ay=-30,
        )

    # fig.show()
    return fig


def confidence_histogram():
    f1_optimal_conf, best_f1 = g.m_full.get_f1_optimal_conf()

    # Histogram of confidence scores (TP vs FP)
    scores_tp, scores_fp = g.m_full.calibration_metrics.scores_tp_and_fp(iou_idx=0)

    tp_y, tp_x = np.histogram(scores_tp, bins=40, range=[0, 1])
    fp_y, fp_x = np.histogram(scores_fp, bins=40, range=[0, 1])
    dx = (tp_x[1] - tp_x[0]) / 2

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores_fp,
            name="FP",
            marker=dict(color="#dd3f3f"),
            opacity=0.5,
            xbins=dict(size=0.025, start=0.0, end=1.0),
        )
    )
    fig.add_trace(
        go.Histogram(
            x=scores_tp,
            name="TP",
            marker=dict(color="#1fb466"),
            opacity=0.5,
            xbins=dict(size=0.025, start=0.0, end=1.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tp_x + dx,
            y=tp_y,
            mode="lines+markers",
            name="TP",
            line=dict(color="#1fb466", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fp_x + dx,
            y=fp_y,
            mode="lines+markers",
            name="FP",
            line=dict(color="#dd3f3f", width=2),
        )
    )

    # Best threshold
    fig.add_shape(
        type="line",
        x0=f1_optimal_conf,
        x1=f1_optimal_conf,
        y0=0,
        y1=tp_y.max() * 1.3,
        line=dict(color="orange", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=f1_optimal_conf,
        y=tp_y.max() * 1.3,
        text=f"F1-optimal threshold: {f1_optimal_conf:.2f}",
        showarrow=False,
    )

    fig.update_layout(
        barmode="overlay", title="Histogram of Confidence Scores (TP vs FP)", width=800, height=500
    )
    fig.update_xaxes(title_text="Confidence Score", range=[0, 1])
    fig.update_yaxes(title_text="Count", range=[0, tp_y.max() * 1.3])
    # fig.show()
    return fig


markdown_calibration_score = Markdown(
    """
## Calibration Score

This section analyzes confidence scores (or predicted probabilities) that the model generates for every predicted bounding box.
""",
    show_border=False,
    height=80,
)
collapsable_calibaration = Collapse(
    [
        Collapse.Item(
            "What is calibration?",
            "What is calibration?",
            Container(
                [
                    Markdown(
                        "In some applications, it's crucial for a model not only to make accurate predictions but also to provide reliable **confidence levels**. A well-calibrated model aligns its confidence scores with the actual likelihood of predictions being correct. For example, if a model claims 90% confidence for predictions but they are correct only half the time, it is **overconfident**. Conversely, **underconfidence** occurs when a model assigns lower confidence scores than the actual likelihood of its predictions. In the context of autonomous driving, this might cause a vehicle to brake or slow down too frequently, reducing travel efficiency and potentially causing traffic issues.",
                        show_border=False,
                    ),
                ]
            ),
        )
    ]
)
text_info = Text(
    "To evaluate the calibration, we draw a <b>Reliability Diagram</b> and calculate <b>Expected Calibration Error</b> (ECE) and <b>Maximum Calibration Error</b> (MCE).",
    "info",
)
iframe_calibration_curve = IFrame("static/11_01_calibration_curve.html", width=720, height=520)
iframe_confidence_score = IFrame("static/11_02_confidence_score.html", width=820, height=520)
iframe_f1score_at_different_iou = IFrame(
    "static/11_03_f1score_at_different_iou.html", width=820, height=520
)
iframe_confidence_histogram = IFrame(
    "static/11_04_confidence_histogram.html", width=820, height=520
)
markdown_calibration_score = Markdown(
    """
## Calibration Score

This section analyzes confidence scores (or predicted probabilities) that the model generates for every predicted bounding box.

🔽(Collapse) **What is calibration?**

In some applications, it's crucial for a model not only to make accurate predictions but also to provide reliable **confidence levels**. A well-calibrated model aligns its confidence scores with the actual likelihood of predictions being correct. For example, if a model claims 90% confidence for predictions but they are correct only half the time, it is **overconfident**. Conversely, **underconfidence** occurs when a model assigns lower confidence scores than the actual likelihood of its predictions. In the context of autonomous driving, this might cause a vehicle to brake or slow down too frequently, reducing travel efficiency and potentially causing traffic issues.
""",
    show_border=False,
)
container = Container(
    widgets=[
        markdown_calibration_score,
        collapsable_calibaration,
        iframe_calibration_curve,
        text_info,
        iframe_confidence_score,
        iframe_f1score_at_different_iou,
        iframe_confidence_histogram,
    ]
)
