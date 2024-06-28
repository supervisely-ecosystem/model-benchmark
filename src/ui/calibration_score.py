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
from src.ui import definitions
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Collapse,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    NotificationBox,
    SelectDataset,
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def reliability_diagram():
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
        # title="Calibration Curve (only positive predictions)",
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

    color_map = {
        "Precision": "#1f77b4",
        "Recall": "orange",
    }
    fig = px.line(
        g.dfsp_down,
        x="scores",
        y=["Precision", "Recall", "F1"],
        # title="Confidence Score Profile",
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
        # title="F1-Score at different IoU Thresholds",
        labels={"value": "Value", "variable": "IoU threshold", "scores": "Confidence Score"},
        color_discrete_sequence=px.colors.sequential.Viridis,
        width=None,
        height=500,
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

    # add annotations for maximum F1-Score for each IoU threshold
    for i, iou in enumerate(iou_names):
        argmax_f1 = np.nanargmax(f1s[i])
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


def confidence_distribution():
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


markdown_calibration_score_1 = Markdown(
    f"""
## Calibration Score

This section analyzes <abbr title="{definitions.confidence_score}">confidence scores</abbr> (or predicted probabilities) that the model generates for every predicted bounding box.
""",
    show_border=False,
    height=80,
)
collapsable_what_is_calibration_curve = Collapse(
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
markdown_calibration_score_2 = Markdown(
    """
To evaluate the calibration, we draw a <b>Reliability Diagram</b> and calculate <b>Expected Calibration Error</b> (ECE).
""",
    show_border=False,
    height=30,
)
# text_info = Text(
#     "To evaluate the calibration, we draw a <b>Reliability Diagram</b> and calculate <b>Expected Calibration Error</b> (ECE) and <b>Maximum Calibration Error</b> (MCE).",
#     "info",
# )
markdown_reliability_diagram = Markdown(
    """
### Reliability Diagram

Reliability diagram, also known as a Calibration curve, helps in understanding whether the confidence scores of detections accurately represent the true probability of a correct detection. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
""",
    show_border=False,
)
collapsable_reliabilty_diagram = Collapse(
    [
        Collapse.Item(
            "How to interpret the Calibration curve",
            "How to interpret the Calibration curve",
            Container(
                [
                    Markdown(
                        """
1. **The curve is above the Ideal Line (Underconfidence):** If the calibration curve is consistently above the ideal line, this indicates underconfidence. The model’s predictions are more correct than the confidence scores suggest. For example, if the model predicts a detection with 70% confidence but, empirically, 90% of such detections are correct, the model is underconfident.
2. **The curve is below the Ideal Line (Overconfidence):** If the calibration curve is below the ideal line, the model exhibits overconfidence. This means it is too sure of its predictions. For instance, if the model predicts with 80% confidence but only 60% of these predictions are correct, it is overconfident.

To quantify the calibration, we calculate **Expected Calibration Error (ECE).** Intuitively, ECE can be viewed as a deviation of the model's calibration curve from the diagonal line, that corresponds to a perfectly calibrated model. When ECE is high, we can not trust predicted probabilities so much.

**Note:** ECE is a measure of **error**. The lower the ECE, the better the calibration. A perfectly calibrated model has an ECE of 0.
""",
                        show_border=False,
                    ),
                ]
            ),
        )
    ]
)
notibox_ECE = NotificationBox(f"Expected Calibration Error (ECE) = {g.m_full.calibration_metrics.expected_calibration_error():.4f}")
iframe_reliability_diagram = IFrame("static/11_01_reliability_diagram.html", width=720, height=520)
markdown_confidence_score_1 = Markdown(
    f"""
## Confidence Score Profile

This section is going deeper in analyzing confidence scores. It gives you an intuition about how these scores are distributed and helps to find the best <abbr title="{definitions.confidence_threshold}">confidence threshold</abbr> suitable for your task or application.
""",
    show_border=False,
)
iframe_confidence_score = IFrame("static/11_02_confidence_score.html", width=820, height=520)
markdown_confidence_score_2 = Markdown(
    """
This chart provides a comprehensive view about predicted confidence scores. It is used to determine an **optimal confidence threshold** based on your requirements.

The plot shows you what the metrics will be if you choose a specific confidence threshold. For example, if you set the threshold to 0.32, you can see on the plot what the precision, recall and f1-score will be for this threshold.
""",
    show_border=False,
)
collapsable_howto_plot_confidence_score = Collapse(
    [
        Collapse.Item(
            "How to plot Confidence Profile?",
            "How to plot Confidence Profile?",
            Container(
                [
                    Markdown(
                        """
First, we sort all predictions by confidence scores from highest to lowest. As we iterate over each prediction we calculate the cumulative precision, recall and f1-score so far. Each prediction is plotted as a point on a graph, with a confidence score on the x-axis and one of three metrics on the y-axis (precision, recall, f1-score).
""",
                        show_border=False,
                    ),
                ]
            ),
        )
    ]
)
markdown_calibration_score_3 = Markdown("**How to find an optimal threshold:** you can find the maximum of the f1-score line on the plot, and the confidence score (X-axis) under this maximum corresponds to F1-optimal confidence threshold. This threshold ensures the balance between precision and recall. You can select a threshold according to your desired trade-offs.")
notibox_F1 = NotificationBox(f"F1-optimal confidence threshold = {g.m_full.get_f1_optimal_conf()[0]:.4f}")
markdown_f1_at_ious = Markdown(
    f"""### Confidence Profile at Different IoU thresholds

This chart breaks down the Confidence Profile into multiple curves, each for one <abbr title="{definitions.iou_threshold}">IoU threshold</abbr>. In this way you can understand how the f1-optimal confidence threshold changes with various IoU thresholds. Higher IoU thresholds mean that the model should align bounding boxes very close to ground truth bounding boxes.
""",
    show_border=False,
)
iframe_f1score_at_different_iou = IFrame(
    "static/11_03_f1score_at_different_iou.html", width=820, height=520
)
markdown_confidence_distribution = Markdown(
    f"""
### Confidence Distribution
    
This graph helps to assess whether high confidence scores correlate with correct detections (<abbr title="{definitions.true_positives}">True Positives</abbr>) and low confidence scores are mostly associated with incorrect detections (<abbr title="{definitions.false_positives}">False Positives</abbr>).

Additionally, it provides a view of how predicted probabilities are distributed. Whether the model skews probabilities to lower or higher values, leading to imbalance?

Ideally, the histogram for TP predictions should have higher confidence, indicating that the model is sure about its correct predictions, and the FP predictions should have very low confidence, or not present at all.
""",
    show_border=False,
)
iframe_confidence_distribution = IFrame(
    "static/11_04_confidence_distribution.html", width=820, height=520
)

container = Container(
    widgets=[
        markdown_calibration_score_1,
        collapsable_what_is_calibration_curve,
        markdown_calibration_score_2,
        markdown_reliability_diagram,
        collapsable_reliabilty_diagram,
        notibox_ECE,
        iframe_reliability_diagram,
        markdown_confidence_score_1,
        iframe_confidence_score,
        markdown_confidence_score_2,
        collapsable_howto_plot_confidence_score,
        markdown_calibration_score_3,
        notibox_F1,
        markdown_f1_at_ious,
        iframe_f1score_at_different_iou,
        markdown_confidence_distribution,
        iframe_confidence_distribution,
    ]
)
