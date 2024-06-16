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
import src.ui.settings as settings
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    SelectDataset,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def overall():
    from importlib import reload

    reload(metric_provider)
    m_full = metric_provider.MetricProvider(
        g.eval_data["matches"],
        g.eval_data["coco_metrics"],
        g.eval_data["params"],
        g.cocoGt,
        g.cocoDt,
    )
    g.m_full = m_full

    score_profile = m_full.confidence_score_profile()
    g.score_profile = score_profile
    # score_profile = m_full.confidence_score_profile_v0()
    f1_optimal_conf, best_f1 = m_full.get_f1_optimal_conf()
    print(f"F1-Optimal confidence: {f1_optimal_conf:.4f} with f1: {best_f1:.4f}")

    matches_thresholded = metric_provider.filter_by_conf(g.eval_data["matches"], f1_optimal_conf)
    m = metric_provider.MetricProvider(
        matches_thresholded, g.eval_data["coco_metrics"], g.eval_data["params"], g.cocoGt, g.cocoDt
    )
    # m.base_metrics()
    g.m = m
    # Overall Metrics
    base_metrics = m.base_metrics()
    r = list(base_metrics.values())
    theta = [metric_provider.METRIC_NAMES[k] for k in base_metrics.keys()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r + [r[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name="Overall Metrics",
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0.0, 1.0]), angularaxis=dict(rotation=90, direction="clockwise")
        ),
        # title="Overall Metrics",
        width=600,
        height=500,
    )

    fig.write_html(g.STATIC_DIR + "/01_overview.html")


if g.RECALC_PLOTS:
    overall()

markdown = Markdown(
    """
# Overall Metrics    

Overview of the model performance across a set of key metrics. Greater values are better. \n\n

При наведении на (?) вопросик на бар чартах:\n
* **Mean Average Precision (mAP)**: A measure of the precision-recall trade-off across different thresholds, reflecting the model's overall detection performance.\n
* **Precision**: The ratio of true positive detections to the total number of positive detections made by the model, indicating its accuracy in identifying objects correctly.\n
* **Recall**: The ratio of true positive detections to the total number of actual objects, measuring the model's ability to find all relevant objects.\n
* **Intersection over Union (IoU)**: The overlap between the predicted bounding boxes and the ground truth, providing insight into the spatial accuracy of detections.\n
* **Classification Accuracy**: The proportion of correctly classified objects among all detected objects, highlighting the model's capability in correctly labeling objects.\n
* **Calibration Score**: A metric evaluating how well the predicted probabilities align with the actual outcomes, assessing the confidence calibration of the model. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.\n
* **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1 on the full COCO dataset on RTX3060 GPU.\n

""",
    show_border=False,
)
iframe_overview = IFrame("static/01_overview.html", width=620, height=520)

container = Container(
    widgets=[
        markdown,
        iframe_overview,
    ]
)

# Input card with all widgets.
card = Card(
    "Overall Metrics",
    "Description",
    content=Container(
        widgets=[
            markdown,
            iframe_overview,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
