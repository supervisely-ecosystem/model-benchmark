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


checkpoint_name = "YOLOv8-L (COCO 2017 val)"

model_overview = Markdown(
    f"""# {checkpoint_name}

## Overview

- **Model**: [YOLOv8-L]()
- **Year**: 2023
- **Authors**: ultralytics
- **Task type**: object detection
- **Training dataset (?)**: COCO 2017 train
- **Model classes (?)**: (80): a, b, c, … (collapse)
- **Model weights (?)**: [/path/to/yolov8l.pt]()
- **License (?)**: AGPL-3.0
- [GitHub](https://github.com/ultralytics/ultralytics)
""",
    show_border=False,
)

key_metrics = Markdown(
    """## Key Metrics

Here, we comprehensively assess the model's performance by presenting a broad set of metrics, including mAP (mean Average Precision), Precision, Recall, IoU (Intersection over Union), Classification Accuracy, Calibration Score, and Inference Speed.

- **Mean Average Precision (mAP)**: An overall measure of detection performance. mAP calculates the average precision across all classes at different levels of IoU thresholds and precision-recall trade-offs.
- **Precision**: Precision indicates how often the model's predictions are actually correct when it predicts an object. This calculates the ratio of correct detections to the total number of detections made by the model.
- **Recall**: Recall measures the model's ability to find all relevant objects in a dataset. This calculates the ratio of correct detections to the total number of instances in a dataset.
- **Intersection over Union (IoU)**: IoU measures how closely predicted bounding boxes match the actual (ground truth) bounding boxes. It is calculated as the area of overlap between the predicted bounding box and the ground truth bounding box, divided by the area of union of these bounding boxes.
- **Classification Accuracy**: We separately measure the model's capability to correctly classify objects. It’s calculated as a proportion of correctly classified objects among all matched detections. The predicted detection is considered matched if it overlaps a ground true bounding box with IoU higher than 0.5.
- **Calibration Score**: This score represents the consistency of predicted probabilities (or confidence scores) made by the model, evaluating how well the predicted probabilities align with actual outcomes. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
- **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1. The inference speed is important in applications, where real-time object detection is required. Additionally, slower models pour more GPU resources, so their inference cost is higher.
"""
,
    show_border=False,
)
iframe_overview = IFrame("static/01_overview.html", width=620, height=520)

container = Container(
    widgets=[
        model_overview,
        key_metrics,
        iframe_overview,
    ]
)

# Input card with all widgets.
# card = Card(
#     "Overall Metrics",
#     "Description",
#     content=Container(
#         widgets=[
#             key_metrics,
#             iframe_overview,
#         ]
#     ),
#     # content_top_right=change_dataset_button,
#     collapsable=True,
# )
