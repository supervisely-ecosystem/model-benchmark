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
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def perclass_ap():
    # AP per-class
    ap_per_class = g.m.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
    # Per-class Average Precision (AP)
    fig = px.scatter_polar(
        r=ap_per_class,
        theta=g.m.cat_names,
        title="Per-class Average Precision (AP)",
        labels=dict(r="Average Precision", theta="Category"),
        width=800,
        height=800,
        range_r=[0, 1],
    )
    # fill points
    fig.update_traces(fill="toself")
    # fig.show()
    fig.write_html(g.STATIC_DIR + "/12_01_perclass.html")


def perclass_outcome_counts():
    # Per-class Counts
    iou_thres = 0

    tp = g.m.true_positives[:, iou_thres]
    fp = g.m.false_positives[:, iou_thres]
    fn = g.m.false_negatives[:, iou_thres]

    # normalize
    support = tp + fn
    tp_rel = tp / support
    fp_rel = fp / support
    fn_rel = fn / support

    # sort by f1
    sort_scores = 2 * tp / (2 * tp + fp + fn)

    K = len(g.m.cat_names)
    sort_indices = np.argsort(sort_scores)
    cat_names_sorted = [g.m.cat_names[i] for i in sort_indices]
    tp_rel, fn_rel, fp_rel = tp_rel[sort_indices], fn_rel[sort_indices], fp_rel[sort_indices]

    # Stacked per-class counts
    data = {
        "count": np.concatenate([tp_rel, fn_rel, fp_rel]),
        "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
        "category": cat_names_sorted * 3,
    }

    df = pd.DataFrame(data)

    color_map = {"TP": "#1fb466", "FN": "#dd3f3f", "FP": "#d5a5a5"}
    fig = px.bar(
        df,
        x="category",
        y="count",
        color="type",
        title="Per-class Outcome Counts",
        labels={"count": "Total Count", "category": "Category"},
        color_discrete_map=color_map,
    )

    # fig.show()
    fig.write_html(g.STATIC_DIR + "/12_02_perclass.html")

    # Stacked per-class counts
    data = {
        "count": np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]]),
        "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
        "category": cat_names_sorted * 3,
    }

    df = pd.DataFrame(data)

    color_map = {"TP": "#1fb466", "FN": "#dd3f3f", "FP": "#d5a5a5"}
    fig = px.bar(
        df,
        x="category",
        y="count",
        color="type",
        title="Per-class Outcome Counts",
        labels={"count": "Total Count", "category": "Category"},
        color_discrete_map=color_map,
    )

    fig.write_html(g.STATIC_DIR + "/12_03_perclass.html")


if g.RECALC_PLOTS:
    perclass_ap()
    perclass_outcome_counts()


markdown_1 = Markdown(
    """

# Per-Class Statistics    
# Per-class Analysis

This section analyzes the model's performance for each specific class. It discovers which classes the model identifies correctly, and which ones it often gets wrong.

## Per-class Average Precision

A quick visual comparison of how well the model performs across all classes. Each axis in the chart represents a different class, and the distance to the center indicates the Average Precision for that class.
""",
    show_border=False,
)
markdown_2 = Markdown(
    """
## Per-class Outcome Counts

A detailed analysis of the model's performance for each class.

1. TP: The graph shows the number of correct detections, where the model accurately identified objects that are actually present in annotations (True Positives).

2. FN: It indicates the number of misses, where the model failed to detect an object that should have been detected according to annotations (False Negatives).

3. FP: The graph displays the number of incorrect (redundant) detections, where the model mistakenly identified something as an object when there wasn't any object in annotations (False Positives).

The graph is sorted by F1-score, which is calculated for each class. F1-score balances all three types of predictions: TP, FN and FP. Thus, the leftmost classes are lower in F1-score, and the rightmost are higher in F1-score.

Each bar is normalized by the number of ground truth instances of the corresponding class, meaning the value of 1.0 represents the number of total ground truth instances in the dataset, and values higher than 1.0 are False Positives (i.e, excessive predictions). You can turn off the normalization switching to absolute values.
""",
    show_border=False,
)
# table_model_preds = Table(g.m.prediction_table())
iframe_perclass_ap = IFrame("static/12_01_perclass.html", width=820, height=820)
iframe_perclass_outcome_counts = IFrame("static/12_02_perclass.html", width=820, height=520)
iframe_perclass_outcome_counts_stacked = IFrame("static/12_03_perclass.html", width=820, height=520)

container = Container(
    widgets=[
        markdown_1,
        iframe_perclass_ap,
        markdown_2,
        iframe_perclass_outcome_counts,
        iframe_perclass_outcome_counts_stacked,
    ]
)

# Input card with all widgets.
card = Card(
    "Per-Class Statistics",
    "Description",
    content=Container(
        widgets=[
            markdown_1,
            iframe_perclass_ap,
            markdown_2,
            iframe_perclass_outcome_counts,
            iframe_perclass_outcome_counts_stacked,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
