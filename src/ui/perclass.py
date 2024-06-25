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
    NotificationBox,
    OneOf,
    RadioGroup,
    SelectDataset,
    Switch,
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
    return fig


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
        # title="Per-class Outcome Counts",
        labels={"count": "Total Count", "category": "Category"},
        color_discrete_map=color_map,
    )

    # fig.show()

    # Stacked per-class counts
    data = {
        "count": np.concatenate([tp[sort_indices], fn[sort_indices], fp[sort_indices]]),
        "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
        "category": cat_names_sorted * 3,
    }

    df = pd.DataFrame(data)

    color_map = {"TP": "#1fb466", "FN": "#dd3f3f", "FP": "#d5a5a5"}
    fig_ = px.bar(
        df,
        x="category",
        y="count",
        color="type",
        # title="Per-class Outcome Counts",
        labels={"count": "Total Count", "category": "Category"},
        color_discrete_map=color_map,
    )

    return fig, fig_


markdown_class_comparison = Markdown(
    """
## Class Comparison

This section analyzes the model's performance for all classes in a common plot. It discovers which classes the model identifies correctly, and which ones it often gets wrong.
""",
    show_border=False,
)
markdown_class_ap = Markdown(
    """
## Average Precision by Class

A quick visual comparison of the model performance across all classes. Each axis in the chart represents a different class, and the distance to the center indicates the Average Precision for that class.""",
    show_border=False,
)
iframe_perclass_ap = IFrame("static/12_01_perclass.html", width=820, height=820)
markdown_class_outcome_counts_1 = Markdown(
    """
### Outcome Counts by Class

This chart breaks down all predictions into True Positives (TP), False Positives (FP), and False Negatives (FN) by classes. This helps to visually assess the type of errors the model often encounters for each class.
""",
    show_border=False,
    height=80,
)

collapsable_normalization = Collapse(
    [
        Collapse.Item(
            "Normalization",
            "Normalization",
            Container(
                [
                    Markdown(
                        "By default, the normalization is used for better intraclass comparison. The total outcome counts are divided by the number of ground truth instances of the corresponding class. This is useful, because the sum of TP + FN always gives 1.0, representing all ground truth instances for a class, that gives a visual understanding of what portion of instances the model detected. So, if a green bar (TP outcomes) reaches the 1.0, this means the model is managed to predict all objects for the class. Everything that is higher than 1.0 corresponds to False Positives, i.e, redundant predictions. You can turn off the normalization switching to absolute values.",
                        show_border=False,
                    ),
                ]
            ),
        )
    ]
)

iframe_perclass_outcome_counts_normalized = IFrame(
    "static/12_02_perclass.html", width=820, height=520
)
iframe_perclass_outcome_counts_absolute = IFrame(
    "static/12_03_perclass.html", width=820, height=520
)

radio_group = RadioGroup(
    [
        RadioGroup.Item(
            "normalized", "Normalized", content=iframe_perclass_outcome_counts_normalized
        ),
        RadioGroup.Item("absolute", "Absolute", content=iframe_perclass_outcome_counts_absolute),
    ]
)
radio_one_of = OneOf(radio_group)

markdown_class_outcome_counts_2 = Markdown(
    """
_Bars in the chart are sorted by F1-score to keep a unified order of classes between different charts._ You can choose the plot view between normalized and absolute values:
""",
    show_border=False,
    height=50,
)
container = Container(
    widgets=[
        markdown_class_ap,
        iframe_perclass_ap,
        markdown_class_outcome_counts_1,
        collapsable_normalization,
        markdown_class_outcome_counts_2,
        radio_group,
        radio_one_of,
    ]
)
