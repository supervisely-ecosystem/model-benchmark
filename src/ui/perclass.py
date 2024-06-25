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
markdown_class_outcome_counts = Markdown(
    """
### Outcome Counts by Class

This chart breaks down all predictions into True Positives (TP), False Positives (FP), and False Negatives (FN) by classes. This helps to visually assess the type of errors the model often encounters for each class.
""",
    show_border=False,
)

iframe_perclass_outcome_counts_relative = IFrame(
    "static/12_02_perclass.html", width=820, height=520
)
iframe_perclass_outcome_counts_absolute = IFrame(
    "static/12_03_perclass.html", width=820, height=520
)

swicther = Switch(
    switched=True,
    on_text="Relative",
    off_text="Absolute",
    width=100,
    on_content=iframe_perclass_outcome_counts_relative,
    off_content=iframe_perclass_outcome_counts_absolute,
)
switch_one_of = OneOf(swicther)

markdown_normalization = Markdown(
    """
#### Normalization

By default, the normalization is used for better intraclass comparison. The total outcome counts are divided by the number of ground truth instances of the corresponding class. This is useful, because the sum of TP + FN always gives 1.0, representing all ground truth instances for a class, that gives a visual understanding of what portion of instances the model detected. So, if a green bar (TP outcomes) reaches the 1.0, this means the model is managed to predict all objects for the class. Everything that is higher than 1.0 corresponds to False Positives, i.e, redundant predictions. You can turn off the normalization switching to absolute values.

_Bars in the chart are sorted by F1-score to keep a unified order of classes between different charts._
_switch to absolute values:_
""",
    show_border=False,
)
markdown_inference_speed_1 = Markdown(
    """
## Inference speed

We evaluate the inference speed in two scenarios: real-time inference (batch size is 1), and batch processing. We also run the model in optimized runtime environments, such as ONNX Runtime and Tensor RT, using consistent hardware. This approach provides a fair comparison of model efficiency and speed. To assess the inference speed we run the model forward 100 times and average it.
""",
    show_border=False,
)
collapsables = Collapse(
    [
        Collapse.Item(
            "Methodology",
            "Methodology",
            Container(
                [
                    Markdown(
                        """
Setting 1: **Real-time processing**

We measure the time spent processing each image individually by setting batch size to 1. This simulates real-time data processing conditions, such as those encountered in video streams, ensuring the model performs effectively in scenarios where data is processed frame by frame.

Setting 2: **Parallel processing**

To evaluate the model's efficiency in parallel processing, we measure the processing speed with batch size of 8 and 16. This helps us understand how well the model scales when processing multiple images simultaneously, which is crucial for applications requiring high throughput.

Setting 3: **Optimized runtime**

We run the model in various runtime environments, including **ONNX Runtime** and **TensorRT**. This is important because python code can be suboptimal. These runtimes often provide significant performance improvements.
""",
                        show_border=False,
                    ),
                ]
            ),
        )
    ]
)
markdown_inference_speed_2 = Markdown(
    """
#### Consistent hardware for fair comparison

To ensure a fair comparison, we use a single hardware setup, specifically an NVIDIA RTX 3060 GPU.

#### Inference details

We divide the inference process into three stages: **preprocess, inference,** and **postprocess** to provide insights into where optimization efforts should be focused. Additionally, it gives us another verification level to ensure that time is measured correctly for each model.

#### Preprocess 

The stage where images are prepared for input into the model. This includes image reading, resizing, and any necessary transformations.

#### Inference

The main computation phase where the _forward_ pass of the model is running. **Note:** we include not only the forward pass, but also modules like NMS (Non-Maximum Suppression), decoding module, and everything that is done to get a **meaningful** prediction.

#### Postprocess

This stage includes tasks such as resizing output masks, aligning predictions with the input image, converting bounding boxes into a specific format or filtering out low-confidence detections.
""",
    show_border=False,
)
container = Container(
    widgets=[
        markdown_class_ap,
        iframe_perclass_ap,
        markdown_class_outcome_counts,
        swicther,
        switch_one_of,
        markdown_normalization,
        markdown_inference_speed_1,
        collapsables,
        markdown_inference_speed_2,
    ]
)
