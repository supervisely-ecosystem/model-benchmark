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
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    SelectDataset,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def iou_distribution():
    fig = go.Figure()
    nbins = 40
    fig.add_trace(go.Histogram(x=g.m.ious, nbinsx=nbins))
    fig.update_layout(
        # title="IoU Distribution",
        xaxis_title="IoU",
        yaxis_title="Count",
        width=600,
        height=500,
    )

    # Add annotation for mean IoU as vertical line
    mean_iou = g.m.ious.mean()
    y1 = len(g.m.ious) // nbins
    fig.add_shape(
        type="line",
        x0=mean_iou,
        x1=mean_iou,
        y0=0,
        y1=y1,
        line=dict(color="orange", width=2, dash="dash"),
    )
    fig.add_annotation(x=mean_iou, y=y1, text=f"Mean IoU: {mean_iou:.2f}", showarrow=False)
    # fig.show()
    return fig


markdown = Markdown(
    """
# IoU Distribution

# Localization Accuracy (IoU)

This section measures how closely the predicted bounding boxes match the actual (ground truth) bounding boxes.

**Avg. IoU** (?) = 0.86

<img src="https://github.com/dataset-ninja/model-benchmark-template/assets/78355358/8d7c63d0-2f3b-4f3f-9fd8-c6383a4bfba4" alt="alt text" width="300" />


*(?) IoU (Intersection over Union) is calculated by dividing the area of overlap between the predicted bounding box and the ground truth bounding box by the area of union of these two boxes. IoU measures the extent of overlap of two instances.*

## IoU Distribution

A histogram of the Intersection over Union (IoU) scores across detections made by the model, where the x-axis represents the IoU score from 0.5 to 1.0, and the y-axis represents the frequency of detections for each score range.


""",
    show_border=False,
)
iframe_iou_distribution = IFrame("static/10_iou_distribution.html", width=620, height=520)

container = Container(
    widgets=[
        markdown,
        iframe_iou_distribution,
    ]
)

# Input card with all widgets.
card = Card(
    "IoU Distribution",
    "Description",
    content=Container(
        widgets=[
            markdown,
            iframe_iou_distribution,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
