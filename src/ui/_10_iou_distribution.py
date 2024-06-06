import os
import random
from collections import defaultdict

# %%
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
    fig.write_html(g.STATIC_DIR + "/10_iou_distribution.html")


if g.RECALC_PLOTS:
    iou_distribution()

txt = Text("text")
iframe_iou_distribution = IFrame("static/10_iou_distribution.html", width=620, height=520)


# Input card with all widgets.
card = Card(
    "IoU Distribution",
    "Description",
    content=Container(
        widgets=[
            txt,
            iframe_iou_distribution,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)


def clean_static_dir():
    # * Utility function to clean static directory, it can be securely removed if not needed.
    static_files = os.listdir(g.STATIC_DIR)

    sly.logger.debug(f"Cleaning static directory. Number of files to delete: {len(static_files)}.")

    for static_file in static_files:
        os.remove(os.path.join(g.STATIC_DIR, static_file))
