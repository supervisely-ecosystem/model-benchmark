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
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


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


markdown = Markdown(
    """
# Confidence Score Profile

This chart helps determine an optimal confidence threshold for the model based on your requirements. Plotting F1-score against confidence thresholds, you can see how changes in the confidence level affect the balance between precision and recall. The maximum of the F1 score indicates the best balance between precision and recall.

*How is it calculated: To build this plot, we cumulatively calculate precision, recall and F1 for each confidence threshold that the model predicts (scores are sorted in descending order), and draw them on the plot, where x-axis is a score, and y-axis is a metric (precision, recall, f1).*

""",
    show_border=False,
)
iframe_confidence_score = IFrame("static/03_confidence_score.html", width=620, height=520)

container = Container(
    widgets=[
        markdown,
        iframe_confidence_score,
    ]
)

# Input card with all widgets.
card = Card(
    "Confidence Score Profile",
    "Description",
    content=Container(
        widgets=[
            markdown,
            iframe_confidence_score,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
