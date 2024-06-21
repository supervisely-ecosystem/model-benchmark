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


def frequently_confused():
    confusion_matrix = g.m.confusion_matrix()

    # Frequency of confusion as bar chart
    confused_df = g.m.frequently_confused(confusion_matrix, topk_pairs=20)
    confused_name_pairs = confused_df["category_pair"]
    confused_prob = confused_df["probability"]
    x_labels = [f"{pair[0]} - {pair[1]}" for pair in confused_name_pairs]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=x_labels, y=confused_prob, marker=dict(color=confused_prob, colorscale="Reds"))
    )
    fig.update_layout(
        # title="Frequently confused class pairs",
        xaxis_title="Class pair",
        yaxis_title="Probability",
    )
    fig.update_traces(text=confused_prob.round(2))
    # fig.show()
    return fig


markdown = Markdown(
    """
# Frequently confused class pairs

# Frequently Confused Classes

This chart displays the top-20 pairs of classes (or fewer, depending on the dataset and model performance) that are most frequently confused by the model. These are class pairs where the model correctly localizes a bounding box, but incorrectly predicts one class in place of another. The chart indicates the probability of confusion between different pairs of classes. For instance, if the probability of confusion for the pair “car - truck” is 0.15, this means that when the model predicts 'car' or 'truck', there is a 15% chance that it might mistakenly predict one instead of the other. This percentage indicates how often the model confuses these two classes with each other when attempting to make a prediction.

*switch: Probability / Amount*

""",
    show_border=False,
)
iframe_frequently_confused = IFrame("static/09_frequently_confused.html", width=620, height=520)

container = Container(
    widgets=[
        markdown,
        iframe_frequently_confused,
    ]
)

# Input card with all widgets.
card = Card(
    "Frequently confused class pairs",
    "Description",
    content=Container(
        widgets=[
            markdown,
            iframe_frequently_confused,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
