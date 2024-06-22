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


def outcome_counts():
    # Outcome counts
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[g.m.TP_count],
            y=["Outcome"],
            name="TP",
            orientation="h",
            marker=dict(color="#1fb466"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=[g.m.FN_count],
            y=["Outcome"],
            name="FN",
            orientation="h",
            marker=dict(color="#dd3f3f"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=[g.m.FP_count],
            y=["Outcome"],
            name="FP",
            orientation="h",
            marker=dict(color="#d5a5a5"),
        )
    )
    fig.update_layout(
        barmode="stack",
        # title="Outcome Counts",
        width=600,
        height=300,
    )
    fig.update_xaxes(title_text="Count")
    fig.update_yaxes(tickangle=-90)

    return fig


# if g.RECALC_PLOTS:
#     outcome_counts()
markdown = Markdown(
    """## Outcome Counts

This chart is used to evaluate the overall model performance by breaking down all predictions into True Positives (TP), False Positives (FP), and False Negatives (FN). This helps to visually assess the type of errors the model often encounters.
""",
    show_border=False,
)
# table_model_preds = Table(g.m.prediction_table())
iframe_outcome_counts = IFrame("static/05_outcome_counts.html", width=620, height=320)

container = Container(
    widgets=[
        markdown,
        iframe_outcome_counts,
    ]
)
