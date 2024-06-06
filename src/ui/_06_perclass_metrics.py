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
    SelectDataset,
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def prepare():
    global per_class_metrics_df
    per_class_metrics_df = g.m.per_class_metrics()
    # Per-class Precision and Recall bar chart
    global per_class_metrics_df_sorted
    per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="f1")


def perclass_PR():
    blue_color = "#1f77b4"
    orange_color = "#ff7f0e"
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=per_class_metrics_df_sorted["precision"],
            x=per_class_metrics_df_sorted["category"],
            name="Precision",
            marker=dict(color=blue_color),
        )
    )
    fig.add_trace(
        go.Bar(
            y=per_class_metrics_df_sorted["recall"],
            x=per_class_metrics_df_sorted["category"],
            name="Recall",
            marker=dict(color=orange_color),
        )
    )
    fig.update_layout(
        barmode="group",
        title="Per-class Precision and Recall (Sorted by F1)",
    )
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Value", range=[0, 1])
    # fig.show()
    fig.write_html(g.STATIC_DIR + "/06_1_perclass_PR.html")


def perclass_P():
    # Per-class Precision bar chart
    # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="precision")
    fig = px.bar(
        per_class_metrics_df_sorted,
        x="category",
        y="precision",
        title="Per-class Precision (Sorted by F1)",
        color="precision",
        color_continuous_scale="Plasma",
    )
    if len(per_class_metrics_df_sorted) <= 20:
        fig.update_traces(
            text=per_class_metrics_df_sorted["precision"].round(2), textposition="outside"
        )
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Precision", range=[0, 1])
    # fig.show()
    fig.write_html(g.STATIC_DIR + "/06_2_perclass_P.html")


def perclass_R():
    # Per-class Precision bar chart
    # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="recall")
    fig = px.bar(
        per_class_metrics_df_sorted,
        x="category",
        y="recall",
        title="Per-class Recall (Sorted by F1)",
        color="recall",
        color_continuous_scale="Plasma",
    )
    if len(per_class_metrics_df_sorted) <= 20:
        fig.update_traces(
            text=per_class_metrics_df_sorted["recall"].round(2), textposition="outside"
        )
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Recall", range=[0, 1])
    # fig.show()
    fig.write_html(g.STATIC_DIR + "/06_3_perclass_R.html")


if g.RECALC_PLOTS:
    prepare()
    perclass_PR()
    perclass_P()
    perclass_R()


# table_model_preds = Table(g.m.prediction_table())
iframe_perclass_PR = IFrame("static/06_1_perclass_PR.html", width=620, height=520)
iframe_perclass_P = IFrame("static/06_2_perclass_P.html", width=620, height=520)
iframe_perclass_R = IFrame("static/06_3_perclass_R.html", width=620, height=520)

# txt1 = Text("Per-class Precision and Recall (Sorted by F1)")
# txt2 = Text("Per-class Precision (Sorted by F1)")
# txt3 = Text("Per-class Recall (Sorted by F1)")

# Input card with all widgets.
card = Card(
    "Outcome Counts",
    "Description",
    content=Container(
        widgets=[
            # txt1,
            iframe_perclass_PR,
            # txt2,
            iframe_perclass_P,
            # txt3,
            iframe_perclass_R,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
