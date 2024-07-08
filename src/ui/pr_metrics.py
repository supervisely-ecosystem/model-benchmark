import os
import random
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui import definitions
from src.utils import CVTask, PlotlyHandler
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    NotificationBox,
    SelectDataset,
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


class RecallVsPrecision(PlotlyHandler):

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        blue_color = "#1f77b4"
        orange_color = "#ff7f0e"
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=g.per_class_metrics_sorted["precision"],
                x=g.per_class_metrics_sorted["category"],
                name="Precision",
                marker=dict(color=blue_color),
            )
        )
        fig.add_trace(
            go.Bar(
                y=g.per_class_metrics_sorted["recall"],
                x=g.per_class_metrics_sorted["category"],
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
        return fig


class Precision(PlotlyHandler):

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="precision")
        fig = px.bar(
            g.per_class_metrics_sorted,
            x="category",
            y="precision",
            title="Per-class Precision (Sorted by F1)",
            color="precision",
            color_continuous_scale="Plasma",
        )
        if len(g.per_class_metrics_sorted) <= 20:
            fig.update_traces(
                text=g.per_class_metrics_sorted["precision"].round(2), textposition="outside"
            )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Precision", range=[0, 1])
        return fig


class Recall(PlotlyHandler):

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        # Per-class Precision bar chart
        # per_class_metrics_df_sorted = per_class_metrics_df.sort_values(by="recall")
        fig = px.bar(
            g.per_class_metrics_sorted,
            x="category",
            y="recall",
            title="Per-class Recall (Sorted by F1)",
            color="recall",
            color_continuous_scale="Plasma",
        )
        if len(g.per_class_metrics_sorted) <= 20:
            fig.update_traces(
                text=g.per_class_metrics_sorted["recall"].round(2), textposition="outside"
            )
        fig.update_xaxes(title_text="Category")
        fig.update_yaxes(title_text="Recall", range=[0, 1])
        return fig


# # table_model_preds = Table(g.m.prediction_table())
# iframe_perclass_PR = IFrame("static/06_1_perclass_PR.html", width=620, height=520)
# iframe_perclass_P = IFrame("static/06_2_perclass_P.html", width=620, height=520)
# iframe_perclass_R = IFrame("static/06_3_perclass_R.html", width=620, height=520)

# base_metrics = g.m.base_metrics()


# markdown_R = Markdown(
#     """## Recall

# This section measures the ability of the model to detect **all relevant instances in the dataset**. In other words, this answers the question: “Of all instances in the dataset, how many of them is the model managed to find out?”

# To measure this, we calculate **Recall**. Recall counts errors, when the model does not detect an object that actually is present in a dataset and should be detected. Recall is calculated as the portion of correct predictions (true positives) over all instances in the dataset (true positives + false negatives).
# """,
#     show_border=False,
# )

# recall_metric = NotificationBox(
#     f"Recall = {base_metrics['recall']:.4f}",
#     f"The model correctly found <b>{g.m.TP_count} of {g.m.TP_count + g.m.FN_count}</b> total instances in the dataset.",
# )

# markdown_R_perclass = Markdown(
#     f"""### Per-class Recall

# This chart further analyzes Recall, breaking it down to each class in separate.

# Since the overall recall is calculated as an average across all classes, we provide a chart showing the recall for each individual class. This illustrates how much each class contributes to the overall recall.

# _Bars in the chart are sorted by <abbr title="{definitions.f1_score}">F1-score</abbr> to keep a unified order of classes between different charts._
# """,
#     show_border=False,
# )


# markdown_P = Markdown(
#     """## Precision

# This section measures the accuracy of all predictions made by the model. In other words, this answers the question: “Of all predictions made by the model, how many of them are actually correct?”.

# To measure this, we calculate **Precision**. Precision counts errors, when the model predicts an object (bounding box), but the image has no objects of the predicted class in this place. Precision is calculated as a portion of correct predictions (true positives) over all model’s predictions (true positives + false positives).
# """,
#     show_border=False,
# )

# precision_metric = NotificationBox(
#     f"Precision = {base_metrics['precision']:.4f}",
#     f"The model correctly predicted <b>{g.m.TP_count} of {g.m.TP_count + g.m.FP_count}</b> predictions made by the model in total.",
# )

# markdown_P_perclass = Markdown(
#     f"""### Per-class Precision

# This chart further analyzes Precision, breaking it down to each class in separate.

# Since the overall precision is computed as an average across all classes, we provide a chart showing the precision for each class individually. This illustrates how much each class contributes to the overall precision.

# _Bars in the chart are sorted by <abbr title="{definitions.f1_score}">F1-score</abbr> to keep a unified order of classes between different charts._""",
#     show_border=False,
# )


# markdown_PR = Markdown(
#     f"""## Recall vs. Precision

# This section compares Precision and Recall on a common graph, identifying **disbalance** between these two.

# _Bars in the chart are sorted by <abbr title="{definitions.f1_score}">F1-score</abbr> to keep a unified order of classes between different charts._
# """,
#     show_border=False,
# )


# container = Container(
#     widgets=[
#         markdown_R,
#         recall_metric,
#         markdown_R_perclass,
#         iframe_perclass_R,
#         markdown_P,
#         precision_metric,
#         markdown_P_perclass,
#         iframe_perclass_P,
#         markdown_PR,
#         iframe_perclass_PR,
#     ]
# )
