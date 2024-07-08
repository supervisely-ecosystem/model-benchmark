import os
import random
from collections import defaultdict
from typing import Optional

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
    Collapse,
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


class PRCurve(PlotlyHandler):

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        # Precision-Recall curve
        fig = px.line(
            x=g.m.recThrs,
            y=g.m.pr_curve().mean(-1),
            # title="Precision-Recall Curve",
            labels={"x": "Recall", "y": "Precision"},
            width=600,
            height=500,
        )
        fig.data[0].name = "Model"
        fig.data[0].showlegend = True
        fig.update_traces(fill="tozeroy", line=dict(color="#1f77b4"))
        fig.add_trace(
            go.Scatter(
                x=g.m.recThrs,
                y=[1] * len(g.m.recThrs),
                name="Perfect",
                line=dict(color="orange", dash="dash"),
                showlegend=True,
            )
        )
        fig.add_annotation(
            text=f"mAP = {g.m.base_metrics()['mAP']:.2f}",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.92,
            showarrow=False,
            bgcolor="white",
        )

        # fig.show()
        return fig


class PRCurveByClass(PlotlyHandler):

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:

        # Precision-Recall curve per-class
        df = pd.DataFrame(g.m.pr_curve(), columns=g.m.cat_names)

        fig = px.line(
            df,
            x=g.m.recThrs,
            y=df.columns,
            # title="Precision-Recall Curve per Class",
            labels={"x": "Recall", "value": "Precision", "variable": "Category"},
            color_discrete_sequence=px.colors.qualitative.Prism,
            width=800,
            height=600,
        )

        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(range=[0, 1])
        # fig.show()

        return fig


# iframe_pr = IFrame("static/07_01_pr_curve.html", width=620, height=520)
# iframe_pr_perclass = IFrame("static/07_02_pr_curve_perclass.html", width=820, height=620)

# markdown_pr_curve = Markdown(
#     f"""
# ## Precision-Recall Curve

# Precision-Recall curve is an overall performance indicator. It helps to visually assess both precision and recall for all predictions made by the model on the whole dataset. This gives you an understanding of how precision changes as you attempt to increase recall, providing a view of **trade-offs between precision and recall** <abbr title="{definitions.f1_score}">(?)</abbr>. Ideally, a high-quality model will maintain strong precision as recall increases. This means that as you move from left to right on the curve, there should not be a significant drop in precision. Such a model is capable of finding many relevant instances, maintaining a high level of precision.
# """,
#     show_border=False,
# )

# collapsables = Collapse(
#     [
#         # Collapse.Item(
#         #     "About Trade-offs between precision and recall",
#         #     "About Trade-offs between precision and recall",
#         #     Container(
#         #         [
#         #             Markdown(
#         #                 "A system with high recall but low precision returns many results, but most of its predictions are incorrect or redundant (false positive). A system with high precision but low recall is just the opposite, returning very few results, most of its predictions are correct. An ideal system with high precision and high recall will return many results, with all results predicted correctly.",
#         #                 show_border=False,
#         #             ),
#         #         ]
#         #     ),
#         # ),
#         Collapse.Item(
#             "What is PR curve?",
#             "What is PR curve?",
#             Container(
#                 [
#                     Markdown(
#                         f"""
# Imagine you sort all the predictions by their <abbr title="{definitions.confidence_score}">confidence scores</abbr> from highest to lowest and write it down in a table. As you iterate over each sorted prediction, you classify it as a <abbr title="{definitions.true_positives}">true positive</abbr> (TP) or a <abbr title="{definitions.false_positives}">false positive</abbr> (FP). For each prediction, you then calculate the cumulative precision and recall so far. Each prediction is plotted as a point on a graph, with recall on the x-axis and precision on the y-axis. Now you have a plot very similar to the PR-curve, but it appears as a zig-zag curve due to variations as you move from one prediction to the next.

# **Forming the Actual PR Curve**: The true PR curve is derived by plotting only the maximum precision value for each recall level across all thresholds.
# This means you connect only the highest points of precision for each segment of recall, smoothing out the zig-zags and forming a curve that typically slopes downward as recall increases.
# """,
#                         show_border=False,
#                     ),
#                 ]
#             ),
#         ),
#     ]
# )


# notibox_map = NotificationBox(f"mAP = {g.m.base_metrics()['mAP']:.2f}")

# markdown_pr_by_class = Markdown(
#     """
# ### Precision-Recall Curve by Class

# In this plot, you can evaluate PR curve for each class individually.""",
#     show_border=False,
#     height=70,
# )

# container = Container(
#     widgets=[
#         markdown_pr_curve,
#         collapsables,
#         notibox_map,
#         iframe_pr,
#         markdown_pr_by_class,
#         iframe_pr_perclass,
#     ]
# )
