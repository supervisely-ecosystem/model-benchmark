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
    Markdown,
    SelectDataset,
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def confidence_score():
    f1_optimal_conf, best_f1 = g.m_full.get_f1_optimal_conf()
    df = pd.DataFrame(g.score_profile)
    df.columns = ["scores", "Precision", "Recall", "F1"]

    # downsample
    if len(df) > 5000:
        df_down = df.iloc[:: len(df) // 1000]
    else:
        df_down = df

    g.df_down = df_down

    color_map = {
        "Precision": "#1f77b4",
        "Recall": "orange",
    }
    fig = px.line(
        df_down,
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
        x0=f1_optimal_conf,
        x1=f1_optimal_conf,
        y0=0,
        y1=best_f1,
        line=dict(color="gray", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=f1_optimal_conf,
        y=best_f1 + 0.04,
        text=f"F1-optimal threshold: {f1_optimal_conf:.2f}",
        showarrow=False,
    )
    # fig.show()
    fig.write_html(g.STATIC_DIR + "/03_confidence_score.html")


if g.RECALC_PLOTS:
    confidence_score()

markdown = Markdown(
    """
This chart helps determine an optimal confidence threshold for the model based on your requirements. Plotting F1-score against confidence thresholds, you can see how changes in the confidence level affect the balance between precision and recall. The maximum of the F1 score indicates the best balance between precision and recall.

*How is it calculated: To build this plot, we cumulatively calculate precision, recall and F1 for each confidence threshold that the model predicts (scores are sorted in descending order), and draw them on the plot, where x-axis is a score, and y-axis is a metric (precision, recall, f1).*

""",
    show_border=False,
)
# table_model_preds = Table(g.m.prediction_table())
iframe_confidence_score = IFrame("static/03_confidence_score.html", width=620, height=520)


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
