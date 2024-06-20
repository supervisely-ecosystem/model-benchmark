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


def f1_score():
    # downsample
    f1s = g.m_full.score_profile_f1s
    f1s_down = f1s[:, :: f1s.shape[1] // 1000]
    iou_names = list(map(lambda x: str(round(x, 2)), g.m.iouThrs.tolist()))
    df = pd.DataFrame(
        np.concatenate([g.dfsp_down["scores"].values[:, None], f1s_down.T], 1),
        columns=["scores"] + iou_names,
    )
    fig = px.line(
        df,
        x="scores",
        y=iou_names,
        # title="F1-Score at different IoU Thresholds",
        labels={"value": "Value", "variable": "IoU threshold", "scores": "Confidence Score"},
        color_discrete_sequence=px.colors.sequential.Viridis,
        width=None,
        height=500,
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

    # add annotations for maximum F1-Score for each IoU threshold
    for i, iou in enumerate(iou_names):
        argmax_f1 = np.nanargmax(f1s[i])
        max_f1 = f1s[i][argmax_f1]
        score = g.score_profile["scores"][argmax_f1]
        fig.add_annotation(
            x=score,
            y=max_f1,
            text=f"Best score: {score:.2f}",
            showarrow=True,
            arrowhead=1,
            arrowcolor="black",
            ax=0,
            ay=-30,
        )

    return fig


markdown = Markdown(
    """
# F1-Score at different IoU Thresholds

This chart shows the outcomes of predictions -- True Positives (TP), False Positives (FP), and False Negatives (FN). The chart helps to assess overall performance of the model in terms of outcomes. For example, if the chart shows a large number of False Negatives, it means that the model is failing to identify many objects that are actually present in the images.

*Hint: You can select a class in the dropdown menu to show outcomes only for the class of interest.*\n
""",
    show_border=False,
)
iframe_f1_score = IFrame("static/04_f1_score.html", width=620, height=520)

container = Container(
    widgets=[
        markdown,
        iframe_f1_score,
    ]
)
# Input card with all widgets.
card = Card(
    "F1-Score at different IoU Thresholds",
    "Description",
    content=Container(
        widgets=[
            # markdown,
            iframe_f1_score,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
