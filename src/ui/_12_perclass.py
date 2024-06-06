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


def perclass_ap():
    # AP per-class
    ap_per_class = g.m.coco_precision[:, :, :, 0, 2].mean(axis=(0, 1))
    # Per-class Average Precision (AP)
    fig = px.scatter_polar(
        r=ap_per_class,
        theta=g.m.cat_names,
        title="Per-class Average Precision (AP)",
        labels=dict(r="Average Precision", theta="Category"),
        width=800,
        height=800,
        range_r=[0, 1],
    )
    # fill points
    fig.update_traces(fill="toself")
    # fig.show()
    fig.write_html(g.STATIC_DIR + "/12_01_perclass_ap.html")


def perclass_outcome_counts():
    score_profile = g.m_full.confidence_score_profile()
    f1_optimal_conf, best_f1 = g.m_full.get_f1_optimal_conf()
    global df_down

    df = pd.DataFrame(score_profile)
    df.columns = ["scores", "Precision", "Recall", "F1"]

    # downsample
    if len(df) > 5000:
        df_down = df.iloc[:: len(df) // 1000]
    else:
        df_down = df

    color_map = {
        "Precision": "#1f77b4",
        "Recall": "orange",
    }
    fig = px.line(
        df_down,
        x="scores",
        y=["Precision", "Recall", "F1"],
        title="Confidence Score Profile",
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
    fig.write_html(g.STATIC_DIR + "/12_02_perclass_outcome_counts.html")


def perclass_outcome_counts_stacked():
    score_profile = g.m_full.confidence_score_profile()
    f1s = g.m_full.score_profile_f1s

    # downsample
    f1s_down = f1s[:, :: f1s.shape[1] // 1000]
    iou_names = list(map(lambda x: str(round(x, 2)), g.m.iouThrs.tolist()))
    df = pd.DataFrame(
        np.concatenate([df_down["scores"].values[:, None], f1s_down.T], 1),
        columns=["scores"] + iou_names,
    )

    fig = px.line(
        df,
        x="scores",
        y=iou_names,
        title="F1-Score at different IoU Thresholds",
        labels={"value": "Value", "variable": "IoU threshold", "scores": "Confidence Score"},
        color_discrete_sequence=px.colors.sequential.Viridis,
        width=None,
        height=500,
    )
    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1], tick0=0, dtick=0.1))

    # add annotations for maximum F1-Score for each IoU threshold
    for i, iou in enumerate(iou_names):
        argmax_f1 = f1s[i].argmax()
        max_f1 = f1s[i][argmax_f1]
        score = score_profile["scores"][argmax_f1]
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

    # fig.show()
    fig.write_html(g.STATIC_DIR + "/12_03_perclass_outcome_counts_stacked.html")


if g.RECALC_PLOTS:
    perclass_ap()
    perclass_outcome_counts()
    perclass_outcome_counts_stacked()

txt = Text("text")
# table_model_preds = Table(g.m.prediction_table())
iframe_perclass_ap = IFrame("static/12_01_perclass_ap.html", width=820, height=820)
iframe_perclass_outcome_counts = IFrame(
    "static/12_02_perclass_outcome_counts.html", width=820, height=520
)
iframe_perclass_outcome_counts_stacked = IFrame(
    "static/12_03_perclass_outcome_counts_stacked.html", width=820, height=520
)


# Input card with all widgets.
card = Card(
    "Per-Class Statistics",
    "Description",
    content=Container(
        widgets=[
            txt,
            iframe_perclass_ap,
            iframe_perclass_outcome_counts,
            iframe_perclass_outcome_counts_stacked,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
