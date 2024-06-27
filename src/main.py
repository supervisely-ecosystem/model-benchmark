import os
import json
import pickle
from typing import List
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly import express as px
from pycocotools.coco import COCO

import supervisely as sly
from supervisely.app.widgets import Container, PlotlyChart
from supervisely.nn.benchmark.metric_provider import MetricProvider

from src.click_data import ClickData
from src.utils import IdMapper


gt_project_id = 36401

base_dir = "APP_DATA/COCO 2017 val - Serve YOLO (v8, v9)"
gt_path = os.path.join(base_dir, "gt_project")
dt_path = os.path.join(base_dir, "dt_project")

eval_data_path = os.path.join(base_dir, "eval_data.pkl")
with open(eval_data_path, "rb") as f:
    eval_data = pickle.load(f)

cocoGt_path = os.path.join(base_dir, "cocoGt.json")
cocoDt_path = os.path.join(base_dir, "cocoDt.json")
with open(cocoGt_path, 'r') as f:
    cocoGt_dataset= json.load(f)
with open(cocoDt_path, 'r') as f:
    cocoDt_dataset = json.load(f)

cocoGt = COCO()
cocoGt.dataset = cocoGt_dataset
cocoGt.createIndex()
cocoDt = cocoGt.loadRes(cocoDt_dataset['annotations'])

m = MetricProvider(eval_data['matches'], eval_data['coco_metrics'], eval_data['params'], cocoGt, cocoDt)

api = sly.Api()
matches = eval_data['matches']
dt_project_id = 39052

gt_id_mapper = IdMapper(cocoGt_dataset)
dt_id_mapper = IdMapper(cocoDt_dataset)

click_data = ClickData(m, gt_id_mapper, dt_id_mapper)
click_data.create_data()


def get_figure():
    iou_thres = 0

    tp = m.true_positives[:, iou_thres]
    fp = m.false_positives[:, iou_thres]
    fn = m.false_negatives[:, iou_thres]

    # normalize
    support = tp + fn
    tp_rel = tp / support
    fp_rel = fp / support
    fn_rel = fn / support

    # sort by f1
    sort_scores = 2 * tp / (2 * tp + fp + fn)

    K = len(m.cat_names)
    sort_indices = np.argsort(sort_scores)
    cat_names_sorted = [m.cat_names[i] for i in sort_indices]
    tp_rel, fn_rel, fp_rel = tp_rel[sort_indices], fn_rel[sort_indices], fp_rel[sort_indices]

    # Stacked per-class counts
    data = {
        "count": np.concatenate([tp_rel, fn_rel, fp_rel]),
        "type": ["TP"] * K + ["FN"] * K + ["FP"] * K,
        "category": cat_names_sorted * 3,
    }

    df = pd.DataFrame(data)

    color_map = {"TP": "#1fb466", "FN": "#dd3f3f", "FP": "#d5a5a5"}
    fig = px.bar(
        df,
        x="category",
        y="count",
        color="type",
        title="Per-class Outcome Counts",
        labels={"count": "Total Count", "category": "Category"},
        color_discrete_map=color_map,
    )

    return fig


fig = get_figure()
chart = PlotlyChart(fig)

@chart.click
def on_click(datapoints: List[PlotlyChart.ClickedDataPoint]):
    datapoint = datapoints[0]
    outcome = datapoint.label
    cat_name = datapoint.x
    print(click_data.outcome_counts_by_class[cat_name][outcome])

layout = Container(
    widgets=[
        chart
    ]
)

app = sly.Application(layout=layout)
