import json
from typing import *

import plotly.graph_objects as go
from tqdm import tqdm

import src.globals as g
import supervisely as sly

# from src.ui.outcome_counts import plotly_outcome_counts
from supervisely._utils import camel_to_snake
from supervisely.app.widgets import Button, Card, Container, Sidebar, Text


def main_func():

    cocoGt_path = "APP_DATA/data/cocoGt.json"  # cocoGt_remap.json"
    cocoDt_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/cocoDt.json"
    eval_data_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/eval_data.pkl"
    api = sly.Api.from_env()

    # bm = sly.nn.MetricLoader(cocoGt_path, cocoDt_path, eval_data_path)
    # bm.upload_layout(g.TEAM_ID, "/model-benchmark/layout")

    bm = sly.nn.benchmark.ObjectDetectionBenchmark(
        api, g.gt_project_id, output_dir=g.STORAGE_DIR + "/benchmark"
    )
    # bm.run_evaluation(model_session=62933)
    bm.evaluate(g.dt_project_id)
    # bm.upload_eval_results("/model-benchmark/evaluation/test-project/")

    bm.visualize(g.dt_project_id)
    bm.upload_visualizations("/model-benchmark/evaluation/test-project/visualizations")


button = Button("Click to calc")
layout = Container(widgets=[Text("some_text"), button])


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
