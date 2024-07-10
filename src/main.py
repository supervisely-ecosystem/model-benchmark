import json
from typing import *

import plotly.graph_objects as go
from tqdm import tqdm

import src.globals as g
import supervisely as sly
from supervisely._utils import camel_to_snake
from supervisely.app.widgets import Button, Card, Container, Sidebar, Text


def main_func():

    cocoGt_path = "APP_DATA/data/cocoGt.json"  # cocoGt_remap.json"
    cocoDt_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/cocoDt.json"
    eval_data_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/eval_data.pkl"

    with sly.nn.MetricsLoader(cocoGt_path, cocoDt_path, eval_data_path) as loader:
        loader.upload_to(g.TEAM_ID, "/model-benchmark/layout")


button = Button("Click to calc")
layout = Container(widgets=[Text("some_text"), button])


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
