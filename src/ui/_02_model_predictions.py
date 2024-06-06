import os
import random
from collections import defaultdict

# %%
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
    FastTable,
    IFrame,
    SelectDataset,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider

# def model_preds():
#     df: pd.DataFrame = g.m.prediction_table()
#     df.to_json()

#     fig.write_html(g.STATIC_DIR + "/01_overview.html")


# if g.RECALC_PLOTS:
#     model_preds()
txt = Text("text")
table_model_preds = FastTable(g.m.prediction_table())
# iframe_overview = IFrame("static/01_overview.html", width=620, height=520)


# Input card with all widgets.
card = Card(
    "Model Predictions",
    "Description",
    content=Container(
        widgets=[
            txt,
            table_model_preds,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
