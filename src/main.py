import json
from typing import Optional

import plotly.graph_objects as go
from tqdm import tqdm

import src.globals as g

# import src.utils as u
import supervisely as sly
from src.ui.calibration_score import (
    ConfidenceDistribution,
    ConfidenceScore,
    F1ScoreAtDifferentIOU,
    ReliabilityDiagram,
)
from src.ui.classwise_error_analysis import ClasswiseErrorAnalysis
from src.ui.confusion_matrix import ConfusionMatrix
from src.ui.frequently_confused import FrequentlyConfused
from src.ui.iou_distribution import IOUDistribution
from src.ui.outcome_counts import OutcomeCounts
from src.ui.overall_errors_analysis import OverallErrorAnalysis
from src.ui.overview import Overview
from src.ui.perclass import PerClassAvgPrecision, PerClassOutcomeCounts
from src.ui.pr_curve import PRCurve, PRCurveByClass
from src.ui.pr_metrics import Precision, Recall, RecallVsPrecision
from src.utils import CVTask, PlotlyHandler
from supervisely._utils import camel_to_snake
from supervisely.app.widgets import Button, Card, Container, Sidebar, Text
from supervisely.nn.benchmark.metrics_loader import MetricsLoader

# import src.ui.detailed_metrics as detailed_metrics
# import src.ui.model_predictions as model_preds
# import src.ui.what_is_section as what_is
# import src.ui.inference_speed as inference_speed


def main_func():

    cocoGt_path = "APP_DATA/data/cocoGt.json"  # cocoGt_remap.json"
    cocoDt_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/cocoDt.json"
    eval_data_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/eval_data.pkl"

    with MetricsLoader(cocoGt_path, cocoDt_path, eval_data_path) as loader:
        loader.upload_to(g.TEAM_ID, "/model-benchmark/layout")


button = Button("Click to calc")
layout = Container(widgets=[Text("some_text"), button])


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
