import json
from typing import Optional

import plotly.graph_objects as go
from tqdm import tqdm

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui.calibration_score import (
    ConfidenceDistribution,
    ConfidenceScore,
    F1ScoreAtDifferentIOU,
    ReliabilityDiagram,
)
from src.ui.confusion_matrix import ConfusionMatrix
from src.ui.frequently_confused import FrequentlyConfused

# import src.ui.inference_speed as inference_speed
from src.ui.iou_distribution import IOU_Distribution
from src.ui.outcome_counts import OutcomeCounts
from src.ui.overview import Overview
from src.ui.perclass import PerClassAvgPrecision, PerClassOutcomeCounts
from src.ui.pr_curve import PRCurve, PRCurveByClass
from src.ui.pr_metrics import Precision, Recall, RecallVsPrecision
from supervisely._utils import camel_to_snake
from supervisely.app.widgets import Button, Card, Container, Sidebar, Text

# import src.ui.detailed_metrics as detailed_metrics
# import src.ui.model_predictions as model_preds
# import src.ui.what_is_section as what_is


_PLOTLY_CHARTS = (
    Overview,
    OutcomeCounts,
    Recall,
    Precision,
    RecallVsPrecision,
    PRCurve,
    PRCurveByClass,
    ConfusionMatrix,
    FrequentlyConfused,
    IOU_Distribution,
    ReliabilityDiagram,
    ConfidenceScore,
    ConfidenceDistribution,
    F1ScoreAtDifferentIOU,
    PerClassAvgPrecision,
    PerClassOutcomeCounts,
)


def main_func():

    def write_fig(
        plotly_chart: u.PlotlyHandler, fig: go.Figure, fig_idx: Optional[int] = None
    ) -> None:
        json_fig = fig.to_json()

        chart_name = camel_to_snake(plotly_chart.__name__)
        basename = f"{chart_name}.json"
        local_path = f"{g.TO_TEAMFILES_DIR}/{chart_name}.json"

        if fig_idx is not None:
            fig_idx = "{:02d}".format(fig_idx)
            basename = f"{chart_name}_{fig_idx}.json"
            local_path = f"{g.TO_TEAMFILES_DIR}/{basename}"

        with open(local_path, "w", encoding="utf-8") as f:
            f.write(json_fig)

        sly.logger.info("Saved: %r", basename)

    for plotly_chart in _PLOTLY_CHARTS:
        fig = plotly_chart.get_figure()
        if fig is not None:
            write_fig(plotly_chart, fig)
        figs = plotly_chart.get_switchable_figures()
        if figs is not None:
            for idx, fig in enumerate(figs, start=1):
                write_fig(plotly_chart, fig, fig_idx=idx)

    table_preds = g.m.prediction_table()
    basename = "prediction_table.json"
    local_path = f"{g.TO_TEAMFILES_DIR}/{basename}"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(table_preds.to_json())
    sly.logger.info("Saved: %r", basename)

    with tqdm(
        desc="Uploading .json to teamfiles",
        total=sly.fs.get_directory_size(g.TO_TEAMFILES_DIR),
        unit="B",
        unit_scale=True,
    ) as pbar:
        g.api.file.upload_directory(
            g.TEAM_ID,
            g.TO_TEAMFILES_DIR,
            g.TF_RESULT_DIR,
            replace_if_conflict=True,
            progress_size_cb=pbar,
        )

    sly.logger.info("Done.")


button = Button("Click to calc")
layout = Container(widgets=[Text("some_text"), button])


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
