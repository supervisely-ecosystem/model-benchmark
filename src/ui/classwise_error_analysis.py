import os
import random
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui import definitions
from src.utils import CVTask, PlotlyHandler


class ClasswiseErrorAnalysis(PlotlyHandler):

    cv_tasks: Tuple[CVTask] = (CVTask.SEGMENTATION.value,)

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        pd.options.mode.chained_assignment = None  # TODO rm later

        df = g.result_df
        df.drop(["mean"], inplace=True)
        df = df[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
        df.sort_values(by="IoU", ascending=False, inplace=True)
        labels = list(df.index)
        color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

        fig = go.Figure()
        for i, column in enumerate(df.columns):
            fig.add_trace(
                go.Bar(
                    name=column,
                    y=df[column],
                    x=labels,
                    marker_color=color_palette[i],
                )
            )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            barmode="stack",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            title={
                "text": "Classwise segmentation error analysis",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
        )
        return fig
