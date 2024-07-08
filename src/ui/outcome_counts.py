import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import src.globals as g
import src.utils as u
import supervisely as sly
from src.ui import definitions
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DatasetThumbnail,
    Dialog,
    GridGalleryV2,
    IFrame,
    Markdown,
    PlotlyChart,
    SelectDataset,
    Table,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


class OutcomeCounts(u.PlotlyHandler):

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        # Outcome counts
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[g.m.TP_count],
                y=["Outcome"],
                name="TP",
                orientation="h",
                marker=dict(color="#1fb466"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=[g.m.FN_count],
                y=["Outcome"],
                name="FN",
                orientation="h",
                marker=dict(color="#dd3f3f"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=[g.m.FP_count],
                y=["Outcome"],
                name="FP",
                orientation="h",
                marker=dict(color="#d5a5a5"),
            )
        )
        fig.update_layout(
            barmode="stack",
            width=600,
            height=300,
        )
        fig.update_xaxes(title_text="Count")
        fig.update_yaxes(tickangle=-90)

        return fig


# markdown = Markdown(
#     f"""## Outcome Counts

# This chart is used to evaluate the overall model performance by breaking down all predictions into <abbr title="{definitions.true_positives}">True Positives</abbr> (TP), <abbr title="{definitions.false_positives}">False Positives</abbr> (FP), and <abbr title="{definitions.false_negatives}">False Negatives</abbr> (FN). This helps to visually assess the type of errors the model often encounters.
# """,
#     show_border=False,
# )
# fig = outcome_counts()
# plotly_outcome_counts = PlotlyChart(fig)
# dialog_gallery = GridGalleryV2(columns_number=4, enable_zoom=False)

# dialog_container = Container([dialog_gallery])
# dialog = Dialog(content=dialog_container)


# @plotly_outcome_counts.click
# def click_handler(datapoints):
#     plotly_outcome_counts.loading = True
#     for datapoint in datapoints:
#         # texts += f"\nx: {datapoint.x}, y: {datapoint.y}"  # или другие поля
#         label = datapoint.label
#         break

#     dialog_gallery.clean_up()

#     image_ids = list(set([x["dt_img_id"] for x in g.click_data.oucome_counts[label]]))
#     image_infos = [x for x in g.dt_image_infos if x.id in image_ids][:20]
#     anns_infos = [x for x in g.dt_anns_infos if x.image_id in image_ids][:20]

#     for idx, (image_info, ann_info) in enumerate(zip(image_infos, anns_infos)):
#         image_name = image_info.name
#         image_url = image_info.full_storage_url

#         dialog_gallery.append(
#             title=image_name,
#             image_url=image_url,
#             annotation_info=ann_info,
#             column_index=idx % dialog_gallery.columns_number,
#             project_meta=g.dt_project_meta,
#         )
#     dialog.title = label
#     plotly_outcome_counts.loading = False
#     dialog.show()


# container = Container(
#     widgets=[
#         markdown,
#         plotly_outcome_counts,
#     ]
# )
