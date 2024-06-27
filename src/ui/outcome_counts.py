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
import supervisely as sly
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


def outcome_counts():
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

TP_hint = "True Positives (TP): These are correctly detected objects. For a prediction to be counted as a true positive, the predicted bounding box must align with a ground truth bounding box with an Intersection over Union (IoU) of 0.5 or more, and the object must be correctly classified"
FP_hint = "False Positives (FP): These are incorrect detections made by the model. They occur when the model predicts a bounding box that either does not overlap sufficiently with any ground truth box (IoU less than 0.5) or incorrectly classifies the object within the bounding box. For example, the model detects a car in the image, but there is no car in the ground truth."
FN_hint = "False Negatives (FN): These are the missed detections. They occur when an actual object in the ground truth is not detected by the model, meaning there is no predicted bounding box with an IoU of 0.5 or more for this object. For example, there is a car in the image, but the model fails to detect it."

markdown = Markdown(
    f"""## Outcome Counts

This chart is used to evaluate the overall model performance by breaking down all predictions into <abbr title="{TP_hint}">True Positives</abbr> (TP), <abbr title="{FP_hint}">False Positives</abbr> (FP), and <abbr title="{FN_hint}">False Negatives</abbr> (FN). This helps to visually assess the type of errors the model often encounters.
""",
    show_border=False,
)
# iframe_outcome_counts = IFrame("static/05_outcome_counts.html", width=620, height=320)
fig = outcome_counts()
plotly_outcome_counts = PlotlyChart(fig)

grid_gallery_v2 = GridGalleryV2(columns_number=5, enable_zoom=False)

pred_project_id = 39104
pred_dataset_id = 92816
images_infos = g.api.image.get_list(dataset_id=pred_dataset_id)[: grid_gallery_v2.columns_number]
anns_infos = [g.api.annotation.download(x.id) for x in images_infos][
    : grid_gallery_v2.columns_number
]
pred_project_meta = sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=pred_project_id))

dialog_container = Container([grid_gallery_v2])
dialog = Dialog(content=dialog_container)

for idx, (image_info, ann_info) in enumerate(zip(images_infos, anns_infos)):
    image_name = image_info.name
    image_url = image_info.full_storage_url

    grid_gallery_v2.append(
        title=image_name,
        image_url=image_url,
        annotation_info=ann_info,
        column_index=idx,
        project_meta=pred_project_meta,
    )


@plotly_outcome_counts.click
def click_handler(datapoints):
    texts = ""
    for datapoint in datapoints:
        # texts += f"\nx: {datapoint.x}, y: {datapoint.y}"  # или другие поля
        label = datapoint.label
        break

    dialog.title = label
    dialog.show()

    print(f"click_handler: {texts}")


container = Container(widgets=[markdown, plotly_outcome_counts])
