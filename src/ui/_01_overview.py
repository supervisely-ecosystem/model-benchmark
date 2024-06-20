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
    GridGallery,
    GridGalleryV2,
    IFrame,
    Markdown,
    SelectDataset,
    Text,
)


def overall():
    # Overall Metrics
    base_metrics = g.m.base_metrics()
    r = list(base_metrics.values())
    theta = [g.metric_provider.METRIC_NAMES[k] for k in base_metrics.keys()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r + [r[0]],
            theta=theta + [theta[0]],
            fill="toself",
            name="Overall Metrics",
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0.0, 1.0]), angularaxis=dict(rotation=90, direction="clockwise")
        ),
        # title="Overall Metrics",
        width=600,
        height=500,
    )
    return fig


def explorer(grid_gallery, selected_image_name="000000575815.jpg"):
    gt_project_id = 38685
    gt_dataset_id = 91896
    pred_project_id = 38684
    pred_dataset_id = 91895
    diff_project_id = 38914
    diff_dataset_id = 92290

    gt_image_infos = g.api.image.get_list(dataset_id=gt_dataset_id)[:5]
    pred_image_infos = g.api.image.get_list(dataset_id=pred_dataset_id)[:5]
    diff_image_infos = g.api.image.get_list(dataset_id=diff_dataset_id)[:5]

    # gt_image_info = g.api.image.get_info_by_name(gt_dataset_id, selected_image_name)
    # pred_image_info = g.api.image.get_info_by_name(pred_dataset_id, selected_image_name)
    # diff_image_info = g.api.image.get_info_by_name(diff_dataset_id, selected_image_name)

    project_metas = [
        sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=x))
        for x in [gt_project_id, pred_project_id, diff_project_id]
    ]

    for gt_image, pred_image, diff_image in zip(gt_image_infos, pred_image_infos, diff_image_infos):
        image_infos = [gt_image, pred_image, diff_image]
        ann_infos = [g.api.annotation.download(x.id) for x in image_infos]

        for idx, (image_info, ann_info, project_meta) in enumerate(
            zip(image_infos, ann_infos, project_metas)
        ):
            image_name = image_info.name
            image_url = image_info.full_storage_url
            grid_gallery.append(
                title=image_name,
                image_url=image_url,
                annotation_info=ann_info,
                column_index=idx,
                project_meta=project_meta,
            )


# if g.RECALC_PLOTS:
#     overall()

markdown = Markdown(
    """
# Overall Metrics    

Overview of the model performance across a set of key metrics. Greater values are better. \n\n

При наведении на (?) вопросик на бар чартах:\n
* **Mean Average Precision (mAP)**: A measure of the precision-recall trade-off across different thresholds, reflecting the model's overall detection performance.\n
* **Precision**: The ratio of true positive detections to the total number of positive detections made by the model, indicating its accuracy in identifying objects correctly.\n
* **Recall**: The ratio of true positive detections to the total number of actual objects, measuring the model's ability to find all relevant objects.\n
* **Intersection over Union (IoU)**: The overlap between the predicted bounding boxes and the ground truth, providing insight into the spatial accuracy of detections.\n
* **Classification Accuracy**: The proportion of correctly classified objects among all detected objects, highlighting the model's capability in correctly labeling objects.\n
* **Calibration Score**: A metric evaluating how well the predicted probabilities align with the actual outcomes, assessing the confidence calibration of the model. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.\n
* **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1 on the full COCO dataset on RTX3060 GPU.\n

""",
    show_border=False,
)
iframe_overview = IFrame("static/01_overview.html", width=620, height=520)

explorer_grid = GridGalleryV2(columns_number=3, enable_zoom=False)
explorer(explorer_grid)

container = Container(
    widgets=[
        markdown,
        iframe_overview,
        explorer_grid,
    ]
)

# Input card with all widgets.
card = Card(
    "Overall Metrics",
    "Description",
    content=Container(
        widgets=[
            markdown,
            iframe_overview,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
