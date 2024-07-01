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
from src.ui import definitions
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


def explorer(grid_gallery: GridGalleryV2, selected_image_name="000000575815.jpg"):

    gt_image_infos = g.api.image.get_list(dataset_id=g.gt_dataset_id)[:5]
    pred_image_infos = g.api.image.get_list(dataset_id=g.dt_dataset_id)[:5]
    diff_image_infos = g.api.image.get_list(dataset_id=g.diff_dataset_id)[:5]

    # gt_image_infos = g.api.image.get_list(dataset_id=g.gt_dataset_id)[:5]
    # pred_image_infos = g.api.image.get_list(dataset_id=g.gt_dataset_id)[:5]
    # diff_image_infos = g.api.image.get_list(dataset_id=g.gt_dataset_id)[:5]

    project_metas = [
        sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=x))
        for x in [g.gt_project_id, g.dt_project_id, g.diff_project_id]
    ]
    # project_metas = [
    #     sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=x))
    #     for x in [g.gt_project_id, g.gt_project_id, g.gt_project_id]
    # ]
    for gt_image, pred_image, diff_image in zip(gt_image_infos, pred_image_infos, diff_image_infos):
        image_infos = [gt_image, pred_image, diff_image]
        ann_infos = [g.api.annotation.download(x.id) for x in image_infos]

        for idx, (image_info, ann_info, project_meta) in enumerate(
            zip(image_infos, ann_infos, project_metas)
        ):
            image_name = image_info.name
            image_url = image_info.full_storage_url
            is_ignore = True if idx == 0 else False
            grid_gallery.append(
                title=image_name,
                image_url=image_url,
                annotation_info=ann_info,
                column_index=idx,
                project_meta=project_meta,
                ignore_tags_filtering=is_ignore,
            )


checkpoint_name = "YOLOv8-L (COCO 2017 val)"

markdown_overview = Markdown(
    f"""# {checkpoint_name}

## Overview

- **Model**: [YOLOv8-L]()
- **Year**: 2023
- **Authors**: ultralytics
- **Task type**: object detection
- **Training dataset (?)**: COCO 2017 train
- **Model classes (?)**: (80): a, b, c, … (collapse)
- **Model weights (?)**: [/path/to/yolov8l.pt]()
- **License (?)**: AGPL-3.0
- [GitHub](https://github.com/ultralytics/ultralytics)
""",
    show_border=False,
)

markdown_key_metrics = Markdown(
    f"""## Key Metrics

Here, we comprehensively assess the model's performance by presenting a broad set of metrics, including mAP (mean Average Precision), Precision, Recall, IoU (Intersection over Union), Classification Accuracy, Calibration Score, and Inference Speed.

- **Mean Average Precision (mAP)**: A comprehensive metric of detection performance. mAP calculates the <abbr title="{definitions.average_precision}">average precision</abbr> across all classes at different levels of <abbr title="{definitions.iou_threshold}">IoU thresholds</abbr> and precision-recall trade-offs. In other words, it evaluates the performance of a model by considering its ability to detect and localize objects accurately across multiple IoU thresholds and object categories.
- **Precision**: Precision indicates how often the model's predictions are actually correct when it predicts an object. This calculates the ratio of correct detections to the total number of detections made by the model.
- **Recall**: Recall measures the model's ability to find all relevant objects in a dataset. This calculates the ratio of correct detections to the total number of instances in a dataset.
- **Intersection over Union (IoU)**: IoU measures how closely predicted bounding boxes match the actual (ground truth) bounding boxes. It is calculated as the area of overlap between the predicted bounding box and the ground truth bounding box, divided by the area of union of these bounding boxes.
- **Classification Accuracy**: We separately measure the model's capability to correctly classify objects. It’s calculated as a proportion of correctly classified objects among all matched detections. A predicted bounding box is considered matched if it overlaps a ground true bounding box with IoU equal or higher than 0.5.
- **Calibration Score**: This score represents the consistency of predicted probabilities (or <abbr title="{definitions.confidence_score}">confidence scores</abbr>) made by the model, evaluating how well the predicted probabilities align with actual outcomes. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
- **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1. The inference speed is important in applications, where real-time object detection is required. Additionally, slower models pour more GPU resources, so their inference cost is higher.
""",
    show_border=False,
)
markdown_explorer = Markdown(
    """## Explore Predictions
""",
    show_border=False,
)
iframe_overview = IFrame("static/01_overview.html", width=620, height=520)

gridgallery_explorer = GridGalleryV2(
    columns_number=3,
    enable_zoom=False,
    default_tag_filters=[{"confidence": [0.6, 1]}, {"outcome": "TP"}],
)
explorer(gridgallery_explorer)

container = Container(
    widgets=[
        markdown_overview,
        markdown_key_metrics,
        iframe_overview,
        markdown_explorer,
        gridgallery_explorer,
    ]
)
