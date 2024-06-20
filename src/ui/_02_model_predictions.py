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
    FastTable,
    GridGallery,
    GridGalleryV2,
    IFrame,
    Markdown,
    SelectDataset,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def grid_gallery_model_preds():
    gt_project_id = 38685
    gt_dataset_id = 91896
    pred_project_id = 38684
    pred_dataset_id = 91895
    diff_project_id = 38740
    diff_dataset_id = 92002

    gt_project_meta = sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=gt_project_id))
    pred_project_meta = sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=pred_project_id))
    diff_project_meta = sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=diff_project_id))

    global grid_gallery
    # initialize widgets we will use in UI

    gt_image_info = g.api.image.get_list(dataset_id=gt_dataset_id)[0]

    for image in g.api.image.get_list(dataset_id=pred_dataset_id):
        if image.name == gt_image_info.name:
            pred_image_info = image
            break

    for image in g.api.image.get_list(dataset_id=diff_dataset_id):
        if image.name == gt_image_info.name:
            diff_image_info = image
            break

    images_infos = [gt_image_info, pred_image_info, diff_image_info]
    anns_infos = [g.api.annotation.download(x.id) for x in images_infos]
    project_metas = [gt_project_meta, pred_project_meta, diff_project_meta]

    for idx, (image_info, ann_info, project_meta) in enumerate(
        zip(images_infos, anns_infos, project_metas)
    ):
        image_name = image_info.name
        image_url = image_info.full_storage_url

        # image_ann = sly.Annotation.from_json(data=ann_info, project_meta=project_meta)
        # g.api.annotation.get_info_by_id(image_info.id)

        grid_gallery.append(
            title=image_name,
            image_url=image_url,
            annotation_info=ann_info,
            column_index=idx,
            project_meta=project_meta,
        )


# if g.RECALC_PLOTS:
#     grid_gallery_model_preds()

markdown = Markdown(
    """## Model Predictions

In this section you can visually assess the model performance through examples. This helps users better understand model capabilities and limitations, giving an intuitive grasp of prediction quality in different scenarios.

You can choose one of the sorting method:

- **Auto**: The algorithm is trying to gather a diverse set of images that illustrate the model's performance across various scenarios.
- **Least accurate**: Displays images where the model made more errors.
- **Most accurate**: Displays images where the model made fewer or no errors.
- **Dataset order**: Displays images in the original order of the dataset.
""",
    show_border=False,
)

markdown_table = Markdown(
    """### Prediction Table

The table helps you in finding samples with specific cases of interest. You can sort by parameters such as the number of predictions, or specific a metric, e.g, recall, then click on a row to view this image and predictions.

**Example**: you can sort by **FN** (False Negatives) in descending order to identify samples where the model failed to detect many objects.
""",
    show_border=False,
)
grid_gallery = GridGalleryV2(columns_number=3, enable_zoom=False)
table_model_preds = FastTable(g.m.prediction_table())
# iframe_overview = IFrame("static/01_overview.html", width=620, height=520)


def handle(_grid_gallery, selected_image_name="000000575815.jpg"):
    gt_project_id = 38685
    gt_dataset_id = 91896
    pred_project_id = 38684
    pred_dataset_id = 91895
    diff_project_id = 38740
    diff_dataset_id = 92002

    gt_image_info = g.api.image.get_info_by_name(gt_dataset_id, selected_image_name)
    pred_image_info = g.api.image.get_info_by_name(pred_dataset_id, selected_image_name)
    diff_image_info = g.api.image.get_info_by_name(diff_dataset_id, "000000575815.jpg")

    images_infos = [gt_image_info, pred_image_info, diff_image_info]
    anns_infos = [g.api.annotation.download(x.id) for x in images_infos]
    project_metas = [
        sly.ProjectMeta.from_json(data=g.api.project.get_meta(id=x))
        for x in [gt_project_id, pred_project_id, diff_project_id]
    ]

    for idx, (image_info, ann_info, project_meta) in enumerate(
        zip(images_infos, anns_infos, project_metas)
    ):
        image_name = image_info.name
        image_url = image_info.full_storage_url
        _grid_gallery.append(
            title=image_name,
            image_url=image_url,
            annotation_info=ann_info,
            column_index=idx,
            project_meta=project_meta,
        )


@table_model_preds.row_click
def handle_table_row(clicked_row: sly.app.widgets.FastTable.ClickedRow):
    global grid_gallery
    grid_gallery.clean_up()
    handle(grid_gallery, clicked_row.row[0])
    grid_gallery.update_data()


container = Container(
    widgets=[
        markdown,
        grid_gallery,
        markdown_table,
        table_model_preds,
    ]
)

# Input card with all widgets.
# card = Card(
#     "Model Predictions",
#     "Description",
#     content=Container(
#         widgets=[
#             markdown,
#             grid_gallery,
#             table_model_preds,
#         ]
#     ),
#     # content_top_right=change_dataset_button,
#     collapsable=True,
# )
