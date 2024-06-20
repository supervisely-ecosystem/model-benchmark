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
    IFrame,
    Markdown,
    SelectDataset,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider

# def prepare():
#     global confusion_matrix
#     confusion_matrix = g.m.confusion_matrix()


def _confusion_matrix():
    confusion_matrix = g.m.confusion_matrix()
    # Confusion Matrix
    # TODO: Green-red
    cat_names = g.m.cat_names
    none_name = "(None)"

    confusion_matrix_df = pd.DataFrame(
        np.log(confusion_matrix), index=cat_names + [none_name], columns=cat_names + [none_name]
    )
    fig = px.imshow(
        confusion_matrix_df,
        labels=dict(x="Ground Truth", y="Predicted", color="Count"),
        # title="Confusion Matrix (log-scale)",
        width=1000,
        height=1000,
    )

    # Hover text
    fig.update_traces(
        customdata=confusion_matrix,
        hovertemplate="Count: %{customdata}<br>Predicted: %{y}<br>Ground Truth: %{x}",
    )

    # Text on cells
    if len(cat_names) <= 20:
        fig.update_traces(text=confusion_matrix, texttemplate="%{text}")

    # fig.show()
    return fig


def confusion_matrix_mini():
    confusion_matrix = g.m.confusion_matrix()
    class_name = "car"
    class_idx = g.m.cat_names.index(class_name)

    y_nz = np.nonzero(confusion_matrix[class_idx, :-1])[0]
    x_nz = np.nonzero(confusion_matrix[:-1, class_idx])[0]
    idxs = np.union1d(y_nz, x_nz)
    if class_idx not in idxs:
        idxs = np.concatenate([idxs, [class_idx]])
    idxs = np.sort(idxs)

    # get confusion matrix for the selected classes
    confusion_matrix_mini = confusion_matrix[idxs][:, idxs].copy()
    self_idx = idxs == class_idx
    v = confusion_matrix_mini[self_idx, self_idx]
    confusion_matrix_mini[np.diag_indices_from(confusion_matrix_mini)] *= 0
    confusion_matrix_mini[self_idx, self_idx] = v

    cat_names_cls = [g.m.cat_names[i] for i in idxs]
    confusion_matrix_df_mini = pd.DataFrame(
        np.log(confusion_matrix_mini), index=cat_names_cls, columns=cat_names_cls
    )
    fig = px.imshow(
        confusion_matrix_df_mini,
        labels=dict(x="Ground Truth", y="Predicted", color="Count"),
        title=f"Confusion Matrix: {class_name} (log-scale)",
    )
    # width=1000, height=1000)

    # Hover text
    fig.update_traces(
        customdata=confusion_matrix_mini,
        hovertemplate="Count: %{customdata}<br>Predicted: %{y}<br>Ground Truth: %{x}<extra></extra>",
    )

    # Text on cells
    if len(idxs) <= 20:
        fig.update_traces(text=confusion_matrix_mini, texttemplate="%{text}")

    # fig.show()
    return fig


markdown = Markdown(
    """
# Confusion Matrix

# Classification Accuracy

This section investigates cases where the model correctly localizes a bounding box, but predicts a wrong class label. For example, the model may often confuse a motorbike with a bicycle. Confusion matrix is designed to discover these mistakes.

**Classification Accuracy**: 0.96

**52 of 54 total predictions were misclassified by the model.**

## Confusion Matrix

Confusion matrix helps to find confusions between different classes of objects made by the model. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class. The diagonal elements represent the number of correct predictions for each class (True Positives), and the off-diagonal elements show the misclassifications.

""",
    show_border=False,
)

iframe_confusion_matrix = IFrame("static/08_1_confusion_matrix.html", width=1020, height=1020)
iframe_confusion_matrix_mini = IFrame("static/08_2_confusion_matrix.html", width=620, height=520)

container = Container(
    widgets=[
        markdown,
        iframe_confusion_matrix,
        iframe_confusion_matrix_mini,
    ]
)

# Input card with all widgets.
card = Card(
    "Confusion Matrix",
    "Description",
    content=Container(
        widgets=[
            markdown,
            iframe_confusion_matrix,
            iframe_confusion_matrix_mini,
        ]
    ),
    # content_top_right=change_dataset_button,
    collapsable=True,
)
