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
    FastTable,
    GridGalleryV2,
    IFrame,
    Markdown,
    PlotlyChart,
    SelectDataset,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


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


# def confusion_matrix_mini():
#     confusion_matrix = g.m.confusion_matrix()
#     class_name = "car"
#     class_idx = g.m.cat_names.index(class_name)

#     y_nz = np.nonzero(confusion_matrix[class_idx, :-1])[0]
#     x_nz = np.nonzero(confusion_matrix[:-1, class_idx])[0]
#     idxs = np.union1d(y_nz, x_nz)
#     if class_idx not in idxs:
#         idxs = np.concatenate([idxs, [class_idx]])
#     idxs = np.sort(idxs)

#     # get confusion matrix for the selected classes
#     confusion_matrix_mini = confusion_matrix[idxs][:, idxs].copy()
#     self_idx = idxs == class_idx
#     v = confusion_matrix_mini[self_idx, self_idx]
#     confusion_matrix_mini[np.diag_indices_from(confusion_matrix_mini)] *= 0
#     confusion_matrix_mini[self_idx, self_idx] = v

#     cat_names_cls = [g.m.cat_names[i] for i in idxs]
#     confusion_matrix_df_mini = pd.DataFrame(
#         np.log(confusion_matrix_mini), index=cat_names_cls, columns=cat_names_cls
#     )
#     fig = px.imshow(
#         confusion_matrix_df_mini,
#         labels=dict(x="Ground Truth", y="Predicted", color="Count"),
#         title=f"Confusion Matrix: {class_name} (log-scale)",
#     )
#     # width=1000, height=1000)

#     # Hover text
#     fig.update_traces(
#         customdata=confusion_matrix_mini,
#         hovertemplate="Count: %{customdata}<br>Predicted: %{y}<br>Ground Truth: %{x}<extra></extra>",
#     )

#     # Text on cells
#     if len(idxs) <= 20:
#         fig.update_traces(text=confusion_matrix_mini, texttemplate="%{text}")

#     # fig.show()
#     return fig


# markdown_confusion_matrix = Markdown(
#     """
# ## Confusion Matrix

# Confusion matrix helps to find the number of confusions between different classes made by the model.
# Each row of the matrix represents the instances in a ground truth class, while each column represents the instances in a predicted class.
# The diagonal elements represent the number of correct predictions for each class (True Positives), and the off-diagonal elements show misclassifications.
# """,
#     show_border=False,
# )


# container = Container(
#     widgets=[
#         markdown_confusion_matrix,
#         # plotly_confusion_matrix,
#         # iframe_confusion_matrix,
#         # iframe_confusion_matrix_mini,
#     ]
# )
