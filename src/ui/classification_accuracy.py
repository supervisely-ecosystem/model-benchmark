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
    FastTable,
    IFrame,
    Markdown,
    NotificationBox,
    SelectDataset,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider

markdown_classification_accuracy = Markdown(
    """
## Classification Accuracy

This section investigates cases where the model correctly localizes a bounding box, but predicts a wrong class label. For example, the model might confuse a motorbike with a bicycle. In this case, the model correctly identified that the object is present on the image, but assigned a wrong label to it.

To quantify it, we calculate **classification accuracy**. This is a portion of correctly classified objects to the total number of correctly localized  <abbr title="The object is localized correctly if the IoU between a prediction and a ground truth box is more than 0.5">objects</abbr>. In other words, if the model correctly found that an object is present on the image, how often it assigns a correct label to it?
""",
    show_border=False,
)

base_metrics = g.m.base_metrics()
classification_accuracy = base_metrics["classification_accuracy"]

notibox_classification_accuracy = NotificationBox(
    f"Classification Accuracy: {classification_accuracy:.2f}",
    f"The model correctly classified <b>{g.m.TP_count}</b> predictions of <b>{(g.m.TP_count+len(g.m.confused_matches))}</b> total predictions, that are matched to the ground truth.",
)

container = Container(
    widgets=[
        markdown_classification_accuracy,
        notibox_classification_accuracy,
    ]
)
