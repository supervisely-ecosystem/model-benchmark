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
    Collapse,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    NotificationBox,
    OneOf,
    SelectDataset,
    Switch,
    Table,
    Text,
)

markdown_inference_speed_1 = Markdown(
    """
## Inference speed

We evaluate the inference speed in two scenarios: real-time inference (batch size is 1), and batch processing. We also run the model in optimized runtime environments, such as ONNX Runtime and Tensor RT, using consistent hardware. This approach provides a fair comparison of model efficiency and speed. To assess the inference speed we run the model forward 100 times and average it.
""",
    show_border=False,
)
collapsables = Collapse(
    [
        Collapse.Item(
            "Methodology",
            "Methodology",
            Container(
                [
                    Markdown(
                        """
Setting 1: **Real-time processing**

We measure the time spent processing each image individually by setting batch size to 1. This simulates real-time data processing conditions, such as those encountered in video streams, ensuring the model performs effectively in scenarios where data is processed frame by frame.

Setting 2: **Parallel processing**

To evaluate the model's efficiency in parallel processing, we measure the processing speed with batch size of 8 and 16. This helps us understand how well the model scales when processing multiple images simultaneously, which is crucial for applications requiring high throughput.

Setting 3: **Optimized runtime**

We run the model in various runtime environments, including **ONNX Runtime** and **TensorRT**. This is important because python code can be suboptimal. These runtimes often provide significant performance improvements.
""",
                        show_border=False,
                    ),
                ]
            ),
        )
    ]
)
markdown_inference_speed_2 = Markdown(
    """
#### Consistent hardware for fair comparison

To ensure a fair comparison, we use a single hardware setup, specifically an NVIDIA RTX 3060 GPU.

#### Inference details

We divide the inference process into three stages: **preprocess, inference,** and **postprocess** to provide insights into where optimization efforts should be focused. Additionally, it gives us another verification level to ensure that time is measured correctly for each model.

#### Preprocess 

The stage where images are prepared for input into the model. This includes image reading, resizing, and any necessary transformations.

#### Inference

The main computation phase where the _forward_ pass of the model is running. **Note:** we include not only the forward pass, but also modules like NMS (Non-Maximum Suppression), decoding module, and everything that is done to get a **meaningful** prediction.

#### Postprocess

This stage includes tasks such as resizing output masks, aligning predictions with the input image, converting bounding boxes into a specific format or filtering out low-confidence detections.
""",
    show_border=False,
)
container = Container(
    widgets=[
        markdown_inference_speed_1,
        collapsables,
        markdown_inference_speed_2,
    ]
)
