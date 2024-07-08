from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from supervisely.collection.str_enum import StrEnum


class IdMapper:
    def __init__(self, coco_dataset: dict):
        self.map_img = {x["id"]: x["sly_id"] for x in coco_dataset["images"]}
        self.map_obj = {x["id"]: x["sly_id"] for x in coco_dataset["annotations"]}


class CVTask(str, StrEnum):

    OBJECT_DETECTION: str = "object_detection"
    SEGMENTATION: str = "segmentation"


class PlotlyHandler:

    cv_tasks: Tuple[CVTask] = tuple(CVTask.values())

    @classmethod
    def get_figure(cls) -> Optional[go.Figure]:
        pass

    @classmethod
    def get_switchable_figures(cls) -> Optional[Tuple[go.Figure]]:
        pass


def silent_np_log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)
