from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


class IdMapper:
    def __init__(self, coco_dataset: dict):
        self.map_img = {x["id"]: x["sly_id"] for x in coco_dataset["images"]}
        self.map_obj = {x["id"]: x["sly_id"] for x in coco_dataset["annotations"]}


class PlotlyHandler:
    @classmethod
    def get_figure(cls) -> go.Figure:
        pass

    @classmethod
    def get_switchable_figures(cls) -> Tuple[go.Figure]:
        pass


def silent_np_log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)
