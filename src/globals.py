import json
import os
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import supervisely as sly
from src.click_data import ClickData
from src.utils import IdMapper
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.nn.benchmark.evaluation.object_detection import metric_provider
from supervisely.nn.benchmark.evaluation.object_detection.metric_provider import (
    METRIC_NAMES,
    MetricProvider,
)

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()

STORAGE_DIR = sly.app.get_data_dir()
STATIC_DIR = os.path.join(STORAGE_DIR, "static")
sly.fs.mkdir(STATIC_DIR)
TF_RESULT_DIR = "/model-benchmark/layout"
TO_TEAMFILES_DIR = f"{STORAGE_DIR}/to_teamfiles"
sly.fs.mkdir(TO_TEAMFILES_DIR, remove_content_if_exists=True)

deployed_nn_tags = ["deployed_nn"]

workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
team_id = sly.env.team_id()
# gt_project_id = 39099
# gt_dataset_id = 92810
# dt_project_id = 39141
# dt_dataset_id = 92872
# diff_project_id = 39249
# diff_dataset_id = 93099
