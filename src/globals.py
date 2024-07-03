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
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider

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

cocoGt_path = "APP_DATA/data/cocoGt.json"  # cocoGt_remap.json"
cocoDt_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/cocoDt.json"
eval_data_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/eval_data.pkl"

with open(cocoGt_path, "r") as f:
    cocoGt_dataset = json.load(f)
with open(cocoDt_path, "r") as f:
    cocoDt_dataset = json.load(f)

# Remove COCO read logs
with HiddenCocoPrints():
    cocoGt = COCO()
    cocoGt.dataset = cocoGt_dataset
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt_dataset["annotations"])

with open(eval_data_path, "rb") as f:
    eval_data = pickle.load(f)

m_full = metric_provider.MetricProvider(
    eval_data["matches"],
    eval_data["coco_metrics"],
    eval_data["params"],
    cocoGt,
    cocoDt,
)
score_profile = m_full.confidence_score_profile()
f1_optimal_conf, best_f1 = m_full.get_f1_optimal_conf()
print(f"F1-Optimal confidence: {f1_optimal_conf:.4f} with f1: {best_f1:.4f}")

matches_thresholded = metric_provider.filter_by_conf(eval_data["matches"], f1_optimal_conf)
m = metric_provider.MetricProvider(
    matches_thresholded, eval_data["coco_metrics"], eval_data["params"], cocoGt, cocoDt
)
f1_optimal_conf, best_f1 = m_full.get_f1_optimal_conf()
df_score_profile = pd.DataFrame(score_profile)
df_score_profile.columns = ["scores", "Precision", "Recall", "F1"]

# downsample
if len(df_score_profile) > 5000:
    dfsp_down = df_score_profile.iloc[:: len(df_score_profile) // 1000]
else:
    dfsp_down = df_score_profile

# Click data
gt_id_mapper = IdMapper(cocoGt_dataset)
dt_id_mapper = IdMapper(cocoDt_dataset)

click_data = ClickData(m, gt_id_mapper, dt_id_mapper)
click_data.create_data()


gt_project_id = 39099
gt_dataset_id = 92810
dt_project_id = 39141
dt_dataset_id = 92872
diff_project_id = 39249
diff_dataset_id = 93099


_workspace_id = 1076
_team_id = 440


dt_image_infos = api.image.get_list(dt_dataset_id)
dt_actual_ids = [x.id for x in dt_image_infos]  # TODO remove later
dt_anns_infos = api.annotation.download_batch(dt_dataset_id, [x.id for x in dt_image_infos])
dt_project_meta = sly.ProjectMeta.from_json(data=api.project.get_meta(id=dt_project_id))
