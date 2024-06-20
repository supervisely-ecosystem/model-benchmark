import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import supervisely as sly
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()


SLY_APP_DATA_DIR = sly.app.get_data_dir()
STATIC_DIR = os.path.join(SLY_APP_DATA_DIR, "static")
sly.fs.mkdir(STATIC_DIR)


cocoGt_path = "APP_DATA/data/cocoGt_remap.json"
cocoDt_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/cocoDt.json"
eval_data_path = "APP_DATA/data/COCO 2017 val (DINO-L, conf-0.05)_001 (#2)/eval_data.pkl"

# Remove COCO read logs
with HiddenCocoPrints():
    cocoGt = COCO(cocoGt_path)

cocoDt = cocoGt.loadRes(cocoDt_path)
# cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
m_full: MetricProvider = None
m: MetricProvider = None
score_profile = None
df_down = None
import pickle

with open(eval_data_path, "rb") as f:
    eval_data = pickle.load(f)
from importlib import reload

reload(metric_provider)
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
