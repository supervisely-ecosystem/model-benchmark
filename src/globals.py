import os
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from supervisely.convert.image.coco.coco_helper import HiddenCocoPrints

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
m_full = None
m = None
score_profile = None
df_down = None
import pickle

with open(eval_data_path, "rb") as f:
    eval_data = pickle.load(f)

RECALC_PLOTS = True
