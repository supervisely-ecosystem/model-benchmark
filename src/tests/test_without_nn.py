import os
import random
import time
from datetime import datetime

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api()

gt_project_id = sly.env.project_id()
dt_project_id = 419
# gt_dataset_ids = [2526]

dt_info = api.project.get_info_by_id(dt_project_id)
api.project.clone(dt_project_id, dt_info.workspace_id, dt_info.name)
time.sleep(3)
dt_project_id = api.project.get_list(dt_info.workspace_id)[-1].id


iou_threshold_per_class = {}
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(gt_project_id))
available_iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
for obj_cls in project_meta.obj_classes:
    if obj_cls.geometry_type == sly.Rectangle:
        continue
    # just for example, you can set different thresholds for different classes
    iou_threshold_per_class[obj_cls.name] = random.choice(available_iou_thresholds)


evaluation_params = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.65,
    "max_detections": 500,
    # "iou_threshold_per_class": iou_threshold_per_class,
    # "average_across_iou_thresholds": False,
}

# 1. Initialize benchmark
bench = sly.nn.benchmark.ObjectDetectionBenchmark(
    api, gt_project_id, evaluation_params=evaluation_params
)
bench.api.retry_count = 1  # set retry count to 1 to avoid long waiting while testing

# 2. Run evaluation
# This will run inference with the model and calculate metrics.
# Evaluation results will be saved in the "./benchmark" directory locally.
# model_session = "http://localhost:8000"
# model_session = 69289
# bench.run_evaluation(model_session=model_session)

bench.evaluate(dt_project_id)

# 3. Generate charts and dashboards
# This will generate visualization files and save them locally.
bench.visualize()

# 4. Upload to Supervisely Team Files
# To open the generated visualizations in the web interface, you need to upload them to Team Files.
bench.output_dir
bench.get_layout_results_dir()

remote_dir = f"/model-benchmark/test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
bench.upload_eval_results(remote_dir + "/evaluation/")
bench.upload_visualizations(remote_dir + "/visualizations/")
