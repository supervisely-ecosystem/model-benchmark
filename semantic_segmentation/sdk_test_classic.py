import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supervisely as sly
from supervisely.nn.benchmark.evaluation import SemanticSegmentationEvaluator
from dotenv import load_dotenv


load_dotenv("supervisely.env")
api = sly.Api()

gt_project_id = 39702
# pred_project_id = 39704  # model A
pred_project_id = 42608  # model B
result_dir = "output_2"

evaluator = SemanticSegmentationEvaluator(
    api=api,
    gt_project_id=gt_project_id,
    pred_project_id=pred_project_id,
    result_dir=result_dir,
)
evaluator.evaluate()
