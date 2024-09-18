import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supervisely as sly
from supervisely.nn.benchmark.semantic_segmentation_benchmark import (
    SemanticSegmentationBenchmark,
)
from dotenv import load_dotenv


load_dotenv("supervisely.env")
api = sly.Api()

gt_project_id = 39599
result_dir = "my_benchmark"
nn_session_id = 63173

benchmark = SemanticSegmentationBenchmark(
    api=api,
    gt_project_id=gt_project_id,
    output_dir=result_dir,
)
benchmark.run_evaluation(model_session=nn_session_id)
