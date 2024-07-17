"""Script for calculating performance metrics based on ground truth and predicted annotations"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beyond_iou.evaluator import Evaluator
from  beyond_iou.loader import build_segmentation_loader, reduce_zero_label
import scipy.ndimage


gt_dir, pred_dir = "./mmseg_data/gt", "./mmseg_data/pred"
boundary_width = 0.01
boundary_iou_d = 0.02
reduce_zero_label = False
num_workers = 4
output_dir = "./output"

if boundary_width % 1 == 0:
    boundary_width = int(boundary_width)

evaluator = Evaluator(
    class_names=[
        "agriculture",
        "background",
        "barren",
        "building",
        "forest",
        "road",
        "water",
    ],
    boundary_width=boundary_width,
    boundary_implementation="exact",
    boundary_iou_d=boundary_iou_d,
)
loader = build_segmentation_loader(
    pred_dir=pred_dir,
    gt_dir=gt_dir,
    gt_label_map=reduce_zero_label if reduce_zero_label else None,
    pred_label_map=reduce_zero_label if reduce_zero_label else None,
    num_workers=num_workers,
)
result = evaluator.evaluate(loader)
result.make_table(path="output/table.md")
