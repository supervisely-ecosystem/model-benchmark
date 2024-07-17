"""Script for drawing annotations on images from preview set"""

import supervisely as sly
from dotenv import load_dotenv
import os
import json
import shutil
import numpy as np


load_dotenv("supervisely.env")
api = sly.Api()

with open("output/preview_set.json", "r") as file:
    preview_set = json.load(file)

preview_img_names = (
    preview_set["highest_iou"] + preview_set["lowest_iou"] + preview_set["diverse"]
)

output_dir = "output/preview_set"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

gt_project_id = 39106
gt_project_dir, pred_project_dir = "./sly_data/gt", "./sly_data/pred"
img_dir, ann_dir = "img", "ann"

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(gt_project_id))
dir_elements = os.listdir(gt_project_dir)
for element in dir_elements:
    if not os.path.isdir(os.path.join(gt_project_dir, element)):
        continue
    img_files = os.listdir(os.path.join(gt_project_dir, element, img_dir))
    img_files = [file for file in img_files if file in preview_img_names]

    for img_file in img_files:
        base_name, extension = img_file.split(".")
        original_path = os.path.join(gt_project_dir, element, img_dir, img_file)
        original_image = sly.image.read(original_path)

        gt_ann_path = os.path.join(gt_project_dir, element, ann_dir, img_file + ".json")
        with open(gt_ann_path, "r") as file:
            gt_ann_json = json.load(file)

        gt_ann = sly.Annotation.from_json(gt_ann_json, project_meta)
        gt_ann_image = np.copy(original_image)
        gt_ann.draw_pretty(gt_ann_image, thickness=3)

        pred_ann_path = os.path.join(
            pred_project_dir, element, ann_dir, img_file + ".json"
        )
        with open(pred_ann_path, "r") as file:
            pred_ann_json = json.load(file)

        pred_ann = sly.Annotation.from_json(pred_ann_json, project_meta)
        pred_ann_image = np.copy(original_image)
        pred_ann.draw_pretty(pred_ann_image, thickness=3)

        sly.image.write(
            f"{output_dir}/{base_name}_collage.{extension}",
            np.hstack((original_image, gt_ann_image, pred_ann_image)),
        )
