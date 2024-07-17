"""Script for converting ground truth and predicted Supervisely datasets to MMSegmentation format"""

import supervisely as sly
import os
import cv2
import shutil
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm


gt_project_id = 39106
gt_source_project_dir, pred_source_project_dir = "./sly_data/gt", "./sly_data/pred"
gt_output_project_dir, pred_output_project_dir = "./mmseg_data/gt", "./mmseg_data/pred"
if os.path.exists(gt_output_project_dir):
    shutil.rmtree(gt_output_project_dir)
os.makedirs(gt_output_project_dir)
if os.path.exists(pred_output_project_dir):
    shutil.rmtree(pred_output_project_dir)
os.makedirs(pred_output_project_dir)

ann_dir = "seg"

load_dotenv("supervisely.env")
api = sly.Api()

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(gt_project_id))
obj_classes = project_meta.obj_classes
# if project_meta.get_obj_class("__bg__") is None:
#     obj_classes = obj_classes.add(
#         sly.ObjClass(name="__bg__", geometry_type=sly.Bitmap, color=(0, 0, 0))
#     )
classes_json = obj_classes.to_json()
classes = [obj["title"] for obj in classes_json]
palette = [obj["color"].lstrip("#") for obj in classes_json]
palette = [[int(color[i : i + 2], 16) for i in (0, 2, 4)] for color in palette]


def prepare_segmentation_data(source_project_dir, output_project_dir, ann_dir, palette):
    temp_project_seg_dir = source_project_dir + "_temp"
    if not os.path.exists(temp_project_seg_dir):
        sly.Project.to_segmentation_task(
            source_project_dir,
            temp_project_seg_dir,
        )

    datasets = os.listdir(temp_project_seg_dir)
    for dataset in datasets:
        if not os.path.isdir(os.path.join(temp_project_seg_dir, dataset)):
            continue
        # convert masks to required format and save to general ann_dir
        mask_files = os.listdir(os.path.join(temp_project_seg_dir, dataset, ann_dir))
        for mask_file in tqdm(mask_files):
            mask = cv2.imread(
                os.path.join(temp_project_seg_dir, dataset, ann_dir, mask_file)
            )[:, :, ::-1]
            result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
            # human masks to machine masks
            for color_idx, color in enumerate(palette):
                colormap = np.where(np.all(mask == color, axis=-1))
                result[colormap] = color_idx
            if mask_file.count(".png") > 1:
                mask_file = mask_file[:-4]
            cv2.imwrite(os.path.join(output_project_dir, mask_file), result)

    shutil.rmtree(temp_project_seg_dir)


prepare_segmentation_data(
    gt_source_project_dir, gt_output_project_dir, ann_dir, palette
)
prepare_segmentation_data(
    pred_source_project_dir, pred_output_project_dir, ann_dir, palette
)
