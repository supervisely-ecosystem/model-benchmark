"""Script for downloading ground truth and predicted Supervisely datasets to local storage"""

import supervisely as sly
from tqdm import tqdm
from dotenv import load_dotenv
import os
import shutil


load_dotenv("supervisely.env")
api = sly.Api()
gt_dest_dir, pred_dest_dir = "./sly_data/gt", "./sly_data/pred"
gt_project_id, pred_project_id = 39106, 39486
gt_dataset_ids, pred_dataset_ids = [92820], [93741]


def download_project(project_id, dataset_ids, dest_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    if len(os.listdir(dest_dir)) == 0:
        dataset_info = api.dataset.get_info_by_id(dataset_ids[0])
        n_images = dataset_info.items_count

        pbar = tqdm(desc="Downloading project...", total=n_images)
        sly.download(
            api,
            project_id,
            dest_dir,
            dataset_ids,
            progress_cb=pbar,
        )


download_project(gt_project_id, gt_dataset_ids, gt_dest_dir)
download_project(pred_project_id, pred_dataset_ids, pred_dest_dir)
