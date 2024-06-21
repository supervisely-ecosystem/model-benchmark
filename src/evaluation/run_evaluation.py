import json
import os
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO

import supervisely as sly
from supervisely.nn.inference import SessionJSON
from supervisely.nn.benchmark.sly2coco import sly2coco
from supervisely.nn.benchmark.calculate_metrics import calculate_metrics
from supervisely.nn.benchmark.metric_provider import MetricProvider


def evaluate(
        gt_project_id,
        model_session_id,
        cache_project=True,
        gt_dataset_ids=None,
        inference_settings=None,
        batch_size=8,
        save_path="APP_DATA"
        ):
    api = sly.Api()
    dt_project_info = run_inference(api, model_session_id, inference_settings, gt_project_id, gt_dataset_ids, batch_size, cache_project)
    
    base_dir = os.path.join(save_path, dt_project_info.name)
    gt_path = os.path.join(base_dir, "gt_project")
    dt_path = os.path.join(base_dir, "dt_project")
    print(f"GT annotations will be downloaded to: {gt_path}")
    sly.download_project(api, gt_project_id, gt_path, gt_dataset_ids, log_progress=True, save_images=False, save_image_info=True)
    print(f"DT annotations will be downloaded to: {dt_path}")
    sly.download_project(api, dt_project_info.id, dt_path, log_progress=True, save_images=False, save_image_info=True)
    
    # 3. Sly2Coco
    cocoGt_json = sly2coco(gt_path, is_dt_dataset=False)
    cocoGt_path = os.path.join(base_dir, "cocoGt.json")
    with open(cocoGt_path, 'w') as f:
        json.dump(cocoGt_json, f)
    print(f"cocoGt.json: {cocoGt_path}")

    cocoDt_json = sly2coco(dt_path, is_dt_dataset=True)
    cocoDt_path = os.path.join(base_dir, "cocoDt.json")
    with open(cocoDt_path, 'w') as f:
        json.dump(cocoDt_json, f)
    print(f"cocoDt.json: {cocoDt_path}")

    assert cocoDt_json['categories'] == cocoGt_json['categories']
    assert [x['id'] for x in cocoDt_json['images']] == [x['id'] for x in cocoGt_json['images']]

    # 4. Calculate metrics
    cocoGt = COCO()
    cocoGt.dataset = cocoGt_json
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt_json['annotations'])
    eval_data = calculate_metrics(cocoGt, cocoDt)
    eval_data_path = os.path.join(base_dir, "eval_data.pkl")
    with open(eval_data_path, "wb") as f:
        pickle.dump(eval_data, f)
    print(f"eval_data.pkl: {eval_data_path}")

    # MetricProvider
    m = MetricProvider(eval_data['matches'], eval_data['coco_metrics'], eval_data['params'], cocoGt, cocoDt)
    print(m.base_metrics())
    print("Done!")


def run_inference(
        api: sly.Api,
        model_session_id: int,
        inference_settings: dict,
        gt_project_id: int,
        gt_dataset_ids: list = None,
        batch_size: int = 8,
        cache_project: bool = True,
        dt_wrokspace_id: int = None,
        dt_project_name: str = None,
        ):
    session = SessionJSON(api, model_session_id, inference_settings=inference_settings)
    session_info = session.get_session_info()

    # TODO: make it in apps
    # evaluation_info = {
    #     "model_name": session_info["model_name"],
    #     "display_model_name": session_info["display_model_name"],
    #     "architecture": session_info["architecture"],
    #     "deploy_params": session_info["deploy_params"],
    #     "pretrained_checkpoint_info": session_info["pretrained_checkpoint_info"],
    #     "custom_checkpoint_info": session_info["custom_checkpoint_info"],
    # }
    evaluation_info = session_info

    task_info = api.task.get_info_by_id(model_session_id)
    app_info = task_info["meta"]["app"]
    app_info = {
        app_info["name"]: app_info["name"],
        app_info["version"]: app_info["version"],
        app_info["id"]: app_info["id"],
    }


    gt_project_info = api.project.get_info_by_id(gt_project_id)
    if dt_project_name is None:
        dt_project_name = gt_project_info.name + " - " + session_info["app_name"]  #  + checkpoint_id
    if dt_wrokspace_id is None:
        dt_wrokspace_id = gt_project_info.workspace_id
    dt_project_info = api.project.create(dt_wrokspace_id, dt_project_name)
    dt_project_id = dt_project_info.id

    iterator = session.inference_project_id_async(
        gt_project_id,
        gt_dataset_ids,
        output_project_id=dt_project_id,
        cache_project_on_model=cache_project,
        batch_size=batch_size,
    )

    for _ in tqdm(iterator):
        pass

    custom_data = {
        "evaluation_info": {
            "gt_project_id": gt_project_id,
            "gt_dataset_ids": gt_dataset_ids,
            "inference_params": {
                "batch_size": batch_size,
                "inference_settings": inference_settings,
            },
            # **evaluation_info,  # TODO: add evaluation_info
            "app_info": app_info,
        },
    }
    api.project.update_custom_data(dt_project_id, custom_data)

    return dt_project_info


if __name__ == "__main__":
    gt_project_id = 36401
    model_session_id = 61668
    inference_settings = {
        "conf": 0.05,
    }
    # evaluate(
    #     gt_project_id,
    #     model_session_id,
    #     cache_project=False,
    #     gt_dataset_ids=None,
    #     inference_settings=inference_settings,
    #     batch_size=8,
    #     save_path="APP_DATA"
    #     )

    base_dir = "APP_DATA/COCO 2017 val - Serve YOLO (v8, v9)"
    gt_path = os.path.join(base_dir, "gt_project")
    dt_path = os.path.join(base_dir, "dt_project")
    
    # 3. Sly2Coco
    cocoGt_json = sly2coco(gt_path, is_dt_dataset=False, accepted_shapes=['rectangle'])
    cocoGt_path = os.path.join(base_dir, "cocoGt.json")
    with open(cocoGt_path, 'w') as f:
        json.dump(cocoGt_json, f)
    print(f"cocoGt.json: {cocoGt_path}")

    cocoDt_json = sly2coco(dt_path, is_dt_dataset=True, accepted_shapes=['rectangle'])
    cocoDt_path = os.path.join(base_dir, "cocoDt.json")
    with open(cocoDt_path, 'w') as f:
        json.dump(cocoDt_json, f)
    print(f"cocoDt.json: {cocoDt_path}")

    assert cocoDt_json['categories'] == cocoGt_json['categories']
    assert [x['id'] for x in cocoDt_json['images']] == [x['id'] for x in cocoGt_json['images']]

    # 4. Calculate metrics
    cocoGt = COCO()
    cocoGt.dataset = cocoGt_json
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt_json['annotations'])
    eval_data = calculate_metrics(cocoGt, cocoDt)
    eval_data_path = os.path.join(base_dir, "eval_data.pkl")
    with open(eval_data_path, "wb") as f:
        pickle.dump(eval_data, f)
    print(f"eval_data.pkl: {eval_data_path}")

    # MetricProvider
    m = MetricProvider(eval_data['matches'], eval_data['coco_metrics'], eval_data['params'], cocoGt, cocoDt)
    print(m.base_metrics())
    print("Done!")
