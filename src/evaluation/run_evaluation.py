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
from src.utils import IdMapper
from src.click_data import ClickData


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


def add_tags_to_dt_project(api: sly.Api, matches: list, dt_project_id: int, cocoGt_dataset: dict, cocoDt_dataset: dict):
    # outcome_tag_meta = sly.TagMeta("outcome", sly.TagValueType.ANY_STRING, applicable_to=sly.TagApplicableTo.OBJECTS_ONLY)
    match_tag_meta = sly.TagMeta("matched_gt_id", sly.TagValueType.ANY_NUMBER, applicable_to=sly.TagApplicableTo.OBJECTS_ONLY)
    iou_tag_meta = sly.TagMeta("iou", sly.TagValueType.ANY_NUMBER, applicable_to=sly.TagApplicableTo.OBJECTS_ONLY)

    # update project meta with new tag metas
    meta = api.project.get_meta(dt_project_id)
    meta = sly.ProjectMeta.from_json(meta)
    meta_old = meta
    # if not meta.tag_metas.has_key("outcome"):
    #     meta = meta.add_tag_meta(outcome_tag_meta)
    if not meta.tag_metas.has_key("matched_gt_id"):
        meta = meta.add_tag_meta(match_tag_meta)
    if not meta.tag_metas.has_key("iou"):
        meta = meta.add_tag_meta(iou_tag_meta)
    if meta != meta_old:
        meta = api.project.update_meta(dt_project_id, meta)

    # get tag metas
    # outcome_tag_meta = meta.get_tag_meta("outcome")
    match_tag_meta = meta.get_tag_meta("matched_gt_id")
    iou_tag_meta = meta.get_tag_meta("iou")

    # mappings
    gt_img_mapping, gt_ann_mapping = get_id_mappings(cocoGt_dataset)
    dt_img_mapping, dt_ann_mapping = get_id_mappings(cocoDt_dataset)

    # add tags to objects
    for match in tqdm(matches):
        if match["type"] == "TP":
            outcome = "TP"
            matched_gt_id = gt_ann_mapping[match["gt_id"]]
            ann_dt_id = dt_ann_mapping[match["dt_id"]]
            iou = match["iou"]
            # api.advanced.add_tag_to_object(outcome_tag_meta.sly_id, ann_dt_id, str(outcome))
            api.advanced.add_tag_to_object(match_tag_meta.sly_id, ann_dt_id, int(matched_gt_id))
            api.advanced.add_tag_to_object(iou_tag_meta.sly_id, ann_dt_id, float(iou))
        elif match["type"] == "FP":
            outcome = "FP"
            # api.advanced.add_tag_to_object(outcome_tag_meta.sly_id, ann_dt_id, str(outcome))
        elif match["type"] == "FN":
            outcome = "FN"
        else:
            raise ValueError(f"Unknown match type: {match['type']}")


def get_id_mappings(coco_dataset: dict):
    img_mapping = {x['id']: x['sly_id'] for x in coco_dataset['images']}
    ann_mapping = {x['id']: x['sly_id'] for x in coco_dataset['annotations']}
    return img_mapping, ann_mapping


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
    
    eval_data_path = os.path.join(base_dir, "eval_data.pkl")
    with open(eval_data_path, "rb") as f:
        eval_data = pickle.load(f)

    cocoGt_path = os.path.join(base_dir, "cocoGt.json")
    cocoDt_path = os.path.join(base_dir, "cocoDt.json")
    with open(cocoGt_path, 'r') as f:
        cocoGt_dataset= json.load(f)
    with open(cocoDt_path, 'r') as f:
        cocoDt_dataset = json.load(f)

    cocoGt = COCO()
    cocoGt.dataset = cocoGt_dataset
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes(cocoDt_dataset['annotations'])

    m = MetricProvider(eval_data['matches'], eval_data['coco_metrics'], eval_data['params'], cocoGt, cocoDt)

    api = sly.Api()
    matches = eval_data['matches']
    dt_project_id = 39052

    gt_id_mapper = IdMapper(cocoGt_dataset)
    dt_id_mapper = IdMapper(cocoDt_dataset)

    click_data = ClickData(m, gt_id_mapper, dt_id_mapper)
    click_data.create_data()
