import os
from pathlib import Path

from dotenv import load_dotenv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
import numpy as np
from tqdm import tqdm
import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()


SLY_APP_DATA_DIR = sly.app.get_data_dir()
STATIC_DIR = os.path.join(SLY_APP_DATA_DIR, "static")
sly.fs.mkdir(STATIC_DIR)

project_id = 38914
dataset_id = 92290

meta = sly.ProjectMeta.from_json(data=api.project.get_meta(id=project_id))
tagmeta_conf = sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
images = api.image.get_list(dataset_id=dataset_id)
if meta.get_tag_meta(tagmeta_conf.name) is None:
    meta = meta.add_tag_meta(tagmeta_conf)

api.project.update_meta(project_id, meta.to_json())

with tqdm(total=len(images)) as pbar:
    for image_batch in sly.batched(images):
        new_anns_batch = []
        image_ids = [x.id for x in image_batch]
        ann_infos = api.annotation.download_batch(dataset_id, image_ids)
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            labels = []
            for label in ann.labels:
                new = label.clone(tags=[sly.Tag(tagmeta_conf, 1.0)])
                labels.append(new)

            new_anns_batch.append(ann.clone(labels=labels))
            pbar.update(1)

        api.annotation.upload_anns(image_ids, new_anns_batch)

pass





    # img_id to [matches]
    img_matches = {}
    for match in matches:
        dt_img_id = dt_img_mapping[match["img_id"]]
        if dt_img_id not in img_matches:
            img_matches[dt_img_id] = []
        img_matches[dt_img_id].append(match)

    for image_ids in sly.batched(img_matches.keys()):
        m = img_matches[img_ids]
        new_anns_batch = []

        ann_infos = api.annotation.download_batch(dataset_id, image_ids)
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            labels = []
            for label in ann.labels:
                new = label.clone(tags=[sly.Tag(tagmeta_conf, 1.0)])
                labels.append(new)

            new_anns_batch.append(ann.clone(labels=labels))
        api.annotation.upload_anns(image_ids, new_anns_batch)