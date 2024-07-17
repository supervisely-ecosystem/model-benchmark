"""Script for getting image ids for each cell of confusion matrix"""

import supervisely as sly
from dotenv import load_dotenv
import json


load_dotenv("supervisely.env")
api = sly.Api()
dataset_ids = [92820]
name2id = {}

for dataset_id in dataset_ids:
    img_infos = api.image.get_list(dataset_id)
    for img_info in img_infos:
        name2id[img_info.name] = img_info.id

with open("output/cell_img_names.json", "r") as file:
    cell_img_names = json.load(file)

cell_img_ids = {}
for cell, img_names in cell_img_names.items():
    img_ids = [name2id[name] for name in img_names]
    cell_img_ids[cell] = img_ids

with open("output/cell_img_ids.json", "w") as file:
    json.dump(cell_img_ids, file)
