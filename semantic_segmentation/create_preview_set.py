"""Script for creating set of model predictions consisting of best, worst and diverse predictions"""

from image_diversity import ClipMetrics
import os
import random
from tqdm import tqdm
import json


subset_size = 3
input_dir = "sly_data/gt/test/img"
n_iter = 15
batch_size = 8


with open("output/img2iou.json", "r") as file:
    img2iou = json.load(file)

img2iou_sorted = sorted(img2iou.items(), key=lambda item: item[1])
lowest_iou_images = [element[0] for element in img2iou_sorted[-subset_size:]]
highest_iou_images = [element[0] for element in img2iou_sorted[:subset_size]]

except_list = lowest_iou_images + highest_iou_images


def get_random_images(subset_size, input_dir, except_list):
    images_filenames = os.listdir(input_dir)
    images_filenames = [img for img in images_filenames if img not in except_list]
    total_length = len(images_filenames)
    selected_indexes = [random.randint(0, total_length - 1) for i in range(subset_size)]
    selected_images_names = [images_filenames[idx] for idx in selected_indexes]
    return selected_images_names


def get_diverse_images(subset_size, input_dir, n_iter, batch_size, except_list):
    diverse_image_names = None
    highest_tce = float("-inf")
    clip_metrics = ClipMetrics(n_eigs=subset_size - 1)

    for iter in tqdm(range(n_iter)):
        img_names = get_random_images(subset_size, input_dir, except_list)
        tce = clip_metrics.tce(
            img_dir=input_dir, img_names=img_names, batch_size=batch_size
        )
        if tce > highest_tce:
            highest_tce = tce
            diverse_image_names = img_names

    return diverse_image_names


diverse_images = get_diverse_images(
    subset_size, input_dir, n_iter, batch_size, except_list
)

with open("output/preview_set.json", "w") as file:
    preview_set_dict = {
        "highest_iou": highest_iou_images,
        "lowest_iou": lowest_iou_images,
        "diverse": diverse_images,
    }
    json.dump(preview_set_dict, file)
