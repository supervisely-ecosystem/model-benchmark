import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beyond_iou.evaluator import Evaluator
from beyond_iou.loader import build_segmentation_loader

import supervisely as sly
from tqdm import tqdm
from dotenv import load_dotenv
import os
import shutil
import cv2
import numpy as np
import scipy.ndimage
from image_diversity import ClipMetrics
import random
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.figure_factory as ff


# load credentials
load_dotenv("supervisely.env")
api = sly.Api()

# define ground truth and predicted dataset ids
gt_dataset_ids, pred_dataset_ids = [92820], [93741]


# function for downloading project from supervisely platform to local storage
def download_project(dataset_ids, dest_dir):
    project_id = api.dataset.get_info_by_id(dataset_ids[0]).project_id
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


# function for data preprocessing
def prepare_segmentation_data(source_project_dir, output_project_dir, palette):
    if os.path.exists(output_project_dir):
        shutil.rmtree(output_project_dir)
    os.makedirs(output_project_dir)

    ann_dir = "seg"

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


# function for calculating performance metrics
def calculate_metrics(
    gt_dir,
    pred_dir,
    boundary_width,
    boundary_iou_d,
    num_workers,
    output_dir,
    class_names,
):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    if boundary_width % 1 == 0:
        boundary_width = int(boundary_width)
    evaluator = Evaluator(
        class_names=class_names,
        boundary_width=boundary_width,
        boundary_implementation="exact",
        boundary_iou_d=boundary_iou_d,
    )
    loader = build_segmentation_loader(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        gt_label_map=None,
        pred_label_map=None,
        num_workers=num_workers,
    )
    result = evaluator.evaluate(loader)
    result.make_table(path="output/table.md")


# function for getting image ids for each cell of confusion matrix
def get_cell_image_ids(dataset_ids, cell_img_names):
    name2id = {}
    for dataset_id in dataset_ids:
        img_infos = api.image.get_list(dataset_id)
        for img_info in img_infos:
            name2id[img_info.name] = img_info.id
    cell_img_ids = {}
    for cell, img_names in cell_img_names.items():
        img_ids = [name2id[name] for name in img_names]
        cell_img_ids[cell] = img_ids
    return cell_img_ids


# function for getting a set of random image names from specific directory
def get_random_images(subset_size, input_dir, except_list):
    images_filenames = os.listdir(input_dir)
    images_filenames = [img for img in images_filenames if img not in except_list]
    total_length = len(images_filenames)
    selected_indexes = [random.randint(0, total_length - 1) for i in range(subset_size)]
    selected_images_names = [images_filenames[idx] for idx in selected_indexes]
    return selected_images_names


# function for finding the most diverse subset of images from specific directory
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


# function for drawing preview images
def draw_preview_set(
    gt_project_dir,
    pred_project_dir,
    gt_project_id,
    preview_img_names,
    output_dir,
):
    output_dir = "output/preview_set"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

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

            gt_ann_path = os.path.join(
                gt_project_dir, element, ann_dir, img_file + ".json"
            )
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


# function for drawing charts
def draw_charts(df, class_names):
    chart_dir = "output/charts"
    if os.path.exists(chart_dir):
        shutil.rmtree(chart_dir)
    os.makedirs(chart_dir)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Basic segmentation metrics",
            "Intersection & Error over Union",
            "Renormalized Error over Union",
        ),
        specs=[[{"type": "polar"}, {"type": "domain"}, {"type": "xy"}]],
    )
    # first subplot
    categories = [
        "mPixel accuracy",
        "mPrecision",
        "mRecall",
        "mF1-score",
        "mIoU",
        "mBoundaryIoU",
        "mPixel accuracy",
    ]
    num_classes = len(df.index) - 1
    precision = round(df.loc["mean", "precision"] * 100, 1)
    recall = round(df.loc["mean", "recall"] * 100, 1)
    f1_score = round(df.loc["mean", "F1_score"] * 100, 1)
    iou = round(df.loc["mean", "IoU"] * 100, 1)
    boundary_iou = round(df.loc["mean", "boundary_IoU"] * 100, 1)
    overall_TP = df["TP"][:num_classes].sum()
    overall_FN = df["FN"][:num_classes].sum()
    pixel_accuracy = round((overall_TP / (overall_TP + overall_FN)) * 100, 1)
    values = [
        pixel_accuracy,
        precision,
        recall,
        f1_score,
        iou,
        boundary_iou,
        pixel_accuracy,
    ]
    trace_1 = go.Scatterpolar(
        mode="lines+text",
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="cornflowerblue",
        line_color="blue",
        opacity=0.6,
        text=values,
        textposition=[
            "bottom right",
            "top center",
            "top center",
            "middle left",
            "bottom center",
            "bottom right",
            "bottom right",
        ],
        textfont=dict(color="blue"),
    )
    fig.add_trace(trace_1, row=1, col=1)
    # second subplot
    labels = ["mIoU", "mBoundaryEoU", "mExtentEoU", "mSegmentEoU"]
    boundary_eou = round(df.loc["mean", "E_boundary_oU"] * 100, 1)
    extent_eou = round(df.loc["mean", "E_extent_oU"] * 100, 1)
    segment_eou = round(df.loc["mean", "E_segment_oU"] * 100, 1)
    values = [iou, boundary_eou, extent_eou, segment_eou]
    trace_2 = go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        textposition="outside",
        textinfo="percent+label",
        marker=dict(colors=["cornflowerblue", "moccasin", "lightgreen", "orangered"]),
    )
    fig.add_trace(trace_2, row=1, col=2)
    # third subplot
    labels = ["boundary", "extent", "segment"]
    boundary_renormed_eou = round(df.loc["mean", "E_boundary_oU_renormed"] * 100, 1)
    extent_renormed_eou = round(df.loc["mean", "E_extent_oU_renormed"] * 100, 1)
    segment_renormed_eou = round(df.loc["mean", "E_segment_oU_renormed"] * 100, 1)
    values = [boundary_renormed_eou, extent_renormed_eou, segment_renormed_eou]
    trace_3 = go.Bar(
        x=labels,
        y=values,
        orientation="v",
        text=values,
        width=[0.5, 0.5, 0.5],
        textposition="outside",
        marker_color=["moccasin", "lightgreen", "orangered"],
    )
    fig.add_trace(trace_3, row=1, col=3)
    fig.update_layout(
        height=400,
        width=1200,
        polar=dict(
            radialaxis=dict(
                visible=True, showline=False, showticklabels=False, range=[0, 100]
            )
        ),
        showlegend=False,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        yaxis=dict(showticklabels=False),
        yaxis_range=[0, int(max(values)) + 4],
    )
    fig.layout.annotations[0].update(y=1.2)
    fig.layout.annotations[1].update(y=1.2)
    fig.layout.annotations[2].update(y=1.2)
    fig.write_image(f"{chart_dir}/overall_error_analysis.png", scale=2.0)
    fig.write_html(f"{chart_dir}/overall_error_analysis.html")
    overall_chart_img = sly.image.read(f"{chart_dir}/overall_error_analysis.png")
    chart_height, chart_width = overall_chart_img.shape[0], overall_chart_img.shape[1]
    # fourth plot
    df.drop(["mean"], inplace=True)
    df = df[["IoU", "E_extent_oU", "E_boundary_oU", "E_segment_oU"]]
    df.sort_values(by="IoU", ascending=False, inplace=True)
    labels = list(df.index)
    color_palette = ["cornflowerblue", "moccasin", "lightgreen", "orangered"]

    fig = go.Figure()
    for i, column in enumerate(df.columns):
        fig.add_trace(
            go.Bar(
                name=column,
                y=df[column],
                x=labels,
                marker_color=color_palette[i],
            )
        )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title={
            "text": "Classwise segmentation error analysis",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(size=24),
    )
    fig.write_image(
        f"{chart_dir}/classwise_error_analysis.png",
        scale=1.0,
        width=chart_width,
        height=chart_height,
    )
    fig.write_html(f"{chart_dir}/classwise_error_analysis.html")
    # fifth plot
    confusion_matrix = np.load("output/confusion_matrix.npy")
    confusion_matrix = confusion_matrix[::-1]
    x = class_names
    y = x[::-1].copy()
    text_anns = [[str(el) for el in row] for row in confusion_matrix]

    fig = ff.create_annotated_heatmap(
        confusion_matrix, x=x, y=y, annotation_text=text_anns, colorscale="orrd"
    )

    fig.update_layout(
        title={
            "text": "Confusion matrix",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        font=dict(size=24),
    )

    fig.add_annotation(
        dict(
            font=dict(color="black", size=24),
            x=0.5,
            y=-0.1,
            showarrow=False,
            text="Predicted",
            xref="paper",
            yref="paper",
        )
    )
    fig.add_annotation(
        dict(
            font=dict(color="black", size=24),
            x=-0.12,
            y=0.5,
            showarrow=False,
            text="Ground truth",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    fig.update_layout(margin=dict(t=150, l=300))
    fig["data"][0]["showscale"] = True
    fig.write_image(
        f"{chart_dir}/confusion_matrix.png",
        scale=1.0,
        width=chart_width,
        height=chart_height,
    )
    fig.write_html(f"{chart_dir}/confusion_matrix.html")
    # combine two chart images into one
    classwise_chart_img = sly.image.read(f"{chart_dir}/classwise_error_analysis.png")
    conf_matrix_chart_img = sly.image.read(f"{chart_dir}/confusion_matrix.png")
    sly.image.write(
        f"{chart_dir}/total_dashboard.png",
        np.vstack((overall_chart_img, classwise_chart_img, conf_matrix_chart_img)),
    )


# function for convenient using all functions above
def united_func(
    gt_dataset_ids, pred_dataset_ids, subset_size=3, n_iter=15, batch_size=8
):
    gt_sly_dir, pred_sly_dir = "./sly_data/gt", "./sly_data/pred"
    download_project(gt_dataset_ids, gt_sly_dir)
    download_project(pred_dataset_ids, pred_sly_dir)

    gt_project_id = api.dataset.get_info_by_id(gt_dataset_ids[0]).project_id
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(gt_project_id))
    obj_classes = project_meta.obj_classes
    classes_json = obj_classes.to_json()
    class_names = [obj["title"] for obj in classes_json]
    palette = [obj["color"].lstrip("#") for obj in classes_json]
    palette = [[int(color[i : i + 2], 16) for i in (0, 2, 4)] for color in palette]

    gt_mmseg_dir, pred_mmseg_dir = "./mmseg_data/gt", "./mmseg_data/pred"
    prepare_segmentation_data(gt_sly_dir, gt_mmseg_dir, palette)
    prepare_segmentation_data(pred_sly_dir, pred_mmseg_dir, palette)

    calculate_metrics(
        gt_dir=gt_mmseg_dir,
        pred_dir=pred_mmseg_dir,
        boundary_width=0.01,
        boundary_iou_d=0.02,
        num_workers=4,
        output_dir="./output",
        class_names=class_names,
    )

    with open("output/cell_img_names.json", "r") as file:
        cell_img_names = json.load(file)

    cell_img_ids = get_cell_image_ids(pred_dataset_ids, cell_img_names)

    with open("output/cell_img_ids.json", "w") as file:
        json.dump(cell_img_ids, file)

    input_dir = "sly_data/gt/test/img"

    with open("output/img2iou.json", "r") as file:
        img2iou = json.load(file)

    img2iou_sorted = sorted(img2iou.items(), key=lambda item: item[1])
    lowest_iou_images = [element[0] for element in img2iou_sorted[-subset_size:]]
    highest_iou_images = [element[0] for element in img2iou_sorted[:subset_size]]
    except_list = lowest_iou_images + highest_iou_images
    diverse_images = get_diverse_images(
        subset_size, input_dir, n_iter, batch_size, except_list
    )
    preview_img_names = highest_iou_images + lowest_iou_images + diverse_images

    draw_preview_set(
        gt_project_dir=gt_sly_dir,
        pred_project_dir=pred_sly_dir,
        gt_project_id=gt_project_id,
        preview_img_names=preview_img_names,
        output_dir="output/preview_set",
    )

    df = pd.read_csv("output/result_df.csv", index_col="Unnamed: 0")
    draw_charts(df, class_names)


united_func(gt_dataset_ids, pred_dataset_ids)
