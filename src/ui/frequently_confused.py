import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

import src.globals as g
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DatasetThumbnail,
    Dialog,
    GridGalleryV2,
    IFrame,
    Markdown,
    OneOf,
    PlotlyChart,
    RadioGroup,
    SelectDataset,
    Switch,
    Text,
)
from supervisely.nn.benchmark import metric_provider
from supervisely.nn.benchmark.metric_provider import METRIC_NAMES, MetricProvider


def frequently_confused():
    confusion_matrix = g.m.confusion_matrix()

    # Frequency of confusion as bar chart
    confused_df = g.m.frequently_confused(confusion_matrix, topk_pairs=20)
    confused_name_pairs = confused_df["category_pair"]
    confused_prob = confused_df["probability"]
    confused_cnt = confused_df["count"]
    x_labels = [f"{pair[0]} - {pair[1]}" for pair in confused_name_pairs]
    figs = []
    for y_labels in (confused_prob, confused_cnt):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=x_labels, y=y_labels, marker=dict(color=confused_prob, colorscale="Reds"))
        )
        fig.update_layout(
            # title="Frequently confused class pairs",
            xaxis_title="Class pair",
            yaxis_title=y_labels.name.capitalize(),
        )
        fig.update_traces(text=y_labels.round(2))
        figs.append(fig)
    return figs


df = g.m.frequently_confused(g.m.confusion_matrix(), topk_pairs=20)
pair = df["category_pair"][0]
prob = df["probability"][0]

markdown = Markdown(
    f"""
### Frequently Confused Classes

This chart displays the most frequently confused pairs of classes.
In general, it finds out which classes visually seem very similar to the model.

The chart calculates the **probability of confusion** between different pairs of classes. 
For instance, if the probability of confusion for the pair “{pair[0]} - {pair[1]}” is {prob:.2f}, this means that when the model predicts either “{pair[0]}” or “{pair[1]}”, there is a {prob*100:.0f}% chance that the model might mistakenly predict one instead of the other.

The measure is class-symmetric, meaning that the probability of confusing a {pair[0]} with a {pair[1]} is equal to the probability of confusing a {pair[1]} with a {pair[0]}.
""",
    show_border=False,
)

# iframe_frequently_confused_prob = IFrame(
#     "static/09_01_frequently_confused.html", width=620, height=520
# )
# iframe_frequently_confused_count = IFrame(
#     "static/09_02_frequently_confused.html", width=620, height=520
# )

fig1, fig2 = frequently_confused()
plotly_frequently_confused_prob = PlotlyChart(fig1)
plotly_frequently_confused_count = PlotlyChart(fig2)

dialog_gallery_freq = GridGalleryV2(columns_number=4, enable_zoom=False)

dialog_container_ = Container([dialog_gallery_freq])
dialog_ = Dialog(content=dialog_container_)


@plotly_frequently_confused_prob.click
def click_handler_prob(datapoints):
    plotly_frequently_confused_prob.loading = True
    for datapoint in datapoints:
        _pair = tuple(datapoint.x.split(" - "))
        break

    dialog_gallery_freq.clean_up()
    image_ids = list(set([x["dt_img_id"] for x in g.click_data.frequently_confused[_pair]]))
    image_infos = [x for x in g.dt_image_infos if x.id in image_ids][:20]
    if len(image_infos) < 5:  # TODO rm later
        image_infos = [image_infos[0]] * 4
    anns_infos = [x for x in g.dt_anns_infos if x.image_id in image_ids][:20]
    if len(anns_infos) < 5:  # TODO rm later
        anns_infos = [anns_infos[0]] * 4

    for idx, (image_info, ann_info) in enumerate(zip(image_infos, anns_infos)):
        image_name = image_info.name
        image_url = image_info.full_storage_url

        dialog_gallery_freq.append(
            title=image_name,
            image_url=image_url,
            annotation_info=ann_info,
            column_index=idx % dialog_gallery_freq.columns_number,
            project_meta=g.dt_project_meta,
        )

    dialog_.title = datapoint.x
    plotly_frequently_confused_prob.loading = False
    dialog_.show()


@plotly_frequently_confused_count.click
def click_handler_count(datapoints):
    plotly_frequently_confused_count.loading = True
    for datapoint in datapoints:
        pair = tuple(datapoint.x.split(" - "))
        break

    image_ids = list(set([x["dt_img_id"] for x in g.click_data.frequently_confused[pair]]))
    image_infos = [x for x in g.dt_image_infos if x.id in image_ids][:20]
    anns_infos = [x for x in g.dt_anns_infos if x.image_id in image_ids][:20]

    for idx, (image_info, ann_info) in enumerate(zip(image_infos, anns_infos)):
        image_name = image_info.name
        image_url = image_info.full_storage_url

        dialog_gallery_freq.append(
            title=image_name,
            image_url=image_url,
            annotation_info=ann_info,
            column_index=idx % dialog_gallery_freq.columns_number,
            project_meta=g.dt_project_meta,
        )
    dialog_.title = datapoint.x
    plotly_frequently_confused_count.loading = False
    dialog_.show()


radio_group = RadioGroup(
    [
        RadioGroup.Item("probabilty", "Probability", content=plotly_frequently_confused_prob),
        RadioGroup.Item("count", "Count", content=plotly_frequently_confused_count),
    ]
)
radio_one_of = OneOf(radio_group)

container = Container(
    widgets=[
        markdown,
        plotly_frequently_confused_prob,
        # radio_group,
        # radio_one_of,
    ]
)
