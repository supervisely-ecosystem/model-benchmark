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
    IFrame,
    Markdown,
    OneOf,
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
# markdown_1 = Markdown(
#     f"""
# ### Frequently Confused Classes

# This chart displays the most frequently confused pairs of classes.
# In general, it finds out which classes visually seem very similar to the model.
# """,
#     show_border=False,
#     height=50,
# )
# info = Text(
#     f"""The chart calculates the <b>probability of confusion</b> between different pairs of classes. For instance, if the probability of confusion for the pair “{pair[0]} - {pair[1]}” is {prob:.2f}, this means that when the model predicts either “{pair[0]}” or “{pair[1]}”, there is a {prob*100:.0f}% chance that the model might mistakenly predict one instead of the other.""",
#     status="info",
# )
# markdown_2 = Markdown(
#     f"""
# The measure is class-symmetric, meaning that the probability of confusing a {pair[0]} with a {pair[1]} is equal to the probability of confusing a {pair[1]} with a {pair[0]}.
# """,
#     show_border=False,
#     height=50,
# )

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

iframe_frequently_confused_prob = IFrame(
    "static/09_01_frequently_confused.html", width=620, height=520
)
iframe_frequently_confused_count = IFrame(
    "static/09_02_frequently_confused.html", width=620, height=520
)

swicther = Switch(
    switched=True,
    on_text="Probability",
    off_text="Count",
    width=100,
    on_content=iframe_frequently_confused_prob,
    off_content=iframe_frequently_confused_count,
)
switch_one_of = OneOf(swicther)

container = Container(
    widgets=[
        markdown,
        swicther,
        switch_one_of,
    ]
)
