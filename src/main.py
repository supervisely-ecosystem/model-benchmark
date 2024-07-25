import json
from typing import *

import plotly.graph_objects as go
from tqdm import tqdm

import src.globals as g
import supervisely as sly

# from src.ui.outcome_counts import plotly_outcome_counts
from supervisely._utils import camel_to_snake
from supervisely.app.widgets import *
from supervisely.app.widgets import Button, Card, Container, Sidebar, Text


def main_func():

    api = sly.Api.from_env()
    project = g.api.project.get_info_by_id(sel_project.get_selected_id())

    bm = sly.nn.ObjectDetectionBenchmark(api, project.id, output_dir=g.STORAGE_DIR + "/benchmark")
    bm.run_evaluation(model_session=sel_app_session.get_selected_id())
    # bm.evaluate(g.dt_project_id)
    eval_res_dir = f"/model-benchmark/evaluation/{project.id}_{project.name}/"
    bm.upload_eval_results(eval_res_dir)

    bm.visualize()
    bm.upload_visualizations(eval_res_dir + "visualizations/")
    if g.task_id is not None:
        files = api.file.list2(g.team_id, eval_res_dir, recursive=False)
        file_id = files[0].id
        g.api.task.set_output_directory(g.task_id, file_id, eval_res_dir)


sel_app_session = SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = SelectProject(default_id=g.project_id, workspace_id=g.workspace_id)
button = Button("Evaluate")
layout = Container(
    widgets=[
        Text("Select GT Project"),
        sel_project,
        sel_app_session,
        button,
    ]
)


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)

# выбор таски
# Run Evaluation
#
