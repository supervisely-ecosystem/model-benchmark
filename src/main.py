import json
from typing import *

import plotly.graph_objects as go
from tqdm import tqdm

import src.globals as g
import supervisely as sly

# from src.ui.outcome_counts import plotly_outcome_counts
from supervisely._utils import abs_url, camel_to_snake, is_development
from supervisely.app.widgets import *
from supervisely.nn.benchmark.object_detection_benchmark import ObjectDetectionBenchmark


def main_func():
    api = g.api
    project = api.project.get_info_by_id(sel_project.get_selected_id())
    session_id = sel_app_session.get_selected_id()

    bm = ObjectDetectionBenchmark(api, project.id, output_dir=g.STORAGE_DIR + "/benchmark")
    sly.logger.info("Session ID={}".format(session_id))
    bm.run_evaluation(model_session=session_id)
    # bm.evaluate(g.dt_project_id)
    eval_res_dir = f"/model-benchmark/evaluation/{project.id}_{project.name}/"
    bm.upload_eval_results(eval_res_dir)

    bm.visualize()
    bm.upload_visualizations(eval_res_dir + "visualizations/")
    creating_report_f.hide()

    template_vis_file = api.file.get_info_by_path(
        sly.env.team_id(), eval_res_dir + "visualizations/template.vue"
    )
    # lnk = f"/model-benchmark?id={template_vis_file.id}"
    # lnk = abs_url(lnk) if is_development() else lnk
    # report_model_benchmark.set(
    #     f"<a href='{lnk}' target='_blank'>Open report for the best model</a>",
    #     "success",
    # )
    report_model_benchmark.set(template_vis_file)
    report_model_benchmark.show()

    g.workflow.add_input(session_id)
    g.workflow.add_input(project)
    g.workflow.add_output(bm.diff_project_info)
    g.workflow.add_output(bm.dt_project_info)
    g.workflow.add_output(eval_res_dir)


sel_app_session = SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = SelectProject(default_id=None, workspace_id=g.workspace_id)
button = Button("Evaluate")
report_model_benchmark = ReportThumbnail()
report_model_benchmark.hide()
creating_report_f = Field(Empty(), "", "Creating report on model...")
creating_report_f.hide()

layout = Container(
    widgets=[
        Text("Select GT Project"),
        sel_project,
        sel_app_session,
        button,
        creating_report_f,
        report_model_benchmark,
    ]
)


@button.click
def handle():
    creating_report_f.show()
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)

# выбор таски
# Run Evaluation
#
