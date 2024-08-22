from typing import Optional

import supervisely as sly
import supervisely.app.widgets as w
from supervisely.nn.benchmark import ObjectDetectionBenchmark

import src.globals as g


def main_func():
    api = g.api
    project = api.project.get_info_by_id(sel_project.get_selected_id())
    session_id = sel_app_session.get_selected_id()
    pbar.show()
    report_model_benchmark.hide()

    bm = ObjectDetectionBenchmark(
        api, project.id, output_dir=g.STORAGE_DIR + "/benchmark", progress=pbar
    )
    sly.logger.info(f"{session_id = }")
    bm.run_evaluation(model_session=session_id)

    session_info = api.task.get_info_by_id(session_id)
    task_dir = f"{session_id}_{session_info['meta']['app']['name']}"
    eval_res_dir = f"/model-benchmark/evaluation/{project.id}_{project.name}/{task_dir}/"
    eval_res_dir = api.storage.get_free_dir_name(g.team_id, eval_res_dir)

    bm.upload_eval_results(eval_res_dir)

    bm.visualize()
    remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")

    report = bm.upload_report_link(remote_dir)
    api.task.set_output_report(g.task_id, report.id, report.name)

    creating_report_f.hide()

    template_vis_file = api.file.get_info_by_path(
        sly.env.team_id(), eval_res_dir + "/visualizations/template.vue"
    )
    report_model_benchmark.set(template_vis_file)
    report_model_benchmark.show()
    pbar.hide()

    g.workflow.add_input(session_id)
    g.workflow.add_input(project)
    g.workflow.add_output(eval_res_dir)
    g.workflow.add_output_report(template_vis_file)

    sly.logger.info(
        f"Predictions project name {bm.dt_project_info.name}, workspace_id {bm.dt_project_info.workspace_id}"
    )
    sly.logger.info(
        f"Differences project name {bm.diff_project_info.name}, workspace_id {bm.diff_project_info.workspace_id}"
    )

    button.loading = False
    app.stop()


sel_app_session = w.SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = w.SelectProject(default_id=None, workspace_id=g.workspace_id)
button = w.Button("Evaluate")
pbar = w.SlyTqdm()
report_model_benchmark = w.ReportThumbnail()
report_model_benchmark.hide()
creating_report_f = w.Field(w.Empty(), "", "Creating report on model...")
creating_report_f.hide()

layout = w.Container(
    widgets=[
        w.Text("Select GT Project"),
        sel_project,
        sel_app_session,
        button,
        creating_report_f,
        report_model_benchmark,
        pbar,
    ]
)


@sel_project.value_changed
def handle(project_id: Optional[int]):
    active = project_id is not None and sel_app_session.get_selected_id() is not None
    if active:
        button.enable()
    else:
        button.disable()


@sel_app_session.value_changed
def handle(session_id: Optional[int]):
    active = session_id is not None and sel_project.get_selected_id() is not None
    if active:
        button.enable()
    else:
        button.disable()


@button.click
def handle():
    creating_report_f.show()
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
