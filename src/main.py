from typing import Optional

import src.globals as g
import src.workflow as w
import supervisely as sly
import supervisely.app.widgets as widgets
from supervisely.nn.benchmark import ObjectDetectionBenchmark


def main_func():
    api = g.api
    project = api.project.get_info_by_id(sel_project.get_selected_id())
    session_id = sel_app_session.get_selected_id()

    # ==================== Workflow input ====================
    w.workflow_input(api, project, session_id)
    # =======================================================

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

    # ==================== Workflow output ====================
    w.workflow_output(api, eval_res_dir, template_vis_file)
    # =======================================================

    sly.logger.info(
        f"Predictions project name {bm.dt_project_info.name}, workspace_id {bm.dt_project_info.workspace_id}"
    )
    sly.logger.info(
        f"Differences project name {bm.diff_project_info.name}, workspace_id {bm.diff_project_info.workspace_id}"
    )

    button.loading = False
    app.stop()


sel_app_session = widgets.SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = widgets.SelectProject(default_id=None, workspace_id=g.workspace_id)
button = widgets.Button("Evaluate")
pbar = widgets.SlyTqdm()
report_model_benchmark = widgets.ReportThumbnail()
report_model_benchmark.hide()
creating_report_f = widgets.Field(widgets.Empty(), "", "Creating report on model...")
creating_report_f.hide()

layout = widgets.Container(
    widgets=[
        widgets.Text("Select GT Project"),
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
def start_evaluation():
    creating_report_f.show()
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)

if g.project_id:
    sel_project.set_project_id(g.project_id)

if g.session_id:
    sel_app_session.set_session_id(g.session_id)

if g.autostart:
    start_evaluation()