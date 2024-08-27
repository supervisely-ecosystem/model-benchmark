from typing import Optional

import src.functions as f
import src.globals as g
import src.workflow as w
import supervisely as sly
import supervisely.app.widgets as widgets
from supervisely.nn.benchmark import ObjectDetectionBenchmark


def main_func():
    api = g.api
    project = api.project.get_info_by_id(g.project_id)

    # ==================== Workflow input ====================
    w.workflow_input(api, project, g.session_id)
    # =======================================================

    pbar.show()
    report_model_benchmark.hide()

    selected_classes = select_classes.get_selected_classes()
    bm = ObjectDetectionBenchmark(
        api,
        project.id,
        output_dir=g.STORAGE_DIR + "/benchmark",
        progress=pbar,
        classes=selected_classes,
    )
    sly.logger.info(f"{g.session_id = }")
    bm.run_evaluation(model_session=g.session_id)

    session_info = api.task.get_info_by_id(g.session_id)
    task_dir = f"{g.session_id}_{session_info['meta']['app']['name']}"
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
        f"Predictions project: "
        f"  name {bm.dt_project_info.name}, "
        f"  workspace_id {bm.dt_project_info.workspace_id}. "
        f"Differences project: "
        f"  name {bm.diff_project_info.name}, "
        f"  workspace_id {bm.diff_project_info.workspace_id}"
    )

    button.loading = False
    app.stop()


select_classes = widgets.ClassesListSelector(multiple=True)
select_classes.hide()
no_classes_label = widgets.Text(
    "Not found any classes in the project that are present in the model",
    status="error",
)
no_classes_label.hide()

sel_app_session = widgets.SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = widgets.SelectProject(default_id=None, workspace_id=g.workspace_id)

button = widgets.Button("Evaluate")
button.disable()

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
        select_classes,
        no_classes_label,
        creating_report_f,
        report_model_benchmark,
        pbar,
    ],
)


def handle_selectors(active: bool):
    no_classes_label.hide()
    select_classes.hide()
    button.loading = True
    if active:
        classes = f.get_classes()
        if len(classes) > 0:
            select_classes.set(classes)
            select_classes.show()
            select_classes.select_all()
            button.loading = False
            button.enable()
            return
        else:
            no_classes_label.show()
    button.loading = False
    button.disable()


@sel_project.value_changed
def handle_sel_project(project_id: Optional[int]):
    g.project_id = project_id
    active = project_id is not None and g.session_id is not None
    handle_selectors(active)


@sel_app_session.value_changed
def handle_sel_app_session(session_id: Optional[int]):
    g.session_id = session_id
    active = session_id is not None and g.project_id is not None
    handle_selectors(active)


@button.click
def start_evaluation():
    creating_report_f.show()
    select_classes.disable()
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)

if g.project_id:
    sel_project.set_project_id(g.project_id)

if g.session_id:
    sel_app_session.set_session_id(g.session_id)

if g.autostart:
    start_evaluation()