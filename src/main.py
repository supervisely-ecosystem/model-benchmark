from typing import Optional

import supervisely as sly
import supervisely.app.widgets as widgets
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.inference.session import SessionJSON

import src.functions as f
import src.globals as g
import src.workflow as w


def main_func():
    api = g.api
    project = api.project.get_info_by_id(g.project_id)
    if g.session is None:
        g.session = SessionJSON(api, g.session_id)
    task_type = g.session.get_deploy_info()["task_type"]

    # ==================== Workflow input ====================
    w.workflow_input(api, project, g.session_id)
    # =======================================================

    pbar.show()
    report_model_benchmark.hide()

    set_selected_classes_and_show_info()
    if task_type == "object detection":
        bm = ObjectDetectionBenchmark(
            api,
            project.id,
            output_dir=g.STORAGE_DIR + "/benchmark",
            progress=pbar,
            classes_whitelist=g.selected_classes,
        )
    elif task_type == "instance segmentation":
        bm = InstanceSegmentationBenchmark(
            api,
            project.id,
            output_dir=g.STORAGE_DIR + "/benchmark",
            progress=pbar,
            classes_whitelist=g.selected_classes,
        )
    sly.logger.info(f"{g.session_id = }")
    bm.run_evaluation(model_session=g.session_id)

    task_info = api.task.get_info_by_id(g.session_id)
    task_dir = f"{g.session_id}_{task_info['meta']['app']['name']}"

    eval_res_dir = f"/model-benchmark/evaluation/{project.id}_{project.name}/{task_dir}/"
    eval_res_dir = api.storage.get_free_dir_name(g.team_id, eval_res_dir)

    bm.upload_eval_results(eval_res_dir)

    bm.visualize()
    remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")

    report = bm.upload_report_link(remote_dir)
    api.task.set_output_report(g.task_id, report.id, report.name)

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


no_classes_label = widgets.Text(
    "Not found any classes in the project that are present in the model", status="error"
)
no_classes_label.hide()
total_classes_text = widgets.Text(status="info")
selected_matched_text = widgets.Text(status="success")
not_matched_text = widgets.Text(status="warning")

sel_app_session = widgets.SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = widgets.SelectProject(default_id=None, workspace_id=g.workspace_id)

button = widgets.Button("Evaluate")
button.disable()

pbar = widgets.SlyTqdm()

report_model_benchmark = widgets.ReportThumbnail()
report_model_benchmark.hide()

controls_card = widgets.Card(
    title="Settings",
    description="Select Ground Truth project and deployed model session",
    content=widgets.Container([sel_project, sel_app_session, button, report_model_benchmark, pbar]),
)


layout = widgets.Container(
    widgets=[controls_card, widgets.Empty(), widgets.Empty()],  # , matched_card, not_matched_card],
    direction="horizontal",
    fractions=[1, 1, 1],
)

main_layout = widgets.Container(
    widgets=[layout, total_classes_text, selected_matched_text, not_matched_text, no_classes_label]
)


def set_selected_classes_and_show_info():
    matched, not_matched = f.get_classes()
    _, matched_model_classes = matched
    _, not_matched_model_classes = not_matched
    total_classes_text.text = (
        f"{len(matched_model_classes) + len(not_matched_model_classes)} classes found in the model."
    )
    selected_matched_text.text = f"{len(matched_model_classes)} classes can be used for evaluation."
    not_matched_text.text = f"{len(not_matched_model_classes)} classes are not available for evaluation (not found in the GT project or have different geometry type)."
    if len(matched_model_classes) > 0:
        g.selected_classes = [obj_cls.name for obj_cls in matched_model_classes]
        selected_matched_text.show()
        if len(not_matched_model_classes) > 0:
            not_matched_text.show()
    else:
        no_classes_label.show()


def handle_selectors(active: bool):
    no_classes_label.hide()
    selected_matched_text.hide()
    not_matched_text.hide()
    if active:
        button.enable()
    else:
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
    # select_classes.hide()
    # not_matched_classes.hide()
    main_func()


app = sly.Application(layout=main_layout, static_dir=g.STATIC_DIR)

if g.project_id:
    sel_project.set_project_id(g.project_id)

if g.session_id:
    sel_app_session.set_session_id(g.session_id)

if g.autostart:
    start_evaluation()

if g.project_id and g.session_id:
    handle_selectors(True)