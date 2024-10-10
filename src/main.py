from typing import Optional

import yaml

import src.functions as f
import src.globals as g
import src.workflow as w
import supervisely as sly
import supervisely.app.widgets as widgets
from supervisely.nn import TaskType
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.benchmark.evaluation.base_evaluator import BaseEvaluator
from supervisely.nn.benchmark.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)
from supervisely.nn.benchmark.evaluation.object_detection_evaluator import (
    ObjectDetectionEvaluator,
)
from supervisely.nn.inference.session import SessionJSON


def main_func():
    api = g.api
    project = api.project.get_info_by_id(g.project_id)
    if g.session is None:
        g.session = SessionJSON(api, g.session_id)
    task_type = g.session.get_deploy_info()["task_type"]

    # ==================== Workflow input ====================
    w.workflow_input(api, project, g.session_id)
    # =======================================================

    report_model_benchmark.hide()

    set_selected_classes_and_show_info()
    if g.selected_classes is None or len(g.selected_classes) == 0:
        return

    pbar.show()
    sec_pbar.show()
    evaluation_parameters = yaml.safe_load(eval_params.get_value())
    if task_type == "object detection":
        bm = ObjectDetectionBenchmark(
            api,
            project.id,
            output_dir=g.STORAGE_DIR + "/benchmark",
            progress=pbar,
            progress_secondary=sec_pbar,
            classes_whitelist=g.selected_classes,
            evaluation_params=evaluation_parameters,
        )
    elif task_type == "instance segmentation":
        bm = InstanceSegmentationBenchmark(
            api,
            project.id,
            output_dir=g.STORAGE_DIR + "/benchmark",
            progress=pbar,
            progress_secondary=sec_pbar,
            classes_whitelist=g.selected_classes,
            evaluation_params=evaluation_parameters,
        )
    sly.logger.info(f"{g.session_id = }")

    task_info = api.task.get_info_by_id(g.session_id)
    task_dir = f"{g.session_id}_{task_info['meta']['app']['name']}"

    res_dir = f"/model-benchmark/{project.id}_{project.name}/{task_dir}/"
    res_dir = api.storage.get_free_dir_name(g.team_id, res_dir)

    bm.run_evaluation(model_session=g.session_id)

    try:
        bm.run_speedtest(g.session_id, g.project_id)
        sec_pbar.hide()
        bm.upload_speedtest_results(res_dir + "/speedtest/")
    except Exception as e:
        sly.logger.warn(f"Speedtest failed. Skipping. {e}")

    bm.visualize()

    bm.upload_eval_results(res_dir + "/evaluation/")
    remote_dir = bm.upload_visualizations(res_dir + "/visualizations/")

    report = bm.upload_report_link(remote_dir)
    api.task.set_output_report(g.task_id, report.id, report.name)

    template_vis_file = api.file.get_info_by_path(
        sly.env.team_id(), res_dir + "/visualizations/template.vue"
    )
    report_model_benchmark.set(template_vis_file)
    report_model_benchmark.show()
    pbar.hide()

    # ==================== Workflow output ====================
    w.workflow_output(api, res_dir, template_vis_file)
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

eval_params = widgets.Editor(
    initial_text=None,
    language_mode="yaml",
    height_lines=16,
)
eval_params_card = widgets.Card(
    title="Evaluation parameters",
    content=eval_params,
    collapsable=True,
)
eval_params_card.collapse()


button = widgets.Button("Evaluate")
button.disable()

pbar = widgets.SlyTqdm()
sec_pbar = widgets.Progress("")

report_model_benchmark = widgets.ReportThumbnail()
report_model_benchmark.hide()

controls_card = widgets.Card(
    title="Settings",
    description="Select Ground Truth project and deployed model session",
    content=widgets.Container(
        [
            sel_project,
            sel_app_session,
            eval_params_card,
            button,
            report_model_benchmark,
            pbar,
            sec_pbar,
        ]
    ),
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


def update_eval_params():
    if g.session is None:
        g.session = SessionJSON(g.api, g.session_id)
    task_type = g.session.get_deploy_info()["task_type"]
    if task_type == TaskType.OBJECT_DETECTION:
        params = ObjectDetectionEvaluator.load_yaml_evaluation_params()
    elif task_type == TaskType.INSTANCE_SEGMENTATION:
        params = InstanceSegmentationEvaluator.load_yaml_evaluation_params()
    eval_params.set_text(params, language_mode="yaml")
    eval_params_card.uncollapse()


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

    if g.session_id:
        update_eval_params()


@button.click
def start_evaluation():
    main_func()


app = sly.Application(layout=main_layout, static_dir=g.STATIC_DIR)

if g.project_id:
    sel_project.set_project_id(g.project_id)

if g.session_id:
    sel_app_session.set_session_id(g.session_id)
    update_eval_params()

if g.autostart:
    start_evaluation()

if g.project_id and g.session_id:
    handle_selectors(True)
