from typing import Dict, Optional, Union

import yaml

import src.functions as f
import src.globals as g
import src.workflow as w
import supervisely as sly
import supervisely.app.widgets as widgets
from supervisely._utils import rand_str
from supervisely.nn import TaskType
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    ObjectDetectionBenchmark,
)
from supervisely.nn.benchmark.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)
from supervisely.nn.benchmark.evaluation.object_detection_evaluator import (
    ObjectDetectionEvaluator,
)
from supervisely.nn.inference.session import SessionJSON


def run_evaluation(
    session_id: Optional[int] = None,
    project_id: Optional[int] = None,
    params: Optional[Union[str, Dict]] = None,
):
    work_dir = g.STORAGE_DIR + "/benchmark_" + rand_str(6)

    if session_id is not None:
        g.session_id = session_id
    if project_id is not None:
        g.project_id = project_id

    project = g.api.project.get_info_by_id(g.project_id)
    if g.session is None:
        g.session = SessionJSON(g.api, g.session_id)
    task_type = g.session.get_deploy_info()["task_type"]

    # ==================== Workflow input ====================
    w.workflow_input(g.api, project, g.session_id)
    # =======================================================

    report_model_benchmark.hide()

    set_selected_classes_and_show_info()
    if g.selected_classes is None or len(g.selected_classes) == 0:
        return

    pbar.show()
    sec_pbar.show()

    evaluation_params = eval_params.get_value() or params
    if isinstance(evaluation_params, str):
        evaluation_params = yaml.safe_load(evaluation_params)

    if task_type == TaskType.OBJECT_DETECTION:
        if evaluation_params is None:
            evaluation_params = ObjectDetectionEvaluator.load_yaml_evaluation_params()
            evaluation_params = yaml.safe_load(evaluation_params)
        bm = ObjectDetectionBenchmark(
            g.api,
            project.id,
            output_dir=work_dir,
            progress=pbar,
            progress_secondary=sec_pbar,
            classes_whitelist=g.selected_classes,
            evaluation_params=evaluation_params,
        )
    elif task_type == TaskType.INSTANCE_SEGMENTATION:
        if evaluation_params is None:
            evaluation_params = InstanceSegmentationEvaluator.load_yaml_evaluation_params()
            evaluation_params = yaml.safe_load(evaluation_params)
        bm = InstanceSegmentationBenchmark(
            g.api,
            project.id,
            output_dir=work_dir,
            progress=pbar,
            progress_secondary=sec_pbar,
            classes_whitelist=g.selected_classes,
            evaluation_params=evaluation_params,
        )
    sly.logger.info(f"{g.session_id = }")

    task_info = g.api.task.get_info_by_id(g.session_id)
    task_dir = f"{g.session_id}_{task_info['meta']['app']['name']}"

    res_dir = f"/model-benchmark/{project.id}_{project.name}/{task_dir}/"
    res_dir = g.api.storage.get_free_dir_name(g.team_id, res_dir)

    session_info = g.session.get_session_info()
    support_batch_inference = session_info.get("batch_inference_support", False)
    max_batch_size = session_info.get("max_batch_size")
    batch_size = 16
    if not support_batch_inference:
        batch_size = 1
    if max_batch_size is not None:
        batch_size = min(max_batch_size, 16)
    bm.run_evaluation(model_session=g.session_id, batch_size=batch_size)

    try:
        batch_sizes = (1, 8, 16)
        if not support_batch_inference:
            batch_sizes = (1,)
        elif max_batch_size is not None:
            batch_sizes = tuple([bs for bs in batch_sizes if bs <= max_batch_size])
        bm.run_speedtest(g.session_id, g.project_id, batch_sizes=batch_sizes)
        sec_pbar.hide()
        bm.upload_speedtest_results(res_dir + "/speedtest/")
    except Exception as e:
        sly.logger.warning(f"Speedtest failed. Skipping. {e}")

    bm.visualize()

    bm.upload_eval_results(res_dir + "/evaluation/")
    remote_dir = bm.upload_visualizations(res_dir + "/visualizations/")

    report = bm.upload_report_link(remote_dir)
    g.api.task.set_output_report(g.task_id, report.id, report.name)

    template_vis_file = g.api.file.get_info_by_path(
        sly.env.team_id(), res_dir + "/visualizations/template.vue"
    )
    report_model_benchmark.set(template_vis_file)
    report_model_benchmark.show()
    pbar.hide()

    # ==================== Workflow output ====================
    w.workflow_output(g.api, res_dir, template_vis_file)
    # =======================================================

    sly.logger.info(
        f"Predictions project: "
        f"  name {bm.dt_project_info.name}, "
        f"  workspace_id {bm.dt_project_info.workspace_id}. "
        f"Differences project: "
        f"  name {bm.diff_project_info.name}, "
        f"  workspace_id {bm.diff_project_info.workspace_id}"
    )

    eval_button.loading = False

    return res_dir


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


eval_button = widgets.Button("Evaluate")
eval_button.disable()

pbar = widgets.SlyTqdm()
sec_pbar = widgets.Progress("")

report_model_benchmark = widgets.ReportThumbnail()
report_model_benchmark.hide()

evaluation_container = widgets.Container(
    [
        sel_project,
        sel_app_session,
        eval_params_card,
        eval_button,
        report_model_benchmark,
        pbar,
        sec_pbar,
        total_classes_text,
        selected_matched_text,
        not_matched_text,
        no_classes_label,
    ]
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
        eval_button.enable()
    else:
        eval_button.disable()


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
