from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import supervisely as sly
import yaml
from supervisely.api.entities_collection_api import CollectionTypeFilter
from supervisely.app.widgets import (
    Button,
    Card,
    Checkbox,
    Container,
    Editor,
    Progress,
    ReportThumbnail,
    SelectAppSession,
    SelectDataset,
    SelectProject,
    SlyTqdm,
    Text,
)
from supervisely.nn.benchmark import (
    InstanceSegmentationBenchmark,
    InstanceSegmentationEvaluator,
    ObjectDetectionBenchmark,
    ObjectDetectionEvaluator,
    SemanticSegmentationBenchmark,
    SemanticSegmentationEvaluator,
)
from supervisely.nn.inference.session import SessionJSON

import src.functions as f
import src.globals as g
import src.workflow as w

no_classes_label = Text(
    "Not found any classes in the project that are present in the model", status="error"
)
no_classes_label.hide()
total_classes_text = Text(status="info")
selected_matched_text = Text(status="success")
not_matched_text = Text(status="warning")

sel_app_session = SelectAppSession(g.team_id, tags=g.deployed_nn_tags, show_label=True)
sel_project = SelectProject(
    default_id=None,
    workspace_id=g.workspace_id,
    allowed_types=[sly.ProjectType.IMAGES],
    compact=True,
)
sel_dataset = SelectDataset(multiselect=True, compact=True)
sel_dataset.hide()
all_datasets_checkbox = Checkbox("All datasets", checked=True)
run_speedtest_checkbox = Checkbox("Run speedtest", checked=True)

iou_per_class_checkbox = Checkbox("Set different IoU for every class", checked=False)
eval_params = Editor(
    initial_text=None,
    language_mode="yaml",
    # height_lines=25,
    height_px=200,
)
eval_params_card = Card(
    title="Evaluation parameters",
    content=Container([eval_params, iou_per_class_checkbox]),
    collapsable=True,
)
eval_params_card.collapse()


eval_button = Button("Evaluate")
eval_button.disable()

eval_pbar = SlyTqdm()
sec_eval_pbar = Progress("")

report_model_benchmark = ReportThumbnail()
report_model_benchmark.hide()

evaluation_container = Container(
    [
        sel_project,
        all_datasets_checkbox,
        sel_dataset,
        sel_app_session,
        eval_params_card,
        run_speedtest_checkbox,
        eval_button,
        report_model_benchmark,
        eval_pbar,
        sec_eval_pbar,
        total_classes_text,
        selected_matched_text,
        not_matched_text,
        no_classes_label,
    ]
)

benchmark_cls_type = Union[
    ObjectDetectionBenchmark, InstanceSegmentationBenchmark, SemanticSegmentationBenchmark
]

evaluator_cls_type = Union[
    ObjectDetectionEvaluator, InstanceSegmentationEvaluator, SemanticSegmentationEvaluator
]


def get_benchmark_and_evaluator_classes(
    task_type: sly.nn.TaskType,
) -> Tuple[benchmark_cls_type, evaluator_cls_type]:
    if task_type == sly.nn.TaskType.OBJECT_DETECTION:
        return ObjectDetectionBenchmark, ObjectDetectionEvaluator
    elif task_type == sly.nn.TaskType.INSTANCE_SEGMENTATION:
        return (InstanceSegmentationBenchmark, InstanceSegmentationEvaluator)
    elif task_type == sly.nn.TaskType.SEMANTIC_SEGMENTATION:
        return (SemanticSegmentationBenchmark, SemanticSegmentationEvaluator)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


@f.with_clean_up_progress(eval_pbar)
def run_evaluation(
    session_id: Optional[int] = None,
    project_id: Optional[int] = None,
    params: Optional[Union[str, Dict]] = None,
    dataset_ids: Optional[Tuple[int]] = None,
    collection_id: Optional[int] = None,
):
    work_dir = g.STORAGE_DIR + "/benchmark_" + sly.rand_str(6)

    g.session_id = session_id or g.session_id
    g.project_id = project_id or g.project_id

    project = g.api.project.get_info_by_id(g.project_id)
    if g.project_classes is None:
        g.project_classes = f.get_project_classes()

    if g.session is None:
        g.session = SessionJSON(g.api, g.session_id)

    model_classes, task_type = f.get_model_info()
    if g.task_type is None:
        g.task_type = task_type
    if g.model_classes is None:
        g.model_classes = model_classes

    if dataset_ids is None:
        if all_datasets_checkbox.is_checked():
            dataset_ids = None
        else:
            dataset_ids = sel_dataset.get_selected_ids()
            if len(dataset_ids) == 0:
                raise ValueError("No datasets selected")

    image_ids = None
    if collection_id is not None:
        imageinfos = g.api.entities_collection.get_items(
            collection_id, CollectionTypeFilter.DEFAULT
        )
        image_ids = [imageinfo.id for imageinfo in imageinfos]

    if dataset_ids is not None or collection_id is not None:
        if not bool(dataset_ids) ^ bool(collection_id):
            raise ValueError(
                "You must select either datasets or a collection, but not both at the same time."
            )

    # ==================== Workflow input ====================
    if sly.is_production():
        w.workflow_input(g.api, project, g.session_id)
    # =======================================================

    report_model_benchmark.hide()

    if g.selected_classes is None or len(g.selected_classes) == 0:
        # raise RuntimeError("No classes available for evaluation")
        matched, _ = f.get_classes()
        _, matched_model_classes = matched
        if not matched_model_classes or len(matched_model_classes) == 0:
            raise RuntimeError("No classes available for evaluation.")
        g.selected_classes = [obj_cls.name for obj_cls in matched_model_classes]

    eval_pbar.show()
    sec_eval_pbar.show()

    params = eval_params.get_value() or params
    if isinstance(params, str):
        params = yaml.safe_load(params)

    bm_cls, evaluator_cls = get_benchmark_and_evaluator_classes(g.task_type)
    if params is None:
        params = evaluator_cls.load_yaml_evaluation_params()
        params = yaml.safe_load(params)
    bm: benchmark_cls_type = bm_cls(
        g.api,
        project.id,
        gt_dataset_ids=dataset_ids,
        gt_images_ids=image_ids,
        output_dir=work_dir,
        progress=eval_pbar,
        progress_secondary=sec_eval_pbar,
        classes_whitelist=g.selected_classes,
        evaluation_params=params,
    )

    bm.evaluator_app_info = g.api.task.get_info_by_id(g.task_id)
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

    if run_speedtest_checkbox.is_checked():
        try:
            batch_sizes = (1, 8, 16)
            if not support_batch_inference:
                batch_sizes = (1,)
            elif max_batch_size is not None:
                batch_sizes = tuple([bs for bs in batch_sizes if bs <= max_batch_size])
            bm.run_speedtest(g.session_id, g.project_id, batch_sizes=batch_sizes)
            sec_eval_pbar.hide()
            bm.upload_speedtest_results(res_dir + "/speedtest/")
        except Exception as e:
            sly.logger.warning(f"Speedtest failed. Skipping. {e}")

    bm.visualize()

    bm.upload_eval_results(res_dir + "/evaluation/")
    bm.upload_visualizations(res_dir + "/visualizations/")

    g.api.task.set_output_report(g.task_id, bm.lnk.id, bm.lnk.name, "Click to open the report")
    report_model_benchmark.set(bm.report)
    report_model_benchmark.show()
    eval_pbar.hide()

    # ==================== Workflow output ====================
    if sly.is_production():
        w.workflow_output(g.api, res_dir, bm.report)
    # =======================================================

    sly.logger.info(
        f"Predictions project {bm.dt_project_info.name}, workspace ID: {bm.dt_project_info.workspace_id}."
    )
    sly.logger.info(f"Report link: {bm.get_report_link()}")

    eval_button.loading = False

    return res_dir


@sel_project.value_changed
def handle_sel_project(project_id: Optional[int]):
    g.project_id = project_id
    handle_selectors()


@sel_app_session.value_changed
def handle_sel_app_session(session_id: Optional[int]):
    g.session_id = session_id
    handle_selectors()


@all_datasets_checkbox.value_changed
def handle_all_datasets_checkbox(checked: bool):
    if checked:
        sel_dataset.hide()
    else:
        sel_dataset.show()


@iou_per_class_checkbox.value_changed
def handle_iou_per_class_checkbox(checked: bool):
    update_eval_params()


def match_classes_and_show_info():
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
        g.selected_classes = None
        no_classes_label.show()


def update_eval_params():
    if g.session_id is None:
        return
    elif g.task_type == sly.nn.TaskType.OBJECT_DETECTION:
        params = ObjectDetectionEvaluator.load_yaml_evaluation_params()
    elif g.task_type == sly.nn.TaskType.INSTANCE_SEGMENTATION:
        params = InstanceSegmentationEvaluator.load_yaml_evaluation_params()
    elif g.task_type == sly.nn.TaskType.SEMANTIC_SEGMENTATION:
        params = ""
    params = set_iou_per_class_if_needed(params, iou_per_class_checkbox.is_checked())
    eval_params.set_text(params, language_mode="yaml")

    if g.task_type == sly.nn.TaskType.SEMANTIC_SEGMENTATION:
        eval_params_card.hide()
    else:
        eval_params_card.show()
        eval_params_card.uncollapse()


def handle_selectors():
    eval_button.loading = True
    no_classes_label.hide()
    selected_matched_text.hide()
    not_matched_text.hide()
    if g.project_id is not None:
        sel_dataset.set_project_id(g.project_id)
        update_available_project_classes()
    if g.session_id is not None:
        update_available_model_classes_and_task_type()
    if g.project_id is not None and g.session_id is not None:
        eval_button.enable()
        match_classes_and_show_info()
    else:
        eval_button.disable()
    update_eval_params()
    eval_button.loading = False


def set_iou_per_class_if_needed(params: str, checked: bool) -> str:
    """
    Set different IoU for every class and update the evaluation parameters.
    Will replace the iou_threshold key-value pair with iou_threshold_per_class (by default, 0.5 for every class)
    """
    if checked:
        params = yaml.safe_load(params)
        params = deepcopy(params)

        if "iou_threshold" in params:
            iou_threshold = params.pop("iou_threshold")
            if g.selected_classes is not None:
                params["iou_threshold_per_class"] = {c: iou_threshold for c in g.selected_classes}
            else:
                params["iou_threshold_per_class"] = {c.name: iou_threshold for c in g.model_classes}
        params = yaml.dump(params)
    return params


def update_available_model_classes_and_task_type():
    g.model_classes, g.task_type = f.get_model_info()


def update_available_project_classes():
    g.project_classes = f.get_project_classes()
