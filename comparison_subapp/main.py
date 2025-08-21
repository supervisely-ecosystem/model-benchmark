import ast
import os

import supervisely as sly
from src.functions import check_for_existing_comparisons
from src.ui.compare import run_compare
from supervisely import handle_exceptions, main_wrapper
from supervisely.nn.benchmark.comparison.base_visualizer import BaseComparisonVisualizer

api = sly.Api()
REPORT_FILENAME = BaseComparisonVisualizer.report_name
TEAM_ID = sly.env.team_id()
EVAL_DIRS = os.environ.get("modal.state.eval_dirs", None)
if EVAL_DIRS is None:
    raise RuntimeError("Environment variable 'modal.state.eval_dirs' is not set.")
EVAL_DIRS = [str(x).strip() for x in ast.literal_eval(EVAL_DIRS)]


@handle_exceptions(has_ui=False)
def run_state_evaluation():
    result_comparison_dir = check_for_existing_comparisons(EVAL_DIRS, TEAM_ID)
    if result_comparison_dir is not None:
        fileinfo = api.file.get_info_by_path(TEAM_ID, result_comparison_dir + REPORT_FILENAME)
        if fileinfo is None:
            raise ValueError("Comparison link ID not found in the storage.")
        sly.logger.info(f"Comparison already exists: {result_comparison_dir} (ID: {fileinfo.id})")
        api.task.set_output_report(
            api.task_id, fileinfo.id, REPORT_FILENAME, "Click to open the report"
        )
    else:
        _ = run_compare(EVAL_DIRS)


main_wrapper("comparison_subapp", run_state_evaluation)
