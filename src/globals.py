import os

import supervisely as sly
from dotenv import load_dotenv

from src.functions import check_for_existing_comparisons
from src.ui.compare import run_compare

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


api = sly.Api.from_env()

STORAGE_DIR = sly.app.get_data_dir()
STATIC_DIR = os.path.join(STORAGE_DIR, "static")
sly.fs.mkdir(STATIC_DIR)

deployed_nn_tags = ["deployed_nn"]

workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
team_id = sly.env.team_id()
task_id = sly.env.task_id(raise_not_found=False)
session_id = os.environ.get("modal.state.sessionId", None)
if session_id is not None:
    session_id = int(session_id)
eval_dirs = os.environ.get("modal.state.evalDirs", None)
if eval_dirs is not None:
    result_comparison_dir = check_for_existing_comparisons(eval_dirs, project_id, team_id)
    if result_comparison_dir is not None:
        comparison_link_id = api.file.get_info_by_path(
            team_id, result_comparison_dir + "/Model Comparison Report.lnk"
        ).id
        api.task.set_output_report(
            task_id,
            comparison_link_id,
            "Model Comparison Report",
            "Click to open the report",
        )
    else:
        result_comparison_dir = run_compare(eval_dirs)

session = None

model_classes = None
task_type = None
project_classes = None
selected_classes = None
eval_dirs = None
