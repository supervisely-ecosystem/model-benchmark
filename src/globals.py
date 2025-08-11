import os

import supervisely as sly
from dotenv import load_dotenv

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
eval_dirs = os.environ.get("modal.state.eval_dirs", None)
if eval_dirs is not None:
    import ast

    from src.functions import check_for_existing_comparisons
    from src.ui.compare import run_compare

    try:
        eval_dirs = [str(x).strip() for x in ast.literal_eval(eval_dirs)]

        if not project_id:
            raise ValueError("Project ID is not set. Please set the project ID in the environment.")

        result_comparison_dir = check_for_existing_comparisons(eval_dirs, project_id, team_id)
        if result_comparison_dir is not None:
            comparison_link_id = next(
                api.storage.list(
                    team_id, result_comparison_dir + "Model Comparison Report.lnk", recursive=False
                ),
                None,
            )
            if comparison_link_id is None:
                raise ValueError("Comparison link ID not found in the storage.")
            comparison_link_id = comparison_link_id.id
            sly.logger.info(
                f"Comparison already exists: {result_comparison_dir}. Using existing comparison link ID: {comparison_link_id}"
            )
            api.task.set_output_report(
                task_id,
                comparison_link_id,
                "Model Comparison Report",
                "Click to open the report",
            )
        else:
            _ = run_compare(eval_dirs)
    except Exception as e:
        sly.logger.error(f"Error during model comparison: {e}")

session = None

model_classes = None
task_type = None
project_classes = None
selected_classes = None
eval_dirs = None
