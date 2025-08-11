import ast
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
    eval_dirs = [str(x).strip() for x in ast.literal_eval(eval_dirs)]

session = None

model_classes = None
task_type = None
project_classes = None
selected_classes = None
