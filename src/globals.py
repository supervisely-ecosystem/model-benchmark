import os
from distutils.util import strtobool

from dotenv import load_dotenv

import supervisely as sly

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
session = None

selected_classes = None
eval_dirs = None
