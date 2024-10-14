from fastapi import Request

import src.globals as g
import supervisely as sly
import supervisely.app.widgets as widgets
from src.ui.compare import compare_button, compare_contatiner, run_compare
from src.ui.evaluation import eval_button, evaluation_container, run_evaluation

tabs = widgets.Tabs(
    labels=["Evaluate models", "Compare models"],
    contents=[evaluation_container, compare_contatiner],
)
tabs_card = widgets.Card(
    title="Model evaluation",
    content=tabs,
    description="Select the task you want to perform",
)

layout = widgets.Container(
    widgets=[tabs_card, widgets.Empty(), widgets.Empty()],
    direction="horizontal",
    fractions=[1, 1, 1],
)

app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
server = app.get_server()


@eval_button.click
def start_evaluation():
    run_evaluation()


@compare_button.click
def start_comparison():
    run_compare()


@server.post("/run_evaluation")
async def inference_pointcloud_ids(request: Request):
    req = await request.json()
    run_evaluation(req["session_id"], req["project_id"])


@server.post("/run_comparison")
async def inference_pointcloud_ids(request: Request):
    req = await request.json()
    run_compare(req["eval_dirs"])
