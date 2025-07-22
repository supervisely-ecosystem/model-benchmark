import supervisely as sly
import supervisely.app.widgets as widgets
from fastapi import Request

import src.globals as g
from src.ui.compare import compare_button, compare_contatiner, run_compare
from src.ui.evaluation import eval_button, evaluation_container, run_evaluation

tabs = widgets.Tabs(
    labels=["Model Evaluation", "Model Comparison"],
    contents=[evaluation_container, compare_contatiner],
)
tabs_card = widgets.Card(
    title="Model Benchmark",
    content=tabs,
    description="Select the task you want to perform",
)

layout = widgets.Container(
    widgets=[tabs_card, widgets.Empty()],
    direction="horizontal",
    fractions=[1, 1],
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
async def evaluate(request: Request):
    req = await request.json()
    try:
        state = req["state"]
        session_id = state["session_id"]
        project_id = state["project_id"]
        dataset_ids = state.get("dataset_ids", None)
        collection_id = state.get("collection_id", None)
        return {
            "data": run_evaluation(
                session_id, project_id, dataset_ids=dataset_ids, collection_id=collection_id
            )
        }
    except Exception as e:
        sly.logger.error(f"Error during model evaluation: {e}")
        return {"error": str(e)}


@server.post("/run_comparison")
async def compare(request: Request):
    req = await request.json()
    try:
        state = req["state"]
        return {"data": run_compare(state["eval_dirs"])}
    except Exception as e:
        sly.logger.error(f"Error during model comparison: {e}")
        return {"error": str(e)}


@server.post("/get_eval_progress")
async def get_eval_progress(request: Request):
    req = await request.json()
    state = req.get("state", {})
    session_id = state.get("session_id", None)
    if session_id is None:
        return {"error": "Session ID is required"}

    if session_id not in g.eval_progress:
        return {"error": "No evaluation progress found for the given session ID"}

    progress = g.eval_progress.get(session_id, {"status": "unknown", "current": 0, "total": 0})
    return {"data": progress}
