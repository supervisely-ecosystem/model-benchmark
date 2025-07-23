import asyncio
import json

import supervisely as sly
import supervisely.app.widgets as widgets
from fastapi import Request, WebSocket, WebSocketDisconnect

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
        # run evaluation in a threadpool so the WS loop keeps processing
        res_dir = await asyncio.to_thread(
            run_evaluation,
            session_id,
            project_id,
            dataset_ids=dataset_ids,
            collection_id=collection_id,
        )
        return {"data": res_dir}
    except Exception as e:
        sly.logger.error(f"Error during model evaluation: {e}")
        return {"error": str(e)}


@server.post("/run_comparison")
async def compare(request: Request):
    req = await request.json()
    try:
        state = req["state"]
        # run comparison in a threadpool so the WS loop keeps processing
        res_dir = await asyncio.to_thread(run_compare, state["eval_dirs"])
        return {"data": res_dir}
    except Exception as e:
        sly.logger.error(f"Error during model comparison: {e}")
        return {"error": str(e)}


@server.websocket("/progress_monitor")
async def progress_monitor(websocket: WebSocket, session_id: int):
    await g.pmm.add_connection(session_id, websocket)
    sly.logger.info(f"WebSocket connected for session {session_id}")
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                sly.logger.info(f"WebSocket disconnected for session {session_id}")
                break
    finally:
        g.pmm.disconnect(session_id)
