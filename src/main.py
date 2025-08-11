import supervisely as sly
import supervisely.app.widgets as widgets
from fastapi import Request

import src.globals as g
from src.functions import check_for_existing_comparisons
from src.ui.compare import (
    compare_button,
    compare_contatiner,
    models_comparison_report,
    run_compare,
)
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

if g.eval_dirs is not None:
    if tabs.get_active_tab() != "Model Comparison":
        tabs.set_active_tab("Model Comparison")
    try:
        if not g.project_id:
            raise ValueError("Project ID is not set. Please set the project ID in the environment.")

        result_comparison_dir = check_for_existing_comparisons(g.eval_dirs, g.project_id, g.team_id)
        if result_comparison_dir is not None:
            fileinfo = g.api.file.get_info_by_path(
                g.team_id, result_comparison_dir + "Model Comparison Report.lnk"
            )
            if fileinfo is None:
                raise ValueError("Comparison link ID not found in the storage.")
            sly.logger.info(
                f"Comparison already exists: {result_comparison_dir} (ID: {fileinfo.id})"
            )
            g.api.task.set_output_report(
                g.task_id,
                fileinfo.id,
                "Model Comparison Report",
                "Click to open the report",
            )
            models_comparison_report.set(fileinfo)
            models_comparison_report.show()
        else:
            _ = run_compare(g.eval_dirs)
        app.stop()
    except Exception as e:
        sly.logger.error(f"Error during model comparison: {e}")


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
