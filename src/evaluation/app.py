import supervisely as sly
from supervisely.app import widgets as W
from src.evaluation.run_evaluation import evaluate
from src.evaluation.run_speedtest import run_speedtest


api = sly.Api()
team_id = sly.env.team_id()


if sly.is_development():
    team_id = sly.env.team_id()
    workspace_id = 1076
    gt_project_id = 39099  # COCO 2017 val
    model_session_id = 2

gt_selector = W.SelectDataset(
    project_id=gt_project_id,
    multiselect=True,
    select_all_datasets=True,
    allowed_project_types=[sly.ProjectType.IMAGES],
    )

model_selector = W.SelectAppSession(
    team_id,
    ["deployed_nn"],
    )

run_eval_check = W.Checkbox("Run evaluation", True)
run_speedtest_check = W.Checkbox("Run speed test", True)
run_button = W.Button("Run")


@run_button.click
def run():
    gt_project_id = gt_selector.get_selected_project_id()
    gt_dataset_ids = gt_selector.get_selected_ids()
    model_session_id = model_selector.get_selected_id()
    if run_eval_check.is_checked():
        evaluate(
            api,
            gt_project_id,
            model_session_id,
            gt_dataset_ids=gt_dataset_ids,
            )
    if run_speedtest_check.is_checked():
        run_speedtest(api, gt_project_id, model_session_id)


# Layout
content = W.Container([
        W.Field(gt_selector, "Ground Truth Project"),
        W.Field(model_selector, "Model session"),
        run_eval_check,
        run_speedtest_check,
        run_button,
])

layout = W.Card("Run Evaluation", content=content)

app = sly.Application(layout=layout)
