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

dt_selector = W.SelectProject(
    workspace_id=workspace_id,
    allowed_types=[sly.ProjectType.IMAGES],
    )

model_selector = W.SelectAppSession(
    team_id,
    ["deployed_nn"],
    )

run_eval_check = W.Checkbox("Run evaluation", True)
run_speedtest_check = W.Checkbox("Run speed test", True)

path_selector = W.TeamFilesSelector(team_id, selection_file_type='folder')

eval_tab = W.Container([
        W.Field(gt_selector, "Ground Truth Project"),
        W.Field(dt_selector, "Predicted Project"),
        W.Field(model_selector, "Model session"),
        run_eval_check,
        run_speedtest_check,
])

load_tab = W.Container([
        W.Field(path_selector, "Eval results", "Path to the base dir of evaluation results"),
])

tabs = W.Tabs(
    ["Run evaluation", "Load Dashboard"],
    [eval_tab, load_tab],
    )


run_button = W.Button("Run")

@run_button.click
def run():
    if tabs.get_active_tab() == "Run evaluation":
        gt_project_id = gt_selector.get_selected_project_id()
        gt_dataset_ids = gt_selector.get_selected_ids()
        model_session_id = model_selector.get_selected_id()
        dt_project_info = None
        if run_eval_check.is_checked():
            dt_project_info = evaluate(
                api,
                gt_project_id,
                model_session_id,
                gt_dataset_ids=gt_dataset_ids,
                )
        if run_speedtest_check.is_checked():
            if dt_project_info is None:
                dt_project_id = dt_selector.get_selected_id()
            else:
                dt_project_id = dt_project_info.id
            benchmarks = run_speedtest(api, gt_project_id, model_session_id)
            api.project.update_custom_data(dt_project_id, {"speedtest": benchmarks})


def update_dt_selector_visibility():
    if not run_eval_check.is_checked() and run_speedtest_check.is_checked():
        dt_selector.show()
    else:
        dt_selector.hide()


layout = W.Container([tabs, run_button])

app = sly.Application(layout=layout)
