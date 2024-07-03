import json

import src.globals as g

# import src.ui.calibration_score as calibration_score
# import src.ui.classification_accuracy as classification_accuracy
import src.ui.confusion_matrix as confusion_matrix

# import src.ui.detailed_metrics as detailed_metrics
# import src.ui.frequently_confused as frequently_confused
# import src.ui.inference_speed as inference_speed
# import src.ui.iou_distribution as iou_distribution
import src.ui.model_predictions as model_preds
import src.ui.outcome_counts as outcome_counts
import src.ui.overview as overview

# import src.ui.perclass as perclass
# import src.ui.pr_curve as pr_curve
# import src.ui.pr_metrics as pr_metrics
# import src.ui.what_is_section as what_is
import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Button, Card, Container, Sidebar, Text


def main_func():

    for module in [overview, outcome_counts, confusion_matrix]:

        fig = module.get_figure()
        json_fig = fig.to_json()

        name = module.__name__.split(".")[-1]

        local_path = f"{g.TO_TEAMFILES_DIR}/{name}.json"

        with open(local_path, "w", encoding="utf-8") as f:
            f.write(json_fig)

        sly.logger.info(f"{name!r} - Done.")

    table_preds = g.m.prediction_table()
    local_path = f"{g.TO_TEAMFILES_DIR}/prediction_table.json"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(table_preds.to_json())

    g.api.file.upload_directory(
        g._team_id, g.TO_TEAMFILES_DIR, g.TF_RESULT_DIR, replace_if_conflict=True
    )

    sly.logger.info("Done.")


button = Button("Click to calc")
layout = Container(widgets=[Text("some_text"), button])


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
