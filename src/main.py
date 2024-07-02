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

    fig = confusion_matrix._confusion_matrix()
    json_fig = fig.to_json()

    local_path = f"{g.TO_TEAMFILES_DIR}/confusion_matrix.json"

    tf_path = f"{g.TF_RESULT_DIR}/confusion_matrix.json"

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(json_fig, f)

    g.api.file.upload(g._team_id, local_path, tf_path)

    sly.logger.info("Done.")


button = Button("Click to calc")
layout = Container(widgets=[Text("some_text"), button])


@button.click
def handle():
    main_func()


app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
