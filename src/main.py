import src.globals as g
import src.ui._01_overview as overview
import src.ui._02_model_predictions as model_preds
import src.ui._03_confidence_score as conf_score
import src.ui._04_f1_score as f1_score
import src.ui._05_outcome_counts as outcome_counts
import src.ui._06_perclass_metrics as perclass_metrics
import src.ui._07_pr_curve as pr_curve
import src.ui._08_confusion_matrix as confusion_matrix
import src.ui._09_frequently_confused as frequently_confused
import src.ui._10_iou_distribution as iou_distribution
import src.ui._11_calibration_score as calibration_score
import src.ui._12_perclass as perclass
import supervisely as sly
from supervisely.app.widgets import Card, Container

# import src.ui.input as input
# import src.ui.output as output
# import src.ui.settings as settings

layout = Container(
    widgets=[
        Card(
            "Model Benchmark",
            "Description",
            content=Container(
                widgets=[
                    overview.container,
                    model_preds.container,
                    conf_score.container,
                    f1_score.container,
                    outcome_counts.container,
                    perclass_metrics.container,
                    pr_curve.container,
                    confusion_matrix.container,
                    frequently_confused.container,
                    iou_distribution.container,
                    calibration_score.container,
                    perclass.container,
                ],
            ),
        ),
    ]
)

app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
