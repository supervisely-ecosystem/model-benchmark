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
from supervisely.app.widgets import Container

# import src.ui.input as input
# import src.ui.output as output
# import src.ui.settings as settings

layout = Container(
    widgets=[
        overview.card,
        model_preds.card,
        conf_score.card,
        f1_score.card,
        outcome_counts.card,
        perclass_metrics.card,
        pr_curve.card,
        confusion_matrix.card,
        frequently_confused.card,
        iou_distribution.card,
        calibration_score.card,
        perclass.card,
    ]
)

app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
