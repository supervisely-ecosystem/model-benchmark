import src.globals as g
import src.ui._01_overview as overview
import src.ui._02_model_predictions as model_preds

# import src.ui._03_confidence_score as conf_score
# import src.ui._04_f1_score as f1_score
import src.ui._05_outcome_counts as outcome_counts
import src.ui._06_perclass_metrics as perclass_metrics
import src.ui._07_pr_curve as pr_curve
import src.ui._08_confusion_matrix as confusion_matrix
import src.ui._09_frequently_confused as frequently_confused
import src.ui._10_iou_distribution as iou_distribution
import src.ui._11_calibration_score as calibration_score
import src.ui._12_perclass as perclass
import src.ui.detailed_metrics as detailed_metrics
import src.ui.what_is_section as what_is
import supervisely as sly
from supervisely.app.widgets import Card, Container, Empty

# import src.ui.input as input
# import src.ui.output as output
# import src.ui.settings as settings

fig = overview.overall()
fig.write_html(g.STATIC_DIR + "/01_overview.html")
model_preds.grid_gallery_model_preds()
# fig = conf_score.confidence_score()
# fig.write_html(g.STATIC_DIR + "/03_confidence_score.html")
# fig = f1_score.f1_score()
# fig.write_html(g.STATIC_DIR + "/04_f1_score.html")
fig = outcome_counts.outcome_counts()
fig.write_html(g.STATIC_DIR + "/05_outcome_counts.html")
perclass_metrics.prepare()
fig = perclass_metrics.perclass_PR()
fig.write_html(g.STATIC_DIR + "/06_1_perclass_PR.html")
fig = perclass_metrics.perclass_P()
fig.write_html(g.STATIC_DIR + "/06_2_perclass_P.html")
fig = perclass_metrics.perclass_R()
fig.write_html(g.STATIC_DIR + "/06_3_perclass_R.html")
pr_curve.prepare()
fig = pr_curve._pr_curve()
fig.write_html(g.STATIC_DIR + "/07_01_pr_curve.html")
fig = pr_curve.pr_curve_perclass()
fig.write_html(g.STATIC_DIR + "/07_02_pr_curve_perclass.html")
fig = confusion_matrix._confusion_matrix()
fig.write_html(g.STATIC_DIR + "/08_1_confusion_matrix.html")
fig = confusion_matrix.confusion_matrix_mini()
fig.write_html(g.STATIC_DIR + "/08_2_confusion_matrix.html")
fig = frequently_confused.frequently_confused()
fig.write_html(g.STATIC_DIR + "/09_frequently_confused.html")
fig = iou_distribution.iou_distribution()
fig.write_html(g.STATIC_DIR + "/10_iou_distribution.html")
fig = calibration_score.calibration_curve()
fig.write_html(g.STATIC_DIR + "/11_01_calibration_curve.html")
fig = calibration_score.confidence_score()
fig.write_html(g.STATIC_DIR + "/11_02_confidence_score.html")
fig = calibration_score.f1score_at_different_iou()
fig.write_html(g.STATIC_DIR + "/11_03_f1score_at_different_iou.html")
fig = calibration_score.confidence_histogram()
fig.write_html(g.STATIC_DIR + "/11_04_confidence_histogram.html")
fig = perclass.perclass_ap()
fig.write_html(g.STATIC_DIR + "/12_01_perclass.html")
fig, fig_ = perclass.perclass_outcome_counts()
fig.write_html(g.STATIC_DIR + "/12_02_perclass.html")
fig_.write_html(g.STATIC_DIR + "/12_03_perclass.html")

layout = Container(
    widgets=[
        Card(
            "Model Benchmark",
            "Description",
            content=Container(
                widgets=[
                    overview.container,
                    model_preds.container,
                    # conf_score.container,
                    # f1_score.container,
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
