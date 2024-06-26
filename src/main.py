import src.globals as g
import src.ui.calibration_score as calibration_score
import src.ui.classification_accuracy as classification_accuracy
import src.ui.confusion_matrix as confusion_matrix
import src.ui.detailed_metrics as detailed_metrics
import src.ui.frequently_confused as frequently_confused
import src.ui.inference_speed as inference_speed
import src.ui.iou_distribution as iou_distribution
import src.ui.model_predictions as model_preds
import src.ui.outcome_counts as outcome_counts
import src.ui.overview as overview
import src.ui.perclass as perclass
import src.ui.pr_curve as pr_curve
import src.ui.pr_metrics as pr_metrics
import src.ui.what_is_section as what_is
import supervisely as sly
from supervisely.app import StateJson
from supervisely.app.widgets import Card, Container, Sidebar, Text

fig = overview.overall()
fig.write_html(g.STATIC_DIR + "/01_overview.html")
model_preds.grid_gallery_model_preds()
# fig = outcome_counts.outcome_counts()
# fig.write_html(g.STATIC_DIR + "/05_outcome_counts.html")
pr_metrics.prepare()
fig = pr_metrics.perclass_PR()
fig.write_html(g.STATIC_DIR + "/06_1_perclass_PR.html")
fig = pr_metrics.perclass_P()
fig.write_html(g.STATIC_DIR + "/06_2_perclass_P.html")
fig = pr_metrics.perclass_R()
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
fig1, fig2 = frequently_confused.frequently_confused()
fig1.write_html(g.STATIC_DIR + "/09_01_frequently_confused.html")
fig2.write_html(g.STATIC_DIR + "/09_02_frequently_confused.html")
fig = iou_distribution.iou_distribution()
fig.write_html(g.STATIC_DIR + "/10_iou_distribution.html")
fig = calibration_score.reliability_diagram()
fig.write_html(g.STATIC_DIR + "/11_01_reliability_diagram.html")
fig = calibration_score.confidence_score()
fig.write_html(g.STATIC_DIR + "/11_02_confidence_score.html")
fig = calibration_score.f1score_at_different_iou()
fig.write_html(g.STATIC_DIR + "/11_03_f1score_at_different_iou.html")
fig = calibration_score.confidence_distribution()
fig.write_html(g.STATIC_DIR + "/11_04_confidence_distribution.html")
fig = perclass.perclass_ap()
fig.write_html(g.STATIC_DIR + "/12_01_perclass.html")
fig, fig_ = perclass.perclass_outcome_counts()
fig.write_html(g.STATIC_DIR + "/12_02_perclass.html")
fig_.write_html(g.STATIC_DIR + "/12_03_perclass.html")

StateJson()["scrollIntoView"] = ""

left_content = Card(
    title="Table of Content",
    content=Container(
        [
            Text("Overview", scroll_to_widget=overview.markdown_overview.widget_id),
            Text("  Key Metrics", scroll_to_widget=overview.markdown_key_metrics.widget_id),
            Text("Explore Predictions", scroll_to_widget=overview.markdown_explorer.widget_id),
            Text("  Model Predictions", scroll_to_widget=model_preds.container.widget_id),
            Text("  What is YOLOv8", scroll_to_widget=what_is.markdown_what_is.widget_id),
            Text("  Experts Insights", scroll_to_widget=what_is.markdown_experts.widget_id),
            Text("  How To Use", scroll_to_widget=what_is.markdown_how_to_use.widget_id),
            Text("Detailed Metrics", scroll_to_widget=detailed_metrics.container.widget_id),
            Text("  Outcome Counts", scroll_to_widget=outcome_counts.container.widget_id),
            Text("  Recall", scroll_to_widget=pr_metrics.markdown_R.widget_id),
            Text("  Precision", scroll_to_widget=pr_metrics.markdown_P.widget_id),
            Text("  Recall vs Precision", scroll_to_widget=pr_metrics.markdown_PR.widget_id),
            Text("  Precision-Recall Curve", scroll_to_widget=pr_curve.markdown_pr_curve.widget_id),
            Text(
                "      PR Curve by Class", scroll_to_widget=pr_curve.markdown_pr_by_class.widget_id
            ),
            Text(
                "   Classification Accuracy",
                scroll_to_widget=classification_accuracy.container.widget_id,
            ),
            Text("      Confusion Matrix", scroll_to_widget=confusion_matrix.container.widget_id),
            Text(
                "       Frequently Confused Class Pairs",
                scroll_to_widget=frequently_confused.container.widget_id,
            ),
            Text(
                "   Localization Accuracy (IoU)",
                scroll_to_widget=iou_distribution.container.widget_id,
            ),
            Text("      IoU Distribution", scroll_to_widget=iou_distribution.container.widget_id),
            Text(
                "   Calibration Score",
                scroll_to_widget=calibration_score.markdown_calibration_score_1.widget_id,
            ),
            Text(
                "       Reliability Diagram",
                scroll_to_widget=calibration_score.markdown_reliability_diagram.widget_id,
            ),
            Text(
                "   Confidence Score Profile",
                scroll_to_widget=calibration_score.markdown_confidence_score_1.widget_id,
            ),
            Text(
                "       Confidence Distribution",
                scroll_to_widget=calibration_score.markdown_confidence_distribution.widget_id,
            ),
            Text(
                "       Average Precision by Class",
                scroll_to_widget=perclass.markdown_class_ap.widget_id,
            ),
            Text(
                "       Outcome Counts by Class",
                scroll_to_widget=perclass.markdown_class_outcome_counts_1.widget_id,
            ),
            Text(
                "   Inference Speed",
                scroll_to_widget=inference_speed.markdown_inference_speed_1.widget_id,
            ),
        ]
    ),
)

right_content = Card(
    "Model Benchmark",
    "Description",
    content=Container(
        widgets=[
            overview.container,
            model_preds.container,
            what_is.container,
            detailed_metrics.container,
            outcome_counts.container,
            pr_metrics.container,
            pr_curve.container,
            classification_accuracy.container,
            confusion_matrix.container,
            frequently_confused.container,
            iou_distribution.container,
            calibration_score.container,
            perclass.container,
            inference_speed.container,
        ],
    ),
)

layout = Container(
    widgets=[
        Sidebar(left_content=left_content, right_content=right_content, width_percent=20),
        outcome_counts.dialog,
    ]
)

app = sly.Application(layout=layout, static_dir=g.STATIC_DIR)
