import src.globals as g
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Collapse,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    NotificationBox,
    SelectDataset,
    Text,
)

title = Markdown(
    """# Detailed Metrics Analysis""",
    show_border=False,
)

note = (
    Markdown(
        "To calculate various metrics, we must set a _confidence threshold_, which also is necessary in deploying a model and applying it to any task. This hyperparameter significantly influences the results of metrics. To eliminate human bias in this process, we automate the determination of the confidence threshold. The threshold is selected based on the best _f1-score_ (guaranteed to give the best f1-score on the given dataset), ensuring a balanced trade-off between precision and recall."
    ),
)

f1_optimal_conf, best_f1 = g.m_full.get_f1_optimal_conf()

info = Text(
    f"""F1-optimal confidence threshold = {f1_optimal_conf:.4f} *(calculated for the given model and dataset)*""",
    status="info",
)

# iframe_confidence_score = IFrame("static/11_02_confidence_score.html")

collapse = Collapse(
    [
        Collapse.Item(
            "Note about confidence threshold",
            "Note about confidence threshold",
            Container(
                [
                    note,
                    info,
                    # iframe_confidence_score,
                ]
            ),
        )
    ]
)


container = Container(
    widgets=[
        title,
        collapse,
    ]
)
