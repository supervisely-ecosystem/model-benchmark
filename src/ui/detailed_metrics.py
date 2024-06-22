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

f1_optimal_conf, best_f1 = g.m_full.get_f1_optimal_conf()

info = NotificationBox(
    f"""F1-optimal confidence threshold = {f1_optimal_conf:.4f}""",
    "Calculated for the given model and dataset",
)

collapse = Collapse(
    [
        Collapse.Item(
            "Note about confidence threshold",
            "Note about confidence threshold",
            Container(
                [
                    Markdown(
                        "To calculate various metrics, we must set a _confidence threshold_, which also is necessary in deploying a model and applying it to any task. This hyperparameter significantly influences the results of metrics. To eliminate human bias in this process, we automate the determination of the confidence threshold. The threshold is selected based on the best _f1-score_ (guaranteed to give the best f1-score on the given dataset), ensuring a balanced trade-off between precision and recall.",
                        show_border=False,
                        height=80,
                    ),
                    info,
                ]
            ),
        )
    ]
)


container = Container(
    widgets=[
        Markdown("""# Detailed Metrics Analysis""", show_border=False, height=50),
        collapse,
    ]
)
