import src.globals as g
import supervisely as sly
from supervisely.nn.inference import Session

geometry_to_task_type = {
    "object detection": [sly.Rectangle],
    "instance segmentation": [sly.Bitmap, sly.Polygon, sly.AlphaMask],
    # "semantic segmentation": [sly.Bitmap, sly.Polygon, sly.AlphaMask],
}


def get_project_classes():
    meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
    return meta.obj_classes


def get_model_info():
    session = Session(g.api, g.session_id)
    model_meta = session.get_model_meta()
    session_info = session.get_session_info()
    return model_meta.obj_classes, session_info["task type"]


def get_classes():
    project_classes = get_project_classes()
    model_classes, task_type = get_model_info()
    if task_type not in geometry_to_task_type:
        raise ValueError(f"Task type {task_type} is not supported yet")
    filtered_classes = []
    for obj_class in project_classes:
        if model_classes.has_key(obj_class.name):
            if obj_class.geometry_type in geometry_to_task_type[task_type]:
                filtered_classes.append(obj_class)

    return filtered_classes
