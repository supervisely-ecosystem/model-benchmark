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
    if g.session is None:
        g.session = Session(g.api, g.session_id)
    model_meta = g.session.get_model_meta()
    session_info = g.session.get_session_info()
    return model_meta.obj_classes, session_info["task type"]


def get_classes():
    project_classes = get_project_classes()
    model_classes, task_type = get_model_info()
    if task_type not in geometry_to_task_type:
        raise ValueError(f"Task type {task_type} is not supported yet")
    matched_proj_cls = []
    matched_model_cls = []
    not_matched_proj_cls = []
    not_matched_model_cls = []
    for obj_class in project_classes:
        if model_classes.has_key(obj_class.name):
            if obj_class.geometry_type in geometry_to_task_type[task_type]:
                matched_proj_cls.append(obj_class)
                matched_model_cls.append(model_classes.get(obj_class.name))
            else:
                not_matched_proj_cls.append(obj_class)
        else:
            not_matched_proj_cls.append(obj_class)

    for obj_class in model_classes:
        if not project_classes.has_key(obj_class.name):
            not_matched_model_cls.append(obj_class)

    return (matched_proj_cls, matched_model_cls), (not_matched_proj_cls, not_matched_model_cls)

def get_matched_class_names():
    (cls_collection, _), _ = get_classes()
    return [obj_cls.name for obj_cls in cls_collection]