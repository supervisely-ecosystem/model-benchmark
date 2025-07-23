import os
from typing import List, Tuple

import supervisely as sly
from fastapi import WebSocket
from supervisely.nn import TaskType
from supervisely.nn.inference import SessionJSON

import src.globals as g

geometry_to_task_type = {
    TaskType.OBJECT_DETECTION: [sly.Rectangle, sly.AnyGeometry],
    TaskType.INSTANCE_SEGMENTATION: [sly.Bitmap, sly.Polygon, sly.AnyGeometry],
    TaskType.SEMANTIC_SEGMENTATION: [sly.Bitmap, sly.Polygon, sly.AnyGeometry],
}


def get_project_classes():
    if g.project_id is None:
        return
    meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
    return meta.obj_classes


def get_model_info():
    if g.session_id is None:
        g.session = None
        return None, None
    g.session = SessionJSON(g.api, g.session_id)
    model_meta = g.session.get_model_meta()
    model_meta = sly.ProjectMeta.from_json(model_meta)
    session_info = g.session.get_session_info()
    return model_meta.obj_classes, session_info["task type"]


def get_classes():
    if g.task_type not in geometry_to_task_type:
        raise ValueError(f"Task type {g.task_type} is not supported yet")
    matched_proj_cls = []
    matched_model_cls = []
    not_matched_proj_cls = []
    not_matched_model_cls = []
    for obj_class in g.project_classes:
        if g.model_classes.has_key(obj_class.name):
            if obj_class.geometry_type in geometry_to_task_type[g.task_type]:
                matched_proj_cls.append(obj_class)
                matched_model_cls.append(g.model_classes.get(obj_class.name))
            else:
                not_matched_proj_cls.append(obj_class)
        else:
            not_matched_proj_cls.append(obj_class)

    for obj_class in g.model_classes:
        if not g.project_classes.has_key(obj_class.name):
            not_matched_model_cls.append(obj_class)

    return (matched_proj_cls, matched_model_cls), (not_matched_proj_cls, not_matched_model_cls)


def validate_paths(paths: List[str]):
    if not paths:
        raise ValueError("No paths selected")

    split_paths = [path.strip("/").split(os.sep) for path in paths]
    path_length = min(len(p) for p in split_paths)

    if not all(len(p) == path_length for p in split_paths):
        raise ValueError(f"Selected paths not on the correct level: {paths}")

    if not all(p.startswith("/model-benchmark") for p in paths):
        raise ValueError(f"Selected paths are not in the benchmark directory: {paths}")

    if not all(p[1] == split_paths[0][1] for p in split_paths):
        raise ValueError(f"Project names are different: {paths}")


def get_parent_paths(paths: List[str]) -> Tuple[str, List[str]]:
    split_paths = [path.strip("/").split(os.sep) for path in paths]
    project_name = split_paths[0][1]
    eval_dirs = [p[2] for p in split_paths]

    return project_name, eval_dirs


def get_res_dir(eval_dirs: List[str]) -> str:

    res_dir = "/model-comparison"
    project_name, eval_dirs = get_parent_paths(eval_dirs)
    res_dir += "/" + project_name + "/"
    res_dir += " vs ".join(eval_dirs)

    res_dir = g.api.file.get_free_dir_name(g.team_id, res_dir)

    return res_dir


# ! temp fix (to allow the app to receive requests)
def with_clean_up_progress(pbar):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                with pbar(message="Application is started ...", total=1) as pb:
                    pb.update(1)
                pbar.hide()

        return wrapper

    return decorator


class ProgressMonitorManager:
    def __init__(self):
        self.open_connections: dict[str, list[WebSocket]] = {}

        import asyncio
        import threading

        self._send_loop = asyncio.new_event_loop()

        def _start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_start_loop, args=(self._send_loop,), daemon=True)
        thread.start()

    async def add_connection(self, session_id: int, websocket: WebSocket):
        await websocket.accept()
        self.open_connections.setdefault(session_id, []).append(websocket)

    def disconnect(self, session_id: int):
        self.open_connections.pop(session_id, None)

    async def send_progress(self, session_id: int, progress: dict):
        import supervisely as sly

        if session_id not in self.open_connections:
            return

        broken = []
        for ws in self.open_connections[session_id]:
            try:
                await ws.send_json({"progress": progress})
            except Exception as e:
                sly.logger.error(f"WebSocket error: {e}")
                broken.append(ws)

        for ws in broken:
            self.open_connections[session_id].remove(ws)

        if not self.open_connections[session_id]:
            del self.open_connections[session_id]

    def send_progress_sync(self, session_id: int, progress: dict):
        """
        Synchronous wrapper for send_progress: always schedules on the dedicated background loop.
        """
        import asyncio

        import supervisely as sly

        if session_id not in self.open_connections:
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self.send_progress(session_id, progress), self._send_loop
            )
        except Exception as e:
            sly.logger.error(f"Failed to send progress: {e}")
            raise
            return


def patch_pbar(sly_tqdm_instance):
    """Patch progress bar to send progress updates via g.pmm.send_progress"""

    class ProgressBarWrapper:
        def __init__(self, wrapped_instance):
            self._wrapped = wrapped_instance

        def __getattr__(self, name):
            attr = getattr(self._wrapped, name)
            if callable(attr):

                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if name == "update":
                        progress_data = {
                            "current": getattr(self._wrapped, "n", 0),
                            "total": getattr(self._wrapped, "total", None),
                            "message": getattr(self._wrapped, "desc", "Progress update"),
                        }
                        try:
                            g.pmm.send_progress_sync(g.session_id, progress_data)
                        except Exception as e:
                            sly.logger.error(f"Error sending progress: {e}")
                    return result

                return wrapper
            return attr

        def __call__(self, *args, **kwargs):
            result = self._wrapped.__call__(*args, **kwargs)
            # Also patch the returned tqdm instance from __call__
            if hasattr(result, "update"):
                original_update = result.update

                def patched_update(*update_args, **update_kwargs):
                    ret = original_update(*update_args, **update_kwargs)
                    progress_data = {
                        "current": getattr(result, "n", 0),
                        "total": getattr(result, "total", None),
                        "message": kwargs.get("message", "Progress update"),
                    }
                    try:
                        g.pmm.send_progress_sync(g.session_id, progress_data)
                    except Exception as e:
                        sly.logger.error(f"Error sending progress: {e}")
                    return ret

                result.update = patched_update
            return result

    return ProgressBarWrapper(sly_tqdm_instance)
