from typing import Union

import supervisely as sly
from supervisely.api.file_api import FileInfo


def check_compatibility(func):
    def wrapper(self, *args, **kwargs):
        if self.is_compatible is None:
            try:
                self.is_compatible = self.check_instance_ver_compatibility()
            except Exception as e:
                sly.logger.error(
                    "Can not check compatibility with Supervisely instance. "
                    f"Workflow features will be disabled. Error: {repr(e)}"
                )
                self.is_compatible = False
        if not self.is_compatible:
            return
        return func(self, *args, **kwargs)

    return wrapper


class Workflow:
    def __init__(self, api: sly.Api, min_instance_version: str = None):
        self.is_compatible = None
        self.api = api
        self._min_instance_version = (
            "6.9.31" if min_instance_version is None else min_instance_version
        )

    def check_instance_ver_compatibility(self):
        if not self.api.is_version_supported(self._min_instance_version):
            sly.logger.info(
                f"Supervisely instance version {self.api.instance_version} does not support workflow features."
            )
            if not sly.is_community():
                sly.logger.info(
                    f"To use them, please update your instance to version {self._min_instance_version} or higher."
                )
            return False
        return True

    @check_compatibility
    def add_input(self, item: Union[sly.ProjectInfo, int]):
        if isinstance(item, int):
            self.api.app.workflow.add_input_task(item)
        if isinstance(item, sly.ProjectInfo):
            self.api.app.workflow.add_input_project(item.id)

    @check_compatibility
    def add_output(self, item: Union[sly.ProjectInfo, str]):
        try:
            if isinstance(item, sly.ProjectInfo):
                self.api.app.workflow.add_output_project(item.id)
            if isinstance(item, str):
                self.api.app.workflow.add_output_folder(item, task_id=sly.env.task_id())
                module_id = (
                    self.api.task.get_info_by_id(self.api.task_id)
                    .get("meta", {})
                    .get("app", {})
                    .get("id")
                )

                template_vis_file = self.api.file.get_info_by_path(
                    sly.env.team_id(), item + "visualizations/template.vue"
                )

                meta = {
                    "customNodeSettings": {
                        "title": f"<h4>Evaluator for Model Benchmark</h4>",
                        "mainLink": {
                            "url": (
                                f"/apps/{module_id}/sessions/{self.api.task_id}"
                                if module_id
                                else f"apps/sessions/{self.api.task_id}"
                            ),
                            "title": "Show Results",
                        },
                    },
                    "customRelationSettings": {
                        "icon": {
                            "icon": "zmdi-assignment",
                            "color": "#674EA7",
                            "backgroundColor": "#CCCCFF",
                        },
                        "title": "<h4>Model Benchmark</h4>",
                        "mainLink": {
                            "url": f"/model-benchmark?id={template_vis_file.id}",
                            "title": "Open Report",
                        },
                    },
                }
                self.api.app.workflow.add_output_file(template_vis_file, meta=meta)

        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")

    @check_compatibility
    def add_output_report(self, template_vis_file: FileInfo):
        try:

            meta = {
                "customRelationSettings": {
                    "icon": {
                        "icon": "zmdi-assignment",
                        "color": "#674EA7",
                        "backgroundColor": "#CCCCFF",
                    },
                    "title": "<h4>Model Benchmark</h4>",
                    "mainLink": {
                        "url": f"/model-benchmark?id={template_vis_file.id}",
                        "title": "Open Report",
                    },
                },
            }
            self.api.app.workflow.add_output_file(template_vis_file, meta=meta)
        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")

    @check_compatibility
    def add_output_project(self, item: sly.ProjectInfo):
        try:
            if isinstance(item, sly.ProjectInfo):
                self.api.app.workflow.add_output_project(item.id)
        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
