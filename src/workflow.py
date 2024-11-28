# This module contains functions that are used to configure the input and output of the workflow for the current app,
# and versioning feature that creates a project version before the task starts.
from typing import List, Optional

import supervisely as sly


def workflow_input(
    api: sly.Api,
    project_info: Optional[sly.ProjectInfo] = None,
    session_id: Optional[int] = None,
    team_files_dirs: Optional[List[str]] = None,
    model_benchmark_reports: Optional[List[sly.api.file_api.FileInfo]] = None,
):
    if project_info:
        # Create a project version before the task starts
        try:
            project_version_id = api.project.version.create(
                project_info,
                f"Evaluator for Model Benchmark",
                f"This backup was created automatically by Supervisely before the Evaluator for Model Benchmark task with ID: {api.task_id}",
            )
        except Exception as e:
            sly.logger.debug(f"Failed to create a project version: {repr(e)}")
            project_version_id = None

        # Add input project to the workflow
        try:
            if project_version_id is None:
                project_version_id = (
                    project_info.version.get("id", None) if project_info.version else None
                )
            api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
            sly.logger.debug(
                f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}"
            )
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")

        # Add input model session to the workflow
        try:
            api.app.workflow.add_input_task(session_id)
            sly.logger.debug(f"Workflow Input: Session ID - {session_id}")
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")

    if team_files_dirs:
        # Add input evaluation results folders to the workflow
        try:
            for team_files_dir in team_files_dirs:
                api.app.workflow.add_input_folder(team_files_dir)
                sly.logger.debug(f"Workflow Input: Team Files dir - {team_files_dir}")
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")

    if model_benchmark_reports:
        # Add input model benchmark reports to the workflow
        try:
            for model_benchmark_report in model_benchmark_reports:
                api.app.workflow.add_input_file(model_benchmark_report)
                sly.logger.debug(f"Workflow Input: Model Benchmark Report ID - {model_benchmark_report.id}")
        except Exception as e:
            sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(
    api: sly.Api,
    eval_team_files_dir: Optional[str] = None,
    model_benchmark_report: Optional[sly.api.file_api.FileInfo] = None,
    model_comparison_report: Optional[sly.api.file_api.FileInfo] = None,
):
    if model_benchmark_report:
        try:
            # Add output evaluation results folder to the workflow
            # eval_dir_relation_settings = sly.WorkflowSettings(title="Evaluation Artifacts")
            # eval_dir_meta = sly.WorkflowMeta(relation_settings=eval_dir_relation_settings)
            # api.app.workflow.add_output_folder(eval_team_files_dir, meta=eval_dir_meta)
            # sly.logger.debug(f"Workflow Output: Team Files dir - {eval_team_files_dir}")

            # Add output model benchmark report to the workflow
            mb_relation_settings = sly.WorkflowSettings(
                title="Model Benchmark",
                icon="assignment",
                icon_color="#dcb0ff",
                icon_bg_color="#faebff",
                url=f"/model-benchmark?id={model_benchmark_report.id}",
                url_title="Open Benchmark Report",
            )
            meta = sly.WorkflowMeta(relation_settings=mb_relation_settings)
            api.app.workflow.add_output_file(model_benchmark_report, meta=meta)
            sly.logger.debug(f"Model Evaluation Report ID - {model_benchmark_report.id}")

        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")

    if model_comparison_report:
        try:
            # Add output model benchmark report to the workflow
            comparison_relation_settings = sly.WorkflowSettings(
                title="Model Comparison",
                icon="assignment",
                icon_color="#ffc084",
                icon_bg_color="#fff2e6",
                url=f"/model-benchmark?id={model_comparison_report.id}",
                url_title="Open Comparison Report",
            )
            meta = sly.WorkflowMeta(relation_settings=comparison_relation_settings)
            api.app.workflow.add_output_file(model_comparison_report, meta=meta)
            sly.logger.debug(f"Model Comparison Report ID - {model_comparison_report.id}")

        except Exception as e:
            sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
