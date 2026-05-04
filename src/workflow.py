# This module contains functions that are used to configure the input and output of the workflow for the current app,
# and versioning feature that creates a project version before the task starts.
from pathlib import Path
from typing import List, Optional

import supervisely as sly


def _add_output_report(
    api: sly.Api,
    report: sly.api.file_api.FileInfo,
    title: str,
    url_title: str,
    log_name: str,
):
    try:
        relation_settings = sly.WorkflowSettings(
            title=title,
            icon="assignment",
            icon_color="#dcb0ff",
            icon_bg_color="#faebff",
            url=f"/model-benchmark?id={report.id}",
            url_title=url_title,
        )
        meta = sly.WorkflowMeta(relation_settings=relation_settings)
        api.app.workflow.add_output_file(report, meta=meta)
        sly.logger.debug(f"{log_name} Report ID - {report.id}")
    except Exception as e:
        sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")


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
        _add_output_report(
            api,
            model_benchmark_report,
            title="Model Benchmark",
            url_title="Open Benchmark Report",
            log_name="Model Evaluation",
        )

    if model_comparison_report:
        _add_output_report(
            api,
            model_comparison_report,
            title="Model Comparison",
            url_title="Open Comparison Report",
            log_name="Model Comparison",
        )


def workflow_existing_comparison(
    api: sly.Api,
    team_id: int,
    eval_dirs: List[str],
    comparison_dir: str,
    comparison_link: sly.api.file_api.FileInfo,
):
    try:
        reports_paths = [path.rstrip("/") + "/visualizations/template.vue" for path in eval_dirs]
        reports = [
            report
            for report in (api.file.get_info_by_path(team_id, path) for path in reports_paths)
            if report is not None
        ]
        if reports:
            workflow_input(api, model_benchmark_reports=reports)
        else:
            workflow_input(api, team_files_dirs=eval_dirs)
    except Exception as e:
        sly.logger.debug(f"Failed to add workflow input for existing comparison: {repr(e)}")

    try:
        report_fileinfo = api.file.get_info_by_path(
            team_id,
            str(Path(comparison_dir) / "visualizations" / "template.vue"),
        )
        workflow_output(api, model_comparison_report=report_fileinfo or comparison_link)
    except Exception as e:
        sly.logger.debug(f"Failed to add workflow output for existing comparison: {repr(e)}")
