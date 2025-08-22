from typing import List

import src.functions as f
import src.globals as g
import src.workflow as w
import supervisely as sly
import supervisely.app.widgets as widgets
from supervisely._utils import rand_str
from supervisely.nn.benchmark.comparison.model_comparison import ModelComparison

compare_button = widgets.Button("Compare")
comp_pbar = widgets.SlyTqdm()
models_comparison_report = widgets.ReportThumbnail(
    title="Models Comparison Report",
    color="#ffc084",
    bg_color="#fff2e6",
)
models_comparison_report.hide()
team_files_selector = widgets.TeamFilesSelector(
    g.team_id,
    multiple_selection=True,
    selection_file_type="folder",
    max_height=350,
    initial_folder="/model-benchmark",
)

compare_contatiner = widgets.Container(
    [
        team_files_selector,
        compare_button,
        models_comparison_report,
        comp_pbar,
    ]
)


@f.with_clean_up_progress(comp_pbar)
def run_compare(eval_dirs: List[str] = None):
    workdir = g.STORAGE_DIR + "/model-comparison-" + rand_str(6)
    team_files_selector.disable()
    models_comparison_report.hide()
    comp_pbar.show()

    g.eval_dirs = eval_dirs or team_files_selector.get_selected_paths()
    f.validate_paths(g.eval_dirs)

    # ==================== Workflow input ====================
    reports = None
    try:
        reports_paths = [path.rstrip("/") + "/visualizations/template.vue" for path in g.eval_dirs]
        reports = [g.api.file.get_info_by_path(g.team_id, path) for path in reports_paths]
    except Exception as e:
        sly.logger.warning(f"Failed to get model benchmark reports FileInfos: {repr(e)}")

    if reports is not None:
        w.workflow_input(g.api, model_benchmark_reports=reports)
    else:
        w.workflow_input(g.api, team_files_dirs=g.eval_dirs)
    # =======================================================

    comp = ModelComparison(g.api, g.eval_dirs, progress=comp_pbar, workdir=workdir)
    comp.visualize()
    res_dir = f.get_res_dir(g.eval_dirs)
    res_dir = comp.upload_results(g.team_id, remote_dir=res_dir, progress=comp_pbar)

    g.api.task._set_custom_output(
        task_id=g.api.task_id,
        file_id=comp.lnk.id,
        file_name=comp.lnk.name,
        file_url=f"/model-benchmark?id={comp.report.id}",
        description="Click to open the report",
        icon="zmdi zmdi-receipt",
        color="#dcb0ff",
        background_color="#faebff",
    )

    models_comparison_report.set(comp.report)
    models_comparison_report.show()

    # ==================== Workflow output ====================
    w.workflow_output(g.api, model_comparison_report=comp.report)
    # =======================================================

    comp_pbar.hide()
    compare_button.loading = False

    sly.logger.info(f"Model comparison report uploaded to: {res_dir}")
    sly.logger.info(f"Report link: {comp.get_report_link()}")

    return res_dir
