from typing import Union
import supervisely as sly
from supervisely.nn.inference import SessionJSON
from tqdm import tqdm


def run_speedtest(
        project_id: int,
        model_session: SessionJSON,
        batch_size_list: list = (1, 8, 16),
        num_iterations: int = 100,
        num_warmup: int = 5,
        ):
    session_info = model_session.get_session_info()

    speedtest_info = {
        "runtime": None,
        "device": session_info["device"],
        "hardware": None,
        "num_iterations": num_iterations,
    }

    benchmarks = []
    for bs in batch_size_list:
        print(f"Running speedtest for batch_size={bs}")
        speedtest_results = []
        iterator = model_session.run_benchmark(
            project_id,
            batch_size=bs,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
            cache_project_on_model=True
            )
        for speedtest in tqdm(iterator):
            speedtest_results.append(speedtest)
        assert len(speedtest_results) == num_iterations, "Speedtest failed to run all iterations."
        avg_speedtest = {k: sum([s[k] for s in speedtest_results]) / len(speedtest_results) for k in speedtest_results[0].keys()}
        benchmark = {
            "benchmark": avg_speedtest,
            "batch_size": bs,
            **speedtest_info,
        }
        benchmarks.append(benchmark)
    return benchmarks


if __name__ == "__main__":
    api = sly.Api()
    gt_project_id = 123
    dt_project_id = 456
    model_session_id = 456
    batch_size_list = [1, 8, 16]
    num_iterations = 100
    num_warmup = 5
    session = SessionJSON(api, model_session_id)

    benchmarks = run_speedtest(
        gt_project_id,
        session,
        batch_size_list,
        num_iterations,
        num_warmup,
        )
    api.project.update_custom_data(dt_project_id, {"speedtest": benchmarks})
    
    
