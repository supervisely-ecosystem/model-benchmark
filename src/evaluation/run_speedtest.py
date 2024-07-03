import os
import numpy as np
import supervisely as sly
from supervisely.nn.inference import SessionJSON
from tqdm import tqdm


def run_speedtest(
        api: sly.Api,
        project_id: int,
        model_session_id: int,
        batch_size_list: list = (1, 8, 16),
        num_iterations: int = 100,
        num_warmup: int = 3,
        inference_settings: dict = None,
        cache_project_on_model=False,
        ):
    model_session = SessionJSON(api, model_session_id, inference_settings=inference_settings)
    session_info = model_session.get_session_info()

    model_info = {
        "app_name": session_info["app_name"],
        "model_name": "yolov8l",  # TODO: get model name from session_info
    }

    speedtest_info = {
        "runtime": None,
        "device": None,
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
            cache_project_on_model=cache_project_on_model,
            )
        for speedtest in tqdm(iterator):
            speedtest_results.append(speedtest)
        assert len(speedtest_results) == num_iterations, "Speedtest failed to run all iterations."
        x = [[s[k] for s in speedtest_results] for k in speedtest_results[0].keys()]
        x = np.array(x, dtype=float)
        avg = x.mean(1)
        std = x.std(1)
        avg_speedtest = {k: float(avg[i]) if not np.isnan(avg[i]).any() else None for i, k in enumerate(speedtest_results[0].keys())}
        std_speedtest = {k: float(std[i]) if not np.isnan(std[i]).any() else None for i, k in enumerate(speedtest_results[0].keys())}
        benchmark = {
            "benchmark": avg_speedtest,
            "benchmark_std": std_speedtest,
            "batch_size": bs,
            **speedtest_info,
        }
        benchmarks.append(benchmark)
    upload_results(api, benchmarks, model_info)
    return benchmarks, model_info


def upload_results(api: sly.Api, benchmarks: list, model_info: dict):
    model_name = model_info["model_name"]
    data_dir = "APP_DATA"
    b_path = os.path.join(data_dir, "speedtest.json")
    sly.json.dump_json_file(benchmarks, b_path, indent=2)
    api.file.upload(sly.env.team_id(), b_path, f"/model-benchmark/speedtest/{model_name}.json")


if __name__ == "__main__":
    api = sly.Api()
    gt_project_id = 123
    dt_project_id = 456
    model_session_id = 456
    batch_size_list = [1, 8, 16]
    num_iterations = 100
    num_warmup = 3

    benchmarks, model_info = run_speedtest(
        api,
        gt_project_id,
        model_session_id,
        batch_size_list,
        num_iterations,
        num_warmup,
        )
    
    upload_results(api, benchmarks, model_info)
