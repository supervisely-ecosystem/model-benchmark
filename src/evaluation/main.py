import supervisely as sly
from supervisely.nn.inference import SessionJSON
from tqdm import tqdm


gt_project_id = 36401  # COCO 2017 val
model_session_id = 2
run_evaluation = True
run_speedtest = True

# optional:
cache_project = True  # Whether to cache project on the agent
dt_project_id = None  # A home project for evaluation, where results will be stored
gt_dataset_ids = None  # Datasets to use for evaluation
speedtest_project_id = None  # Use this project for speedtest, instead of gt_project_id

# evaluation params:
batch_size = 16
inference_params = {
    "conf": 0.05,
    "input_size": 640,
}

# speedtest params:
speedtest_params = {
    "batch_size": [1, 8, 16],
    "num_iterations": 100,
    "num_warmup": 3,
}

# validate
if run_speedtest and not run_evaluation:
    assert dt_project_id is not None, "Need to specify dt_project_id for speedtest"
if run_evaluation:
    assert dt_project_id is None, "dt_project_id must be None"



api = sly.Api()

session = SessionJSON(api, model_session_id, inference_settings=inference_params)
session_info = session.get_session_info()


if run_evaluation:
    
    dt_project_name = ...
    dt_project_info = api.project.create(sly.env.workspace_id(), dt_project_name)
    dt_project_id = dt_project_info.id

    evaluation_info = {
        "model_name": session_info["model_name"],
        "display_model_name": session_info["display_model_name"],
        "architecture": session_info["architecture"],
        "deploy_params": session_info["deploy_params"],
        "pretrained_checkpoint_info": session_info["pretrained_checkpoint_info"],
        "custom_checkpoint_info": session_info["custom_checkpoint_info"],
    }

    task_info = api.task.get_info_by_id(model_session_id)
    # TODO: app_info
    # - app_info:
    #   - name
    #   - version
    #   - url
    #   - ...

    iterator = session.inference_project_id_async(
        gt_project_id,
        gt_dataset_ids,
        output_project_id=dt_project_id,
        cache_project_on_model=cache_project,
    )

    for _ in tqdm(iterator):
        pass

    # 2. Download DT project

    # 3. Calculate metrics



if run_speedtest:
    if speedtest_project_id is None:
        speedtest_project_id = gt_project_id

    speedtest_info = {
        "runtime": session_info["runtime"],
        "device": session_info["device"],
        "hardware": session_info["hardware"],
        "num_iterations": speedtest_params["num_iterations"],
    }

    speedtest_results = []
    for bs in speedtest_params["batch_size"]:
        print(f"Running speedtest for batch_size={bs}")
        iterator = session.run_benchmark(
            speedtest_project_id,
            batch_size=bs,
            num_iterations=speedtest_params["num_iterations"],
            num_warmup=speedtest_params["num_warmup"],
            cache_project_on_model=True
            )
        for speedtest in tqdm(iterator):
            speedtest_results.append(speedtest)