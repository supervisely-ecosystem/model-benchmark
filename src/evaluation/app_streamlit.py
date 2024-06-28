import streamlit as st
import os
import subprocess


tab1, tab2 = st.tabs(["Run evaluation", "Load Dashboard"])

with tab1:
    st.markdown("#### Evaluation")
    gt_project_id = st.text_input("GT project id:", help="Enter the project id of the ground truth dataset")
    if gt_project_id:
        st.multiselect("Select dataset", ["ds1", "ds2"])
    model_session_id = st.text_input("Model session id")
    if model_session_id:
        st.json({"model": "YOLOv8", "dataset": "COCO"})
        
    with st.expander("Advanced options"):
        batch_size = st.number_input("Batch size", min_value=0, value=8)
        inference_settings_str = st.text_area("Inference settings", "conf: 0.05")
        cache_project = st.checkbox("Cache project on the model for future use")

    st.markdown("#### Speed test")
    run_speedtest = st.checkbox("Run speed test")
    if run_speedtest:
        st.text_area("Speed test settings:", "batch_size: 8\nnum_workers: 4")

with tab2:
    dt_project_id = st.text_input("DT project id")

if st.button("Run"):
    st.divider()
    with st.spinner("Running evaluation..."):
        script = "src/evaluation/run_evaluation.py"
        cmd = f"python {script} --gt_project_id {gt_project_id} --model_session_id {model_session_id} " \
              f"--cache_project {cache_project} --batch_size {batch_size}"
        cmd = cmd.split()
        cmd += ["--inference_settings", inference_settings_str]
        st.code(cmd)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, shell=True)

        # Read the output and error (if any) line by line
        # output = []
        # for stdout_line in iter(process.stdout.readline, ""):
        #     s = stdout_line.strip()
        st.write_stream(iter(process.stdout.readline, "\n"))
        for stderr_line in iter(process.stderr.readline, ""):
            print(stderr_line, end="")
            st.write(stderr_line)