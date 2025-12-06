FROM python:3.12

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
    supervisely[model-benchmark]==6.73.479 \
    numpy==2.3.3

LABEL python_sdk_version=6.73.479
