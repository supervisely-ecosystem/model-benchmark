FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libmagic-dev \
    libexiv2-dev \
    libboost-all-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN python3 -m pip install --upgrade \
    setuptools==70.3.0 \
    numpy==2.3.3 \
    supervisely[model-benchmark]==6.73.486

LABEL python_sdk_version=6.73.486
