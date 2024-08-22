<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/model-benchmark/releases/download/v0.0.4/poster.jpg"/>

# Evaluator for Model Benchmark

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/model-benchmark)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/model-benchmark)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/model-benchmark.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/model-benchmark.png)](https://supervise.ly)

</div>

## Overview

The Evaluator for Model Benchmark is a versatile application designed to assess the performance of various machine learning models in a consistent and reliable manner. This app provides a streamlined process for evaluating models and generating comprehensive reports to help you learn different metrics and make informed decisions.

## Preparation

Before running the Evaluator for Model Benchmark, please ensure that you have the following:

- Served model in Supervisely (currently available [Serve YOLOv8 | v9 | v10](https://ecosystem.supervisely.com/apps/yolov8/serve))
- You have prepared a Ground Truth project with the appropriate annotations (classes should be the same as in the model)

## How To Run

**Step 1:** Open the app from the Supervisely Ecosystem.

**Step 2:** Select the project you wish to evaluate.

**Step 3:** Choose the model you want to evaluate from the list of served models.

**Step 4:** Start the evaluation process by clicking the “Run” button. The app will process the data and evaluate the model(s) based on the selected benchmarks. You can monitor the progress in the app’s interface.
