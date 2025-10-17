<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/model-benchmark/releases/download/v0.0.4/poster.jpg"/>

# Evaluator for Model Benchmark

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/model-benchmark)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/model-benchmark)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/model-benchmark.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/model-benchmark.png)](https://supervisely.com)

</div>

## Overview

The Evaluator for Model Benchmark is a versatile application designed to assess the performance of various machine learning models in a consistent and reliable manner. This app provides a streamlined process for evaluating models and generating comprehensive reports to help you learn different metrics and make informed decisions.

The Evaluator app offers a range of evaluation metrics, including precision, recall, F1 score, mAP, and more. The app also includes a **Model Comparison** feature that allows you to compare the performance of multiple models side by side.

**Changelog:**

- **v0.1.0** – Public release (for object detection task type)
- **v0.1.2** – Support for instance segmentation task type
- **v0.1.4** – Speedtest benchmark added
- **v0.1.15** – Model Comparison feature added

## Preparation

Before running the Evaluator for Model Benchmark, please ensure that you have the following:

- A served model in Supervisely (currently available for object detection and instance segmentation models)
- You have prepared a Ground Truth project with the appropriate annotations (classes should be the same as in the model)

## How To Run

**Step 1:** Open and launch the app from the Supervisely Ecosystem.

**Step 2**:

- _Model Evaluation_:

  **Step 2.1:** Select the Ground Truth project and the model you want to evaluate.

  **Step 2.2:** Press the “Evaluate” button to start the evaluation process. After the evaluation is complete, you can find a link to the report in the app’s interface.

- _Model Comparison:_

  **Step 2.1:** Select the folder with the Ground Truth project name.

  **Step 2.1:** Select one or more evaluation folders with the model name.

  **Step 2.2:** Press the “Compare” button to start the comparison process. After the comparison is complete, you can find a link to the report in the app’s interface.
