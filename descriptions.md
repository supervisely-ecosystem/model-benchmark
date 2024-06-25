# Overview

## Key Metrics

Here, we comprehensively assess the model's performance by presenting a broad set of metrics, including mAP (mean Average Precision), Precision, Recall, IoU (Intersection over Union), Classification Accuracy, Calibration Score, and Inference Speed.

- **Mean Average Precision (mAP)**: An overall measure of detection performance. mAP calculates the average precision across all classes at different levels of IoU thresholds and precision-recall trade-offs.
- **Precision**: Precision indicates how often the model's predictions are actually correct when it predicts an object. This calculates the ratio of correct detections to the total number of detections made by the model.
- **Recall**: Recall measures the model's ability to find all relevant objects in a dataset. This calculates the ratio of correct detections to the total number of instances in a dataset.
- **Intersection over Union (IoU)**: IoU measures how closely predicted bounding boxes match the actual (ground truth) bounding boxes. It is calculated as the area of overlap between the predicted bounding box and the ground truth bounding box, divided by the area of union of these bounding boxes.
- **Classification Accuracy**: We separately measure the model's capability to correctly classify objects. It’s calculated as a proportion of correctly classified objects among all matched detections. The predicted detection is considered matched if it overlaps a ground true bounding box with IoU higher than 0.5.
- **Calibration Score**: This score represents the consistency of predicted probabilities (or confidence scores) made by the model, evaluating how well the predicted probabilities align with actual outcomes. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.
- **Inference Speed**: The number of frames per second (FPS) the model can process, measured with a batch size of 1. The inference speed is important in applications, where real-time object detection is required. Additionally, slower models pour more GPU resources, so their inference cost is higher.



## Model Predictions

In this section you can visually assess the model performance through examples. This helps users better understand model capabilities and limitations, giving an intuitive grasp of prediction quality in different scenarios.

**(!) Info**

You can choose one of the sorting method:

- **Auto**: The algorithm is trying to gather a diverse set of images that illustrate the model's performance across various scenarios.
- **Least accurate**: Displays images where the model made more errors.
- **Most accurate**: Displays images where the model made fewer or no errors.
- **Dataset order**: Displays images in the original order of the dataset.



**Prediction Table**

The table helps you in finding samples with specific cases of interest. You can sort by parameters such as the number of predictions, or specific a metric, e.g, recall, then click on a row to view this image and predictions.

**(!) Info**

**Example**: you can sort by **FN (**False Negatives) in descending order to identify samples where the model failed to detect many objects.



## What is YOLOv8 model (collapse)?

Можно также добавить ссылку на наш блог пост, если есть

!\[blog post link\]

**О чем еще здесь можно рассказать:**

- Ключевая инфа о модели текстом: год, конференция, paper, гитхаб, какой скор на лидерборде от авторов, в каком сценарии эта модель была или есть SOTA и в каком году. Что-то ещё из того что писали про свою модель сами авторы, взять из ридми на гитхабе.
- Особенности модели, чем отличается от остальных, какую проблему решали авторы этой моделью.
- Для чего эта модель идеально подходит, какие сценарии использования? Возможно авторы проектировали модель под специальный use case, описать это. Например, YOLO подходит для real-time object detection, для real-time detection на видео.
- Историческая справка, как развивалась модель, прошлые версии.
- Краткий анализ метрик. На чем модель фейлит, а в чем хорошо предсказывает.

## Expert insights?

linkedin - ответ на вопрос когда применять когда нет, что лучше или хуже, что нужно учитывать. текст в свободной форме

## How To Use: Training, inference, evaluation loop (collapse)

Однотипная диаграмка, и небольшой текст со ссылками - Sly apps, inference notebooks, docker images, … небольшой раздел со ссылками на документацию (embeddings sampling, improvement loop, active learning, labeling jobs, model comparison, .… – стандартизован для всех моделей). какие-то модели будут частично интегрированы

Jupyter notebooks + python scripts + apps + videos + guides + …



# Detailed Metrics Analysis

**Note about confidence threshold:**

To calculate various metrics, we must set a _confidence threshold_, which also is necessary in deploying a model and applying it to any task. This hyperparameter significantly influences the results of metrics. To eliminate human bias in this process, we automate the determination of the confidence threshold. The threshold is selected based on the best _f1-score_ (guaranteed to give the best f1-score on the given dataset), ensuring a balanced trade-off between precision and recall.

**| F1-optimal confidence threshold = 0.35** (calculated for the given model and dataset)

Подробнее о том как мы считаем best confidence threshold: \[link\]



## Outcome Counts

This chart is used to evaluate the overall model performance by breaking down all predictions into True Positives (TP), False Positives (FP), and False Negatives (FN). This helps to visually assess the type of errors the model often encounters.


## Recall

This section measures the ability of the model to detect **all relevant instances in the dataset**. In other words, this answers the question: “Of all instances in the dataset, how many of them is the model managed to find out?”

To measure this, we calculate **Recall.** Recall counts errors, when the model does not detect an object that actually is present in a dataset and should be detected. Recall is calculated as the portion of correct predictions (true positives) over all instances in the dataset (true positives + false negatives).

More information: \[link\]:

Там рассказать что recall считается отдельно по всем классам и для каждого IoU threshold, а затем берется среднее по всему этому.

Recall is the portion of **correct** predictions (true positives) over all actual instances in the dataset (true positives + false negative). A recall of 0.7 indicates that the model identifies 70% of all actual positives in the dataset.

Recall is averaged across all classes and IoU thresholds \[0.50:0.95\].

**| Recall** (?) **\= 0.51** _(green-red color scale)_

The model correctly found **4615 of 9012** total instances in the dataset.

**Per-class Recall**

This chart further analyzes Recall, breaking it down to each class in separate.

**(!) Info**

Since the overall recall is calculated as an average across all classes, we provide a chart showing the recall for each individual class. This illustrates how much each class contributes to the overall recall.

_Bars in the chart are sorted by F1-score to keep a unified order of classes between different charts._



## Precision

This section measures the accuracy of all predictions made by the model. In other words, this answers the question: “Of all predictions made by the model, how many of them are actually correct?”.

To measure this, we calculate **Precision.** Precision counts errors, when the model predicts an object (bounding box), but the image has no objects in this place (or it has another class than the model predicted). Precision is calculated as a portion of correct predictions (true positives) over all model’s predictions (true positives + false positives).

More information: \[link\]:

(?) - Precision is the portion of **correct** predictions (true positives) over all model’s predictions (true positives + false positives). A precision of 0.8 means that 80% of the instances that the model predicted as positive (e.g., detected objects) are actually positive (correct detections).

Precision is averaged across all classes and IoU thresholds \[0.50:0.95\].

**| Precision (?) = 0.66**

The model correctly predicted **5012 of 6061** predictions made by the model in total.

**Per-class Precision**

This chart further analyzes Precision, breaking it down to each class in separate.

**(!) Info**

Since the overall precision is computed as an average across all classes, we provide a chart showing the precision for each class individually. This illustrates how much each class contributes to the overall precision.

_Bars in the chart are sorted by F1-score to keep a unified order of classes between different charts._



## Recall vs. Precision

This section compares Precision and Recall on a common graph, identifying **disbalance** between these two.

_Bars in the chart are sorted by F1-score to keep a unified order of classes between different charts._



## Precision-Recall Curve

Precision-Recall curve is an overall performance indicator. It helps to visually assess both precision and recall for all predictions made by the model on the whole dataset. This gives you an understanding of how precision changes as you attempt to increase recall, providing a view of **trade-offs between precision and recall**. Ideally, a high-quality model will maintain strong precision as recall increases. This means that as you move from left to right on the curve, there should not be a significant drop in precision. Such a model is capable of finding many relevant instances, maintaining a high level of precision.

🔽(Collapse) **About Trade-offs between precision and recall**

A system with high recall but low precision returns many results, but most of its predictions are incorrect or redundant (false positive). A system with high precision but low recall is just the opposite, returning very few results, most of its predictions are correct. An ideal system with high precision and high recall will return many results, with all results predicted correctly.

🔽(Collapse) **What is PR curve?**

Imagine you sort all the predictions by their confidence scores from highest to lowest and write it down in a table. As you iterate over each sorted prediction, you classify it as a true positive (TP) or a false positive (FP). For each prediction, you then calculate the cumulative precision and recall so far. Each prediction is plotted as a point on a graph, with recall on the x-axis and precision on the y-axis. Now you have a plot very similar to the PR-curve, but it appears as a zig-zag curve due to variations as you move from one prediction to the next.

**Forming the Actual PR Curve**: The true PR curve is derived by plotting only the maximum precision value for each recall level across all thresholds. This means you connect only the highest points of precision for each segment of recall, smoothing out the zig-zags and forming a curve that typically slopes downward as recall increases.

**mAP = 0.51**



### Precision-Recall curve by Class

In this plot, you can evaluate PR curve for each class individually.



## Classification Accuracy

This section investigates cases where the model correctly localizes a bounding box, but predicts a wrong class label. For example, the model might confuse a motorbike with a bicycle. In this case, the model correctly identified that the object is present on the image, but assigned a wrong label to it.

To quantify it, we calculate **Classification accuracy**. This is a portion of correctly classified objects to the total number of correctly localized objects ?-_(the object is localized correctly if the IoU between a prediction and a ground truth box is more than 0.5)_. In other words, if the model correctly found that an object is present on the image, how often it assigns a correct label to it?

**| Classification Accuracy: 0.96**

The model correctly classified **52** predictions **of 54** total predictions, that are matched to the ground truth.

### Confusion Matrix

Confusion matrix helps to find the number of confusions between different classes made by the model. Each row of the matrix represents the instances in a ground truth class, while each column represents the instances in a predicted class. The diagonal elements represent the number of correct predictions for each class (True Positives), and the off-diagonal elements show misclassifications.



**Mini Confusion Matrix**

- skip for now


### Frequently Confused Classes

This chart displays the most frequently confused pairs of classes. In general, it finds out which classes visually seem very similar to the model.

The chart calculates the **probability of confusion** between different pairs of classes. For instance, if the probability of confusion for the pair “car - truck” is 0.15, this means that when the model predicts either “car” or “truck”, there is a 15% chance that the model might mistakenly predict one instead of the other.

The measure is class-symmetric, meaning that the probability of confusing a car with a truck is equal to the probability of confusing a truck with a car.

_switch: Probability / Amount_



## Localization Accuracy (IoU)

This section measures how closely predicted bounding boxes generated by the model are aligned with the actual (ground truth) bounding boxes.

To measure it, we calculate the **Intersection over Union (IoU).** Intuitively, the higher the IoU, the closer two bounding boxes are. IoU is calculated by dividing the **area of overlap** between the predicted bounding box and the ground truth bounding box by the **area of union** of these two boxes.

**| Avg. IoU** (?) **= 0.86**

### IoU Distribution

This histogram represents the distribution of IoU scores between all predictions and their matched ground truth objects. This gives you a sense of how well the model aligns bounding boxes. Ideally, if the model aligns boxes very well, rightmost bars (from 0.9 to 1.0 IoU) should be much higher than others.



## Calibration Score

This section analyzes confidence scores (or predicted probabilities) that the model generates for every predicted bounding box.

🔽(Collapse) **What is calibration?**

In some applications, it's crucial for a model not only to make accurate predictions but also to provide reliable **confidence levels**. A well-calibrated model aligns its confidence scores with the actual likelihood of predictions being correct. For example, if a model claims 90% confidence for predictions but they are correct only half the time, it is **overconfident**. Conversely, **underconfidence** occurs when a model assigns lower confidence scores than the actual likelihood of its predictions. In the context of autonomous driving, this might cause a vehicle to brake or slow down too frequently, reducing travel efficiency and potentially causing traffic issues.

To evaluate the calibration, we draw a **Reliability Diagram** and calculate **Expected Calibration Error** (ECE) and **Maximum Calibration Error** (MCE).

### Reliability Diagram

Reliability diagram, also known as a Calibration curve, helps in understanding whether the confidence scores of detections accurately represent the true probability of a correct detection. A well-calibrated model means that when it predicts a detection with, say, 80% confidence, approximately 80% of those predictions should actually be correct.

🔽(Collapse) **How to interpret the Calibration curve:**

1. **The curve is above the Ideal Line (Underconfidence):** If the calibration curve is consistently above the ideal line, this indicates underconfidence. The model’s predictions are more correct than the confidence scores suggest. For example, if the model predicts a detection with 70% confidence but, empirically, 90% of such detections are correct, the model is underconfident.
2. **The curve is below the Ideal Line (Overconfidence):** If the calibration curve is below the ideal line, the model exhibits overconfidence. This means it is too sure of its predictions. For instance, if the model predicts with 80% confidence but only 60% of these predictions are correct, it is overconfident.

To quantify the calibration score, we calculate **Expected Calibration Error (ECE).** Intuitively, ECE can be viewed as a deviation of the Calibration curve from the Perfectly calibrated line. When ECE is high, we can not trust predicted probabilities so much.

**| Expected Calibration Error (ECE)** (?) **= 0.15**

## Confidence Score Profile

This section is going deeper in analyzing confidence scores. It gives you an intuition about how these scores are distributed and helps to find the best confidence threshold suitable for your task or application.

**Confidence Score Profile**

This chart provides a comprehensive view about predicted confidence scores. It is used to determine an optimal _confidence threshold_ based on your requirements.

This plot shows you what the metrics will be if you choose a specific confidence threshold. For example, if you set the threshold to 0.32, you can see on the plot what the precision, recall and f1-score will be for this threshold.

🔽(Collapse) **How to plot Confidence score Profile?**

First, we sort all predictions by confidence scores from highest to lowest. As we iterate over each prediction we calculate the cumulative precision, recall and f1-score so far. Each prediction is plotted as a point on a graph, with a confidence score on the x-axis and one of three metrics on the y-axis (precision, recall, f1-score).

**To find an optimal threshold**, you can select the confidence score under the maximum of the f1-score line. This f1-optimal threshold ensures the balance between precision and recall. You can select a threshold according to your desired trade-offs.

**F1-optimal confidence threshold = _0.263_**



### Confidence Distribution

This graph helps to assess whether high confidence scores correlate with correct detections (True Positives) and whether low confidence scores are mostly associated with incorrect detections (False Positives).

Additionally, it provides a view of how predicted probabilities are distributed. Whether the model skews probabilities to lower or higher values, leading to imbalance?

Ideally, the histogram for TP predictions should have higher confidence, indicating that the model is sure about its correct predictions, and the FP predictions should have very low confidence, or not present at all.

В описании приложить схематично идеальный график. Объяснения

_Сделать stacked bar chart._



## Class Comparison

This section analyzes the model's performance for all classes in a common plot. It discovers which classes the model identifies correctly, and which ones it often gets wrong.

### Average Precision by Class

A quick visual comparison of the model performance across all classes. Each axis in the chart represents a different class, and the distance to the center indicates the Average Precision for that class.

### Outcome Counts by Class

This chart breaks down all predictions into True Positives (TP), False Positives (FP), and False Negatives (FN) by classes. This helps to visually assess the type of errors the model often encounters for each class.

**Normalization:** by default the normalization is used for better intraclass comparison. The total outcome counts are divided by the number of ground truth instances of the corresponding class. This is useful, because the sum of TP + FN always gives 1.0, representing all ground truth instances for a class, that gives a visual understanding of what portion of instances the model detected. So, if a green bar (TP outcomes) reaches the 1.0, this means the model is managed to predict all objects for the class. Everything that is higher than 1.0 corresponds to False Positives, i.e, redundant predictions. You can turn off the normalization switching to absolute values.

_Bars in the chart are sorted by F1-score to keep a unified order of classes between different charts._



_switch to absolute values:_



## Inference speed

We evaluate the inference speed in two scenarios: real-time inference (batch size is 1), and batch processing. We also run the model in optimized runtime environments, such as ONNX Runtime and Tensor RT, using consistent hardware. This approach provides a fair comparison of model efficiency and speed. To assess the inference speed we run the model forward 100 times and average it.

\[**Methodology**\] 🔽 (collapsable)

Setting 1: **Real-time processing**

We measure the time spent processing each image individually by setting batch size to 1. This simulates real-time data processing conditions, such as those encountered in video streams, ensuring the model performs effectively in scenarios where data is processed frame by frame.

Setting 2: **Parallel processing**

To evaluate the model’s efficiency in parallel processing, we measure the processing speed with batch size of 8 and 16. This helps us understand how well the model scales when processing multiple images simultaneously, which is crucial for applications requiring high throughput.

Setting 3: **Optimized runtime**

We run the model in various runtime environments, including **ONNX Runtime** and **TensorRT**. This is important because python code can be suboptimal. These runtimes often provide significant performance improvements.

**Consistent hardware for fair comparison**

To ensure a fair comparison, we use a single hardware setup, specifically an NVIDIA RTX 3060 GPU.

**Inference details**

We divide the inference process into three stages: **preprocess, inference,** and **postprocess** to provide insights into where optimization efforts should be focused. Additionally, it gives us another verification level to ensure that time is measured correctly for each model.

**Preprocess**: The stage where images are prepared for input into the model. This includes image reading, resizing, and any necessary transformations.

**Inference**: The main computation phase where the _forward_ pass of the model is running. **Note:** we include not only the forward pass, but also modules like NMS (Non-Maximum Suppression), decoding module, and everything that is done to get a **meaningful** prediction.

**Postprocess**: This stage includes tasks such as resizing output masks, aligning predictions with the input image, converting bounding boxes into a specific format or filtering out low-confidence detections.

