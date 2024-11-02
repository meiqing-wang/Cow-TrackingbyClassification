# Cow-TrackingbyClassification


This repository is for the manuscript titled "Continuous Tracking Holstein Dairy Cows in a Free-Stall Barn Using a Tracking-by-Classification Method," submitted to ***Computers and Electronics in Agriculture***. Please note that the data is currently unavailable but will be provided once the manuscript is accepted.

The code is built using PyTorch and has been tested on Ubuntu 22.04 environment (Check: Python3.12.4, Ultralytics8.2.58, PyTorch2.3.1, CUDA11.8, cuDNN8.7) with 2*RTX 4090 GPUs. 

## Contents
- [Cow-TrackingbyClassification](#cow-trackingbyclassification)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Train](#train)
    - [Object detector](#object-detector)
    - [Classifier](#classifier)
  - [Test](#test)
    - [Running the inference](#running-the-inference)
  - [Results](#results)
  - [Demos](#demos)
    - [Demos of the video during the day](#demos-of-the-video-during-the-day)
  - [Acknowledgements](#acknowledgements)

## Introduction
Monitoring individual dairy cattle for behavior, health status, and production information is crucial for effective farm management and ensuring animal welfare. Continuous tracking and individual animal identification (ID) are essential to achieve these goals. Recently, interest in contactless computer vision (CV) approaches have gained significant interest. However, most existing studies have focused on short-term tracking, where animals remained within the field of view (FOV). In commercial free-stall barns, cows frequently move in and out of the FOV due to complex environment and daily management activities, making continuous tracking more challenging. This study aimed to develop a CV-based method for identification and continuous tracking of Holstein cows in a free-stall barn, addressing the challenge of reidentifying cows as they re-enter the FOV. The proposed method first used an object detector based on YOLOv8 to localize cows in each frame, followed by a classification step to predict the ID probabilities for each animal. Associations between detections and IDs were determined through two rounds of Hungarian matching to maximize classification probabilities. The method was trained and tested on video footages from a pen containing 13 Holstein cows. It outperformed both SORT and DeepSORT in Identification F1 score (IDF1) and Higher Order Tracking Accuracy (HOTA) across videos of varying durations, achieving an average IDF1 of 0.9458 and HOTA of 0.9287. These results demonstrate its strong potential for accurately maintaining IDs to enable long-term tracking of Holstein cows within complex environments, supporting its integration into behavior monitoring systems to enhance animal health and welfare.



## Train
### Object detector
To train the object detector, the following script is used for all model sizes.
```sh
python train_yolov8.py
```
Note that the number of epochs is hardcoded inside the script. The file `det.yaml` contains the path to the training and validation sets. 

After 500 epochs, the following results were obtained.
| Model    | #Epochs | Training time |    P   |    R   |  AP50  | AP50-95 |  Size   |
|----------|---------|---------------|--------|--------|--------|---------|---------|
| YOLOv8n  |   500   |    33 min     | 0.957  | 0.948  | 98.3%  | 86.3%   |  6.3 MB |
| YOLOv8s  |   500   |    37 min     | 0.946  | 0.956  | 98.6%  | 88.0%   | 22.5 MB |
| YOLOv8m  |   500   |    50 min     | 0.953  | 0.965  | 98.5%  | 89.7%   | 52.0 MB |
| YOLOv8l  |   500   |  1 h 5 min    | 0.958  | 0.963  | 98.3%  | 89.8%   | 87.7 MB |
| YOLOv8x  |   500   | 1 h 28 min    | 0.962  | 0.962  | 98.4%  | 89.8%   | 137.7 MB |

### Classifier
To train the classifier, the following script is used for all model sizes:
```sh
python train_yolov8-cls.py
```
Note that the paths to the training and validation sets, and the number of epochs are hardcoded.

After 1,000 epochs, the following results were obtained.
<table style="font-size: 0.8em;">
  <tr>
    <th>Model</th>
    <th>Epochs</th>
    <th>Time</th>
    <th>Top1</th>
    <th>Top5</th>
    <th>Size</th>
  </tr>
  <tr>
    <td>YOLOv8n-cls</td>
    <td>1,000</td>
    <td>3h 7m</td>
    <td>0.873</td>
    <td>0.988</td>
    <td>3.0MB</td>
  </tr>
  <tr>
    <td>YOLOv8s-cls</td>
    <td>1,000</td>
    <td>3h 21m</td>
    <td>0.898</td>
    <td>0.992</td>
    <td>10.3MB</td>
  </tr>
  <tr>
    <td>YOLOv8m-cls</td>
    <td>1,000</td>
    <td>4h 33m</td>
    <td>0.898</td>
    <td>0.993</td>
    <td>31.7MB</td>
  </tr>
  <tr>
    <td>YOLOv8l-cls</td>
    <td>1,000</td>
    <td>6h 56m</td>
    <td>0.917</td>
    <td>0.993</td>
    <td>72.6MB</td>
  </tr>
  <tr>
    <td>YOLOv8x-cls</td>
    <td>1,000</td>
    <td>8h 50m</td>
    <td>0.916</td>
    <td>0.993</td>
    <td>112.5MB</td>
  </tr>
</table>


## Test
### Running the inference


## Results

## Demos

### Demos of the video during the day

The below two short videos show the cows with #1 and #6 were re-indentified correctly when they reappearred in the FOV.

<div style="display: flex; justify-content: space-around; gap: 20px;">
  <!-- First Video -->
  <video controls width="45%">
    <source src="https://raw.githubusercontent.com/meiqing-wang/Cow-TrackingbyClassification/main/Demos/DayClip1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <!-- Second Video -->
  <video controls width="45%">
    <source src="https://raw.githubusercontent.com/meiqing-wang/Cow-TrackingbyClassification/main/Demos/DayClip2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Acknowledgements