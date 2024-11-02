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
To train the object detector, use the script `train_yolov8.py`.
```sh
python train_yolov8.py
```
Note that the number of epochs is hardcoded inside the script. The file `det.yaml` contains the path to the training and validation sets. 

After 500 epochs, the following results were obtained.
| Model   | Epochs | Time   |   P   |   R   |  AP50  | AP50-95 |  Size  |
|---------|--------|--------|-------|-------|--------|---------|--------|
| YOLOv8n |  500   | 33m    | 0.957 | 0.948 | 98.3%  | 86.3%   | 6.3MB  |
| YOLOv8s |  500   | 37m    | 0.946 | 0.956 | 98.6%  | 88.0%   | 22.5MB |
| YOLOv8m |  500   | 50m    | 0.953 | 0.965 | 98.5%  | 89.7%   | 52.0MB |
| YOLOv8l |  500   | 1h 5m  | 0.958 | 0.963 | 98.3%  | 89.8%   | 87.7MB |
| YOLOv8x |  500   | 1h 28m | 0.962 | 0.962 | 98.4%  | 89.8%   | 137.7MB|


### Classifier
To train the classifier, use the script `train_yolov8-cls.py`.
```sh
python train_yolov8-cls.py
```
Note that the paths to the training and validation sets, and the number of epochs are hardcoded.

After 1,000 epochs, the following results were obtained.
| Model      | Epochs | Time    | Top1 | Top5 | Size  |
|------------|--------|---------|------|------|-------|
| YOLOv8n-cls| 1,000  | 3h 7m   | 0.873| 0.988| 3.0MB |
| YOLOv8s-cls| 1,000  | 3h 21m  | 0.898| 0.992| 10.3MB|
| YOLOv8m-cls| 1,000  | 4h 33m  | 0.898| 0.993| 31.7MB|
| YOLOv8l-cls| 1,000  | 6h 56m  | 0.917| 0.993| 72.6MB|
| YOLOv8x-cls| 1,000  | 8h 50m  | 0.916| 0.993| 112.5MB|


## Test
### Running the inference
To run the model on a custom video, use the script `inference_lap.py`.
```sh
python inference_lap.py \
--video_input=input_video.mp4 \    # input video
--video_output=output_video.mp4 \  # output video with predicted bounding boxes
--output_file=hypotheses.txt \     # text files containing the predicted bounding boxes (hypothesis)
--thresh_1=0.9 \                   # value for threshold 1
--thresh_2=0.6                     # value for threshold 2
```

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