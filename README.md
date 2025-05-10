# Cow-TrackingbyClassification


This repository is for the manuscript titled "Continuous Tracking Holstein Dairy Cows in a Free-Stall Barn Using a Tracking-by-Classification Method", submitted to ***Biosystems Engineering***. 

Please note that the data is currently unavailable but will be provided once the manuscript is accepted.

The code is built using PyTorch and has been tested on Ubuntu 22.04 environment (Check: `Python3.12.4`, `Ultralytics8.2.58`, `PyTorch2.3.1`, `CUDA11.8`, `cuDNN8.7`) with 2*RTX 4090 GPUs. 

## Contents
- [Cow-TrackingbyClassification](#cow-trackingbyclassification)
  - [Introduction](#introduction)
  - [Train](#train)
    - [Object detector](#object-detector)
    - [Classifier](#classifier)
  - [Test](#test)
    - [Run the tracking model](#run-the-tracking-model)
    - [Evaluate tracking performance](#evaluate-tracking-performance)
  - [Results](#results)
    - [Object detector](#object-detector-1)
    - [Classifier](#classifier-1)
    - [Tracking model](#tracking-model)
  - [Demos](#demos)
    - [Daytime video demos](#daytime-video-demos)
    - [Nighttime video demos](#nighttime-video-demos)
  - [Acknowledgements](#acknowledgements)

## Introduction
Monitoring individual dairy cattle for behavior, health status, and production information is crucial for effective farm management and ensuring animal welfare. Continuous tracking and individual animal identification (ID) are essential to achieve these goals. Recently, interest in contactless computer vision (CV) approaches have gained significant interest. However, most existing studies have focused on short-term tracking, where animals remained within the field of view (FOV). In commercial free-stall barns, cows frequently move in and out of the FOV due to complex environment and daily management activities, making continuous tracking more challenging. This study aimed to develop a CV-based method for identification and continuous tracking of Holstein cows in a free-stall barn, addressing the challenge of reidentifying cows as they re-enter the FOV. The proposed method first used an object detector based on YOLOv8 to localize cows in each frame, followed by a classification step to predict the ID probabilities for each animal. Associations between detections and IDs were determined through two rounds of Hungarian matching to maximize classification probabilities. The method was trained and tested on video footages from a pen containing 13 Holstein cows. It outperformed both SORT and DeepSORT in Identification F1 score (IDF1) and Higher Order Tracking Accuracy (HOTA) across videos of varying durations, achieving an average IDF1 of 0.9458 and HOTA of 0.9287. These results demonstrate its strong potential for accurately maintaining IDs to enable long-term tracking of Holstein cows within complex environments, supporting its integration into behavior monitoring systems to enhance animal health and welfare.



## Train
### Object detector
To train the **object detector**, use the script `train_yolov8.py`.
```sh
python train_yolov8.py
```
Note that the number of epochs is hardcoded inside the script. The file `det.yaml` contains the path to the training and validation sets. 

The **dataset** should have the following **file structure**, as defined in the YOLO documentation for **detection** (https://docs.ultralytics.com/datasets/detect/)
```
dataset  
└───train
│   └───images
│       │   image_1.png
│       │   image_2.png
│           ...
│   └───labels
│       │   image_1.txt
│       │   image_2.txt
│           ...
└───val
    └───images
    └───labels
```

**Labels** should follow the YOLO format, i.e., use normalized values with class set to 0 (cow).
```
class x_center y_center width height

e.g.,
0 0.165088 0.887326 0.251871 0.225347
0 0.207268 0.561160 0.306168 0.309833
```

### Classifier
To train the **classifier**, use the script `train_yolov8-cls.py`.
```sh
python train_yolov8-cls.py
```
Note that the paths to the training and validation sets, and the number of epochs are hardcoded.

The images of each class, i.e., individual, should follow the following **file structure**, as defined in the YOLO documentation for **classification** (https://docs.ultralytics.com/datasets/classify/)
```
dataset  
└───train
│   └───cow_0
│       │   image_1.png
│       │   image_2.png
│       │   image_3.png
│           ...
│   └───cow_1
│       ...
│   └───cow_13
└───val
    └───cow_0
    └───cow_1
        ...
    └───cow_13
```

## Test
### Run the tracking model
To run the **tracking model** on a video, use the script `inference_lap.py`.
```sh
python inference_lap.py \
  --video_input=input_video.mp4 \    # input video
  --video_output=output_video.mp4 \  # output video with predicted bounding boxes
  --output_file=hypothesis.txt \     # text files containing the predicted bounding boxes (hypothesis)
  --thresh_1=0.9 \                   # value for threshold 1
  --thresh_2=0.6                     # value for threshold 2
```

Our detection weights can be found [here](https://drive.google.com/drive/folders/1t62qPpnjR8Fas6SKNh-9ScoatH8SgETM?usp=drive_link) and our classification weights [here](https://drive.google.com/drive/folders/19qtdIplTlH0fuEhyLhQcSSEEtZN32A46?usp=sharing).

### Evaluate tracking performance
The python library `py-motmetrics` was used to compute MOT metrics (https://github.com/cheind/py-motmetrics).

To get the MOT metrics of a video, run `master_eval.sh`.
```sh
sh master_eval.sh \
  /path/to/video_labels \    # labels (GT) of the video
  /path/to/eval_folder \     # folder where to store temporary files
  /path/to/hypothesis.txt \  # hypothesis file generated during tracking
  /video_name                # give a name to the video (required)
```

Labels should follow the YOLO format, i.e., use normalized values. This time, the classes should be added and correspond to the identities.
```
class x_center y_center width height

e.g.,
0 0.165088 0.887326 0.251871 0.225347
1 0.207268 0.561160 0.306168 0.309833
```

## Results
### Object detector
Results obtained after 500 epochs.
| Model   | Epochs | Time |   P   |   R   |  AP50  | AP50-95 |  Size  |
|---------|--------|--------|-------|-------|--------|---------|--------|
| YOLOv8n |  500 | 33 m | 0.957 | 0.948 | 98.3%  | 86.3%   | 6.3MB  |
| YOLOv8s |  500 | 37 m | 0.946 | 0.956 | 98.6%  | 88.0%   | 22.5MB |
| YOLOv8m |  500 | 50 m | 0.953 | 0.965 | 98.5%  | 89.7%   | 52.0MB |
| YOLOv8l |  500 | 1 h 5 m  | 0.958 | 0.963 | 98.3%  | 89.8%   | 87.7MB |
| YOLOv8x |  500 | 1 h 28 m | 0.962 | 0.962 | 98.4%  | 89.8%   | 137.7MB|

### Classifier
Results obtained after 1,000 epochs.
| Model      | Epochs | Time    | Top1 | Top5 | Size  |
|------------|--------|---------|------|------|-------|
| YOLOv8n-cls| 1,000  | 3 h 7 m | 0.873| 0.988| 3.0MB |
| YOLOv8s-cls| 1,000  | 3 h 21 m  | 0.898| 0.992| 10.3MB|
| YOLOv8m-cls| 1,000  | 4 h 33 m  | 0.898| 0.993| 31.7MB|
| YOLOv8l-cls| 1,000  | 6 h 56 m  | 0.917| 0.993| 72.6MB|
| YOLOv8x-cls| 1,000  | 8 h 50m  | 0.916| 0.993| 112.5MB|

### Tracking model
Results obtained on 6 videos of different duration.
| # | Duration | Time  | Animals | Camera | Reappearances |   MOTA  |   IDF1  |   HOTA  | MT | ML | IDSW |
|---|----------|-------|---------|--------|---------------|---------|---------|---------|----|----|------|
| 1 | 00:05:00 | Day   |   13    |   1    |       0       | 0.8702  | 0.9305  | 0.9158  | 11 |  1 |  19  |
| 2 | 00:05:00 | Night |   10    |   2    |       4       | 0.9434  | 0.9712  | 0.9686  |  9 |  1 |  1   |
| 3 | 00:30:01 | Day   |   11    |   2    |      12       | 0.9771  | 0.9883  | 0.9828  | 10 |  1 |  34  |
| 4 | 00:30:00 | Night |   11    |   1    |      10       | 0.7725  | 0.8918  | 0.8701  | 10 |  0 |  280 |
| 5 | 01:00:01 | Day   |   11    |   2    |      29       | 0.9381  | 0.9666  | 0.9421  | 10 |  0 |  448 |
| 6 | 01:00:01 | Night |   13    |   1    |      32       | 0.8635  | 0.9261  | 0.8925  | 10 |  0 |  545 |

## Demos

### Daytime video demos

The full-length video demo (1-hour, daytime) can be accessed here: [Full Video Demo](https://drive.google.com/file/d/133jPBBrWl3O0cRraki9zeIU-703AIofC/view?usp=sharing).

The two short videos below demonstrate that cows with IDs #1 and #6 were correctly re-identified upon re-entering the field of view.
<div style="display: flex; justify-content: space-around; gap: 20px;">
  
  <!-- First GIF -->
  <img src="./Demos/DayClip1.gif" alt="DayClip1 GIF" width="45%">

  <!-- Second GIF -->
  <img src="./Demos/DayClip2.gif" alt="DayClip2 GIF" width="45%">
</div>

### Nighttime video demos

The full-length video demo (1-hour, nighttime) can be accessed here: [Full Video Demo](https://drive.google.com/file/d/1X3R91-qqkj-Dg4KkHnZfIZCjKWIsY3qW/view?usp=sharing).

The four short videos below show that the cow with ID #4 was correctly re-identified each time she re-entered the field of view.
| ![Night Clip 1](./Demos/NightClip1.gif) | ![Night Clip 2](./Demos/NightClip2.gif) |
|-----------------------------------------|-----------------------------------------|
| ![Night Clip 3](./Demos/NightClip3.gif) | ![Night Clip 4](./Demos/NightClip4.gif) |

## Acknowledgements

Thanks to the AgroVet-Strickhof team led by Dr. Melissa Terranova and to the barn staff managed by Mirjam Klöppel, for the help in data collection. Thanks to the labelers, Sumin Li, Xiaqi Ma, Yixin Zhang, Zhibo Zeng, and Xinjie Zhao.
