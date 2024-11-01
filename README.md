# Cow-TrackingbyClassification


This repository is for the manuscript titled "Continuous Tracking Holstein Dairy Cows in a Free-Stall Barn Using a Tracking-by-Classification Method," submitted to ***Computers and Electronics in Agriculture***. Please note that the data is currently unavailable but will be provided once the manuscript is accepted.

The code is built using PyTorch (check) and has been tested on Ubuntu 22.04 environment (Check: Python3.6, PyTorch_0.4.0, CUDA12.4, cuDNN5.1) with 2*RTX 4090 GPUs. 

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Demos](#Demos)
6. [Acknowledgements](#acknowledgements)

## Introduction
Monitoring individual dairy cattle for behavior, health status, and production information is crucial for effective farm management and ensuring animal welfare. Continuous tracking and individual animal identification (ID) are essential to achieve these goals. Recently, interest in contactless computer vision (CV) approaches have gained significant interest. However, most existing studies have focused on short-term tracking, where animals remained within the field of view (FOV). In commercial free-stall barns, cows frequently move in and out of the FOV due to complex environment and daily management activities, making continuous tracking more challenging. This study aimed to develop a CV-based method for identification and continuous tracking of Holstein cows in a free-stall barn, addressing the challenge of reidentifying cows as they re-enter the FOV. The proposed method first used an object detector based on YOLOv8 to localize cows in each frame, followed by a classification step to predict the ID probabilities for each animal. Associations between detections and IDs were determined through two rounds of Hungarian matching to maximize classification probabilities. The method was trained and tested on video footages from a pen containing 13 Holstein cows. It outperformed both SORT and DeepSORT in Identification F1 score (IDF1) and Higher Order Tracking Accuracy (HOTA) across videos of varying durations, achieving an average IDF1 of 0.9458 and HOTA of 0.9287. These results demonstrate its strong potential for accurately maintaining IDs to enable long-term tracking of Holstein cows within complex environments, supporting its integration into behavior monitoring systems to enhance animal health and welfare.



## Train


## Test


## Results

## Demos

### Demos of the video during the day

The below two short videos show the cows with #1 and #6 were re-indentified correctly when they reappearred in the FOV.

<div style="display: flex; justify-content: space-around; gap: 20px;">
  <!-- First Video -->
  <video controls width="45%">
    <source src="https://github.com/meiqing-wang/Cow-TrackingbyClassification/blob/main/Demos/DayClip1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <!-- Second Video -->
  <video controls width="45%">
    <source src="https://github.com/meiqing-wang/Cow-TrackingbyClassification/blob/main/Demos/DayClip12.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


## Acknowledgements