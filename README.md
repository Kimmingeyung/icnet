# icnet
-------------------------------------------------------------------------------------------------------
This repository contains the TensorFlow implementation for the following paper

ICNet for RT Semantic Segmentation (ECCV_2018)

if you use this code for your research, please consider citing:

@inProceedings{
  title={ICNet for Real-Time Semantic Segmentation on High-Resolution Images},
  author={Hengshuang Zhao1, Xiaojuan Qi1, Xiaoyong Shen2, Jianping Shi3, Jiaya Jia1,2},
  booktitle={ECCV},
  year={2018}
}

# Project Page
-------------------------------------------------------------------------------------------------------
# Dependencies
-------------------------------------------------------------------------------------------------------
Requirements:
<ul>
  <li>python 3.6 with Numpy and opencv-python </li>
  <li>tensorflow (version 1.0+) </li>
  <li>etc</li>
</ul>

My code has been tested with python 3.6, tensorflow 1.13.0, CUDA 11.3 on Window10 


# Runing the demo
-------------------------------------------------------------------------------------------------------
# Installation
-------------------------------------------------------------------------------------------------------
# Dataset
-------------------------------------------------------------------------------------------------------
# Training
-------------------------------------------------------------------------------------------------------
  <code> python train.py </code>

You can change the training data, learning rate and other parameters by editing train.py
The total number of training epochs is 100 ; learning rate is initialized as 1e-3
and training epoch of 100 with linear decay after 50 epoches

# Evaluation
-------------------------------------------------------------------------------------------------------
# Statement
-------------------------------------------------------------------------------------------------------
# Contact 
-------------------------------------------------------------------------------------------------------
Kim Min Geyung 


# License 
-------------------------------------------------------------------------------------------------------




