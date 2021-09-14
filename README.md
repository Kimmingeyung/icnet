# icnet
-------------------------------------------------------------------------------------------------------
This repository contains the TensorFlow implementation for the following paper

ICNet for RT Semantic Segmentation (ECCV_2018)

if you use this code for your research, please consider citing:

  <pre><code>@inProceedings{
      title={ICNet for Real-Time Semantic Segmentation on High-Resolution Images},
      author={Hengshuang Zhao1, Xiaojuan Qi1, Xiaoyong Shen2, Jianping Shi3, Jiaya Jia1,2},
      booktitle={ECCV},
      year={2018}
   }</code></pre>
  

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
I used the cityscapes-dataset
When using the provided data make sure to respect the cityscapes-dataset license. 

+  https://www.cityscapes-dataset.com/
+  The cityscapes dataset for semantic urban understanding, arXiv: 1604.01685

Below is the complete set of training data. Download it into the data/ forder 

+ https://drive.google.com/open?
+ [구글드라이브](https://drive.google.com/drive/folders/1qWLE0xiz51r5drrGwwFTFjmwste0cuEH-)

+ data directory:
<pre><code>
  cityscapes
      |-------leftImg8bit---train
      |-------gtFine -------train
</code></pre>

# Training
-------------------------------------------------------------------------------------------------------
  <pre><code> python train.py </code></pre>

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




