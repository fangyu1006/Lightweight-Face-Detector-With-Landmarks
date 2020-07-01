# Lightwight Face Detector

## Contents
* [Introduction](#introduction)
  * [Functions](#functions)
  * [Test environment](#test environment)
* [Evaluation](#evaluation)
  * [Widerface](#Widerface)
  * [Parameter and flop](#Parameter and flop)
  * [Speed](#speed)
* [How to use](#how to use)
  * [Installation](#installation)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Convertor](#convertor)
  * [C++_inference](#c++_inference)
* [References](#references)


## Introduction
This project provides a serias of lightweight face detectors with landmarks which can be deployed on mobile devices.
 - Modify the anchor size of [Retinaface-mobile0.25](https://github.com/biubug6/Pytorch_Retinaface)
 - Add landmarks estimation to [Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) 
### Functions
 - Train/test/evaluation/ncnn/tensorflow/tflite/C++ inference of Retinaface-mobile0.25
 - Train/test/evaluation/ncnn/tensorflow/tflite/C++ inference of Face-Detector-1MB slim and RFB version
 - Add 5 landmarks estimation to Face-Detector-1MB
 - Support the inference using pytorch/ncnn/tensorflow/tflite

### Test environment
- Ubuntu18.04
- Python3.7
- Pytorch1.2
- CUDA10.0 + CUDNN7.5

## Evaluation
### Widerface

 - Evaluation result on wider face val (input image size: **320*240**）
 
 <ethods|Easy|Medium|Hard
------|--------|----------|--------
libfacedetection v1（caffe）|0.65 |0.5       |0.233
libfacedetection v2（caffe）|0.714 |0.585       |0.306
version-slim(origin)|0.765     |0.662       |0.385
version-RFB(origin)|0.784     |0.688       |**0.418**
version-slim(our)|0.795     |0.683       |0.34.5
version-RFB(our)|**0.814**     |**0.710**       |0.363
Retinaface-Mobilenet-0.25(our)  |0.811|0.697|0.376

 - Evaluation result on wider face val (input image size: **640*480**)

Methods|Easy|Medium|Hard 
------|--------|----------|--------
libfacedetection v1（caffe）|0.741 |0.683       |0.421
libfacedetection v2（caffe）|0.773 |0.718       |0.485
version-slim(origin)|0.757     |0.721       |0.511
version-RFB(origin)|0.851     |0.81       |0.541
version-slim(our)|0.850     |0.808       |0.595
version-RFB(our)|0.865    |0.828       |0.622
Retinaface-Mobilenet-0.25(our)  |**0.873**|**0.836**|**0.638**


### Parameter and flop

Methods|parameter(M)|flop(M) 
------|--------|----------
version-slim(our)|0.343     |98.793
version-RFB(our)|0.359    |118.435
Retinaface-Mobilenet-0.25(our)  |0.426|193.921

### Speed
 - Test speed on [RK3399](http://opensource.rock-chips.com/wiki_RK3399) using tflite format

Input image size: **320*240**

Methods|Speed(ms)
------|-------
MTCNN|325
version-slim(our)|82 
version-RFB(our)|94
Retinaface-Mobilenet-0.25(our)|103

Input image size: **640*480**
Methods|Speed(ms)
------|-------
MTCNN|420
version-slim(our)|342
version-RFB(our)|380
Retinaface-Mobilenet-0.25(our)|438

## How to use
### Installation
##### Clone and install
1. git clone this project

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. The dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

2. We provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training

1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 or 
  CUDA_VISIBLE_DEVICES=0 python train.py --network slim or
  CUDA_VISIBLE_DEVICES=0 python train.py --network RFB
  ```

If you don't want to train, we also provide a trained model on ./weights
  ```Shell
  mobilenet0.25_Final.pth 
  RBF_Final.pth
  slim_Final.pth
  ```
## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or slim or RFB
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)


## Convertor
 - Convert pytorch to onnx/ncnn/caffe/tensorflow/tflite
 - Converting script on convertor folder

1. Generate onnx file
```Shell
python convert_to_onnx.py --trained_model weight_file --network mobile0.25 or slim or RFB
```
2. Onnx file change to ncnn(*.param and *.param)
```Shell
cp *.onnx ./Detector_cpp/Face_Detector_ncnn/tools
cd ./Detector_cpp/Face_Detector_ncnn/tools
./onnx2ncnn face.param face.bin
```
3. Simplify onnx file
```Shell
pip install onnx-simplifier
python-m onnxsim input_onnx_model output_onnx_model
```
4. Convert to Caffe 
```Shell
python convertCaffe.py
```
5. Convert to Tensorflow
```Shell
python demoCaffe.py
python froze_graph_from_ckpt.py
```
6. Convert to Tensorflow lite
```Shell
python convert_to_tflite.py
```

## C++_inference 
 - C++ inference code for ncnn/tf/tflite on Detector_cpp folder
1. Build Project(set opencv path in CmakeList.txt)
```Shell
mkdir build
cd build
cmake ..
make -j4
```
2. run
```Shell
./FaceDetector *.jpg
./FaceDetectorFolder [folder path]
```

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [Retinaface (pytorch)](https://github.com/biubug6/Pytorch_Retinaface)
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
