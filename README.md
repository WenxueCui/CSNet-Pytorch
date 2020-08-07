# CSNet-Pytorch

Pytorch code for paper 
* "Deep Networks for Compressed Image Sensing" ICME2017

* "Image Compressed Sensing Using Convolutional Neural Network" TIP2020

## Requirements and Dependencies

* Ubuntu 16.04 CUDA 10.0
* Python3 （Testing in Python3.5）
* Pytorch 1.1.0   Torchvision 0.2.2

## Details of Implementations

In our code, two model version are included:

* simple version of CSNet (Similar with paer)
* Enhanced version of CSNet (skip connection + resudial learning)

## How to Run

### Training CSNet

Editing the path of training data in file `train.py`.

* For CSNet training in terms of subrate=0.1:

`Python train.py --sub_rate=0.1 --block_size=32`

### Testing CSNet

Editing the path of trained model in file `test.py`.

* For CSNet testing in terms of subrate=0.1:

`python test.py --sub_rate=0.1 --block_size=32`

## CSNet results
### Subjective results

![image](https://github.com/WenxueCui/CSNet-Pytorch/raw/master/images/results.jpg)

### Objective results
![image](https://github.com/WenxueCui/CSNet-Pytorch/raw/master/images/table.jpg)
