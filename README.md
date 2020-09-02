# CSNet-Pytorch

Pytorch code for paper 

* "Deep Networks for Compressed Image Sensing" ICME2017

* "Image Compressed Sensing Using Convolutional Neural Network" TIP2019

## Requirements and Dependencies

* Ubuntu 16.04 CUDA 10.0
* Python3 （Testing in Python3.5）
* Pytorch 1.1.0   
* Torchvision 0.2.2

## Details of Implementations

In our code, two model version are included:

* simple version of CSNet (Similar with paper ICME2017)
* Enhanced version of CSNet (local skip connection + global skip connection + resudial learning)

## How to Run

### Training CSNet
* Preparing the dataset for training

* Editing the path of training data in file `train.py`.

* For CSNet training in terms of subrate=0.1:

```python train.py --sub_rate=0.1 --block_size=32```

### Testing CSNet
* Preparing the dataset for testing

* Editing the path of trained model in file `test.py` and `test_new.py`.

* For CSNet testing in terms of subrate=0.1:  (**ps: For this testing code, there is a big gap compared with the result in the publised paper. And I am confused about it. If you know the reason, please let me know. Thanks very much!**)

```python test.py --sub_rate=0.1 --block_size=32```

* For CSNet testing (new testing code) in terms of subrate=0.1:

```python test_new.py --cuda --sub_rate=0.1 --block_size=32```

## CSNet results
### Subjective results

![image](https://github.com/WenxueCui/CSNet-Pytorch/raw/master/images/results.jpg)

### Objective results
![image](https://github.com/WenxueCui/CSNet-Pytorch/raw/master/images/table.jpg)

## Additional instructions

* For training data, you can choose any natural image dataset.
* The training data is very important, if you can not achieve ideal result, maybe you can focus on the augmentation of training data or the structure of the network.
* If you like this repo, Star or Fork to support my work. Thank you.
* If you have any problem for this code, please email: wenxuecui@stu.hit.edu.cn

