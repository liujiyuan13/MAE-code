# Masked Auto-Encoder (MAE)
Pytorch implementation of Masked Auto-Encoder:

* Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377v1). arXiv 2021.

![](mae.png)

## Usage
1. Clone to the local.
```
> git clone https://github.com/liujiyuan13/MAE-code.git MAE-code
```
2. Install required packages.
```
> cd MAE-code
> pip install requirements.txt
```
3. Prepare datasets. 
- For *Cifar10*, *Cifar100* and *STL*, skip this step for it will be done automatically;
- For *ImageNet1K*, download([option](https://www.image-net.org/download)) and unzip the train(val) set into *./data/ImageNet1K/train(val)*.
4. Set parameters.
- All parameters are kept in *defualt_args(\*)* function of * main_\*.py* file.
5. Run the code.
```
> python main_mae.py	# train MAE encoder
> python main_eval.py	# evaluate MAE encoder
```
6. Visualize the ouput.
```
> tensorboard --logdir=./log --port 8888
```

## Detail

## Result

## About


