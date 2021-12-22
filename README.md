# Masked Auto-Encoder (MAE)

Pytorch implementation of Masked Auto-Encoder:

* Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377v1). arXiv 2021.

<div align="center">
<img src=https://github.com/liujiyuan13/MAE-code/blob/main/img/mae.png width=80% />
</div>


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
- For *ImageNet1K*, [download](https://www.image-net.org/download) and unzip the train(val) set into `./data/ImageNet1K/train(val)`.
4. Set parameters.
- All parameters are kept in `default_args()` function of `main_mae(eval).py` file.
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

### Project structure

```
...
+ ckpt				# checkpoint
+ data 				# data folder
+ img 				# store images for README.md
+ log 				# log files
.gitignore 			
lars.py 			# LARS optimizer
main_eval.py 			# main file for evaluation
main_mae.py  			# main file for MAE training
model.py 			# model definitions of MAE and EvalNet
README.md 
util.py 			# helper functions
vit.py 				# definition of vision transformer
```

### Encoder setting

In the paper, *ViT-Base*, *ViT-Large* and *ViT-Huge* are used. 
You can switch between them by simply changing the parameters in `default_args()`.
Details can be found [here](https://openreview.net/forum?id=YicbFdNTTy) and are listed in following table.

|  Name | Layer Num. | Hidden Size |   MLP Size  | Head Num. |
|:-----:|:----------:|:-----------:|:-----------:|:---------:|
|  Arg  |  vit_depth |   vit_dim   | vit_mlp_dim | vit_heads |
| ViT-B |     12     |     768     |     3072    |     12    |
| ViT-L |     24     |     1024    |     4096    |     16    |
| ViT-H |     32     |     1280    |     5120    |     16    |

### Evaluation setting

I implement four network training strategies concerned in the paper, including 
- **pre-training** is used to train MAE encoder and done in `main_mae.py`.
- **linear probing** is used to evaluate MAE encoder. During training, MAE encoder is fixed.
	+ `args.n_partial = 0`
- **partial fine-tuning** is used to evaluate MAE encoder. During training, MAE encoder is partially fixed.
	+ `args.n_partial = 0.5` --> fine-tuning MLP sub-block with the transformer fixed
	+ `1<=args.n_partial<=args.vit_depth-1` --> fine-tuning MLP sub-block and last layers of transformer
- **end-to-end fine-tuning** is used to evaluate MAE encoder. During training, MAE encoder is fully trainable.
	+ `args.n_partial = args.vit_depth`

Note that the last three strategies are done in `main_eval.py` where parameter `args.n_partial` is located.

At the same time, I follow the parameter settings in the paper appendix.
Note that **partial fine-tuning** and **end-to-end fine-tuning** use the same setting.
Nevertheless, I replace `RandAug(9, 0.5)` with `RandomResizedCrop` and leave `mixup`, `cutmix` and `drop path` techniques in further implementation. 


## Result

The experiment reproduce will takes a long time and I am unfortunately busy these days.
If you get some results and are willing to contribute, please reach me via email. Thanks!

By the way, **I have run the code from start to end.** 
**It works!**
So don't worry about the implementation errors. 
If you find any, please raise issues or email me.


## Licence

This repository is under [GPL V3](https://github.com/liujiyuan13/MAE-code/blob/main/LICENSE).

## About

Thanks project [*vit-pytorch*](https://github.com/lucidrains/vit-pytorch), [*pytorch-lars*](https://github.com/JosephChenHub/pytorch-lars) and [*DeepLearningExamples*](https://github.com/NVIDIA/DeepLearningExamples) for their codes contribute to this repository a lot!

Homepage: <https://liujiyuan13.github.io>

Email: <liujiyuan13@163.com>
