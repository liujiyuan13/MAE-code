'''
This is written by Jiyuan Liu, Dec. 21, 2021.
Homepage: https://liujiyuan13.github.io.
Email: liujiyuan13@163.com.
All rights reserved.
'''

import time
import math
import argparse
import torch
import tensorboard_logger

from vit import ViT
from model import MAE
from util import *

# for re-produce
set_seed(0)


def build_model(args):
    '''
    Build MAE model.
    :param args: model args
    :return: model
    '''
    # build model
    v = ViT(image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.n_class,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim)

    mae = MAE(encoder=v,
              masking_ratio=args.masking_ratio,
              decoder_dim=args.decoder_dim,
              decoder_depth=args.decoder_depth,
              device=args.device).to(args.device)

    return mae


def train(args):
    # load data
    data_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=args.batch_size,
                                          n_worker=args.n_worker,
                                          is_train=True)

    # build mae model
    model = build_model(args)
    model.train()

    # build optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_lr,
                                  weight_decay=args.weight_decay,
                                  betas=args.momentum)

    # learning rate scheduler: warmup + consine
    def lr_lambda(epoch):
        if epoch < args.epochs_warmup:
            p = epoch / args.epochs_warmup
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        else:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # tensorboard
    tb_logger = tensorboard_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    for epoch in range(1, args.epochs + 1):
        # records
        ts = time.time()
        losses = AverageMeter()

        # train by epoch
        for idx, (images, targets) in enumerate(data_loader):
            # put images into device
            images = images.to(args.device)
            # forward
            loss = model(images)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # record
            losses.update(loss.item(), args.batch_size)

        # log
        tb_logger.log_value('loss', losses.avg, epoch)

        # print
        if epoch % args.print_freq == 0:
            print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch, time.time() - ts, losses.avg))

        # save checkpoint
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.ckpt_folder, 'epoch_{:d}.ckpt'.format(epoch))
            save_ckpt(model, optimizer, args, epoch, save_file=save_file)

    # save the last checkpoint
    save_file = os.path.join(args.ckpt_folder, 'last.ckpt')
    save_ckpt(model, optimizer, args, epoch, save_file=save_file)


def default_args(data_name, trail=0):
    # params
    args = argparse.ArgumentParser().parse_args()

    # device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_dir = 'data'
    args.data_name = data_name
    args.image_size = 256
    args.n_worker = 8

    # model
    # - use ViT-Base whose parameters are referred from "Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers
    # - for Image Recognition at Scale. ICLR 2021. https://openreview.net/forum?id=YicbFdNTTy".
    args.patch_size = 32
    args.vit_dim = 768
    args.vit_depth = 12
    args.vit_heads = 12
    args.vit_mlp_dim = 3072
    args.masking_ratio = 0.75  # the paper recommended 75% masked patches
    args.decoder_dim = 512  # paper showed good results with 512
    args.decoder_depth = 8  # paper showed good results with 8

    # train
    args.batch_size = 4096
    args.epochs = 800
    args.base_lr = 1.5e-4
    args.lr = args.base_lr * args.batch_size / 256
    args.weight_decay = 5e-2
    args.momentum = (0.9, 0.95)
    args.epochs_warmup = 40
    args.warmup_from = 1e-4
    args.lr_decay_rate = 1e-2
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2

    # print and save
    args.print_freq = 5
    args.save_freq = 100

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    return args


if __name__ == '__main__':
    data_name = 'cifar10'
    train(default_args(data_name))
