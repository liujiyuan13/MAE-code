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
from lars import LARS
from model import EvalNet, LabelSmoothing
from util import *

# for re-produce
set_seed(0)


def build_model(args):
    '''
    build EvalNet model and restore weights
    :param args: model args
    :return: model
    '''
    # build encoder
    v = ViT(image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.n_class,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim).to(args.device)

    # build linear probing
    enet = EvalNet(encoder=v,
                   n_class=args.n_class,
                   masking_ratio=0,
                   device=args.device).to(args.device)

    # restore weights
    state_dict_encoder = enet.encoder.state_dict()
    state_dict_loaded = torch.load(args.ckpt)['model']
    for k in state_dict_encoder.keys():
        state_dict_encoder[k] = state_dict_loaded['encoder.' + k]
    enet.encoder.load_state_dict(state_dict_encoder)

    return enet


def train(args):
    # load data
    data_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=args.batch_size,
                                          n_worker=args.n_worker,
                                          is_train=True)
    test_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=args.batch_size,
                                          n_worker=args.n_worker,
                                          is_train=False)

    # build model
    model = build_model(args)

    # build optimizer
    if args.n_partial == 0:
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=args.base_lr,
        #                             weight_decay=args.weight_decay,
        #                             momentum=args.momentum)
        optimizer = LARS(model.parameters(),
                         lr=args.base_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)
    else:
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
        # set training mode
        model.encoder.eval()
        model.fc.train()
        if args.n_partial == 0.5 or (type(args.n_partial) is int and 1 <= args.n_partial <= args.vit_depth):
            model.encoder.mlp_head.train()
            for i in range(1, int(args.n_partial)+1):
                model.encoder.transformer.layers[args.vit_depth-i].train()
        elif args.n_partial == 0:
            pass
        else:
            raise ValueError('please check requirements of \'args.n_partial\'.')

        # records
        ts = time.time()
        losses = AverageMeter()

        # train by epoch
        for idx, (images, targets) in enumerate(data_loader):
            # put images into device
            images, targets = images.to(args.device), targets.to(args.device)
            # forward
            output = model(images)
            # compute loss
            if args.label_smoothing:
                criterion = LabelSmoothing(smoothing=args.smoothing)   # use label smoothing technique
            else:
                criterion = torch.nn.CrossEntropyLoss()   # common and simplest one
            loss = criterion(output, targets)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # record
            losses.update(loss.item(), args.batch_size)

        # log
        tb_logger.log_value('loss_eval_partial_{}'.format(args.n_partial), losses.avg, epoch)

        # eval
        if epoch % args.eval_freq == 0:
            acc = test(args, model=model, data_loader=test_loader)
            tb_logger.log_value('acc_eval_partial_{}'.format(args.n_partial), acc, epoch)

        # print
        if epoch % args.print_freq == 0:
            print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch, time.time() - ts, losses.avg))

    # save the last checkpoint
    save_file = os.path.join(args.ckpt_folder, 'enet_partial_{}.ckpt'.format(args.n_partial))
    save_ckpt(model, optimizer, args, epoch, save_file=save_file)


def test(args, model=None, ckpt_path=None, data_loader=None):
    '''
    train the model
    :param args: args
    :param model: the test model
    :param ckpt_path: checkpoint path, if model is given, this is deactivated
    :param data_loader: data loader
    :return: accuracy
    '''

    # load data
    if data_loader is None:
        data_loader, args.n_class = load_data(args.data_dir,
                                              args.data_name,
                                              image_size=args.image_size,
                                              batch_size=args.batch_size,
                                              n_worker=args.n_worker,
                                              is_train=False)

    # restore mae model
    assert model is not None or ckpt_path is not None
    if model is None:
        model = build_model(args)
        model = load_ckpt(model, ckpt_path)
    model.eval()

    # test
    accs = AverageMeter()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            # put images into device
            images = images.to(args.device)
            # forward
            output = model(images)
            # eval
            _, y_pred = torch.max(output, dim=1)
            acc = accuracy(targets.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            # record
            accs.update(acc, args.batch_size)

    return accs.avg


def default_args(data_name, trail=0, ckpt_file='last.ckpt'):
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
    args.patch_size = 32
    args.vit_dim = 768
    args.vit_depth = 12
    args.vit_heads = 12
    args.vit_mlp_dim = 3072
    args.masking_ratio = 0  # the paper recommended to use uncorrupted images

    # linear probing or partial fine-tuning or fine-tuning
    # - 0: linear probing, the encoder is fixed
    # - 0.5: fine-tuning MLP sub-block with the transformer fixed
    # - 1~(args.vit_depth-1): partial fine-tuning, including MLP sub-block and last layers of transformer
    # - args.vit_depth: fine-tuning, including MLP sub-block and all layers of transformer
    args.n_partial = 0

    # train
    if args.n_partial == 0:
        args.batch_size = 16384
        args.epochs = 90
        args.base_lr = 1e-1
        args.lr = args.base_lr * args.batch_size / 256
        args.weight_decay = 0
        args.momentum = 0.9
        args.epochs_warmup = 10
    else:
        args.batch_size = 1024
        args.epochs = 100
        args.base_lr = 1e-3
        args.lr = args.base_lr * args.batch_size / 256
        args.weight_decay = 5e-2
        args.momentum = (0.9, 0.999)
        args.epochs_warmup = 5
    args.warmup_from = 1e-4
    args.lr_decay_rate = 1e-2
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2

    # extra
    args.label_smoothing = True
    args.smoothing = 0.1

    # print and save
    args.print_freq = 5
    args.eval_freq = 5

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    args.ckpt = os.path.join(args.ckpt_folder, ckpt_file)

    return args


if __name__ == '__main__':
    data_name = 'imagenet'
    train(default_args(data_name))
