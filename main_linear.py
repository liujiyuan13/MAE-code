import time
import math
import argparse
import torch
import tensorboard_logger as tb_logger

from vit import ViT
from model import LinearProb
from util import *

# for re-produce
set_seed(0)


def build_model(args):
    '''
    Build MAE model.
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
    lbp = LinearProb(encoder=v,
                     n_class=args.n_class,
                     masking_ratio=0).to(args.device)

    return lbp


def train(args):
    # load data
    data_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=args.batch_size,
                                          n_worker=args.n_worker,
                                          is_train=True)

    # build linear probing model and restore encoder weights
    model = build_model(args)
    state_dict_encoder = model.encoder.state_dict()
    state_dict_loaded = torch.load(args.ckpt)['model']
    for k in state_dict_encoder.keys():
        state_dict_encoder[k] = state_dict_loaded[k]
    model.encoder.load_state_dict(state_dict_encoder)

    # build optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)

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

    for epoch in range(1, args.epochs + 1):
        # set training mode
        model.encoder.eval()
        model.fc.train()

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
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, targets)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # record
            losses.update(loss.item(), args.batch_size)

        # log
        args.tb_logger.log_value('loss_linear_probing', losses.avg, epoch)

        # eval
        if epoch % args.eval_freq == 0:
            acc = test(args, model)
            args.tb_logger.log_value('acc_linear_probing', acc, epoch)

        # print
        if epoch % args.print_freq == 0:
            print('- epoch {:4d}, time, {:2f}s, loss {:4f}'.format(epoch, time.time() - ts, losses.avg))

    # save the last checkpoint
    save_file = os.path.join(args.ckpt_dir, 'linear_probing.ckpt')
    save_ckpt(model, optimizer, args, epoch, save_file=save_file)


def test(args, model=None, ckpt_path=None):
    '''
    train the model
    :param args: args
    :param model: the test model
    :param ckpt_path: checkpoint path, if model is given, this is deactivated
    :return: accuracy
    '''

    # load data
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
            _, y_pred = torch.max(output, dim=1).detach().cpu().numpy()
            acc = accuracy(targets, y_pred)
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
    args.batch_size = 256  # 16384
    args.n_worker = 8

    # model
    args.patch_size = 32
    args.vit_dim = 1024
    args.vit_depth = 6
    args.vit_heads = 8
    args.vit_mlp_dim = 2048
    args.masking_ratio = 0.75  # the paper recommended 75% masked patches
    args.decoder_dim = 512  # paper showed good results with just 512
    args.decoder_depth = 6  # anywhere from 1 to 8

    # train
    args.epochs = 90
    args.base_lr = 1e-1
    args.lr = args.base_lr * args.batch_size / 256
    args.weight_decay = 0
    args.momentum = 0.9
    args.epochs_warmup = 10
    args.warmup_from = 1e-4
    args.lr_decay_rate = 1e-2
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2

    # print and save
    args.print_freq = 1
    args.eval_freq = 1
    args.save_freq = args.epochs + 1

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    args.tb_logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    args.ckpt = os.path.join(args.ckpt_folder, ckpt_file)

    return args


if __name__ == '__main__':
    data_name = 'cifar10'
    train(default_args(data_name))
