import time
import argparse
import torch
import tensorboard_logger as tb_logger

from vit import ViT
from mae import MAE
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
    v = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.n_class,
        dim=args.vit_dim,
        depth=args.vit_depth,
        heads=args.vit_heads,
        mlp_dim=args.vit_mlp_dim
    )

    mae = MAE(
        encoder=v,
        masking_ratio=args.masking_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth
    ).to(args.device)

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
    optim = torch.optim.Adam(model.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay)

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
            optim.zero_grad()
            loss.backward()
            optim.step()
            # record
            losses.update(loss.item(), args.batch_size)

        # log
        args.tb_logger.log_value('loss_train', losses.avg, epoch)

        # print
        if epoch % args.print_freq == 0:
            print('- epoch {:4d}, time, {:2f}s, loss {:4f}'.format(epoch, time.time() - ts, losses.avg))

        # save checkpoint
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.ckpt_dir, 'epoch_{:4d}.ckpt'.format(epoch))
            save_ckpt(model, optim, args, epoch, save_file=save_file)

    # save the last checkpoint
    save_file = os.path.join(args.ckpt_dir, 'last.ckpt')
    save_ckpt(model, optim, args, epoch, save_file=save_file)


def test(args, model=None, ckpt_path=None):
    '''
    test the model
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
            output = model.encoder(images)
            # eval

            # accs.update(acc, args.batch_size)

    return accs.avg


def default_args(data_name, trail=0):
    # params
    args = argparse.ArgumentParser().parse_args()

    # device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_dir = 'data'
    args.data_name = data_name
    args.image_size = 256
    args.batch_size = 128
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
    args.learning_rate = 1e-3
    args.weight_decay = 1e-3
    args.epochs = 1000

    # print and save
    args.print_freq = 1
    args.save_freq = args.epochs + 1

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    args.tb_logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    return args


if __name__ == '__main__':
    data_name = 'cifar10'
    train(default_args(data_name))
