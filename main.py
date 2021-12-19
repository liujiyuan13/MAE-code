import torch

from vit import ViT
from mae import MAE
from util import *
from conf import ARGS

# for re-produce
set_seed(0)


def build_mae(args):
    '''
    Build MAE model.
    :param args: model args
    :return: model
    '''
    # build model
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )

    mae = MAE(
        encoder=v,
        masking_ratio=0.75,  # the paper recommended 75% masked patches
        decoder_dim=512,  # paper showed good results with just 512
        decoder_depth=6  # anywhere from 1 to 8
    ).to(args.device)

    return mae


def train(args):

    # load data
    data_loader, _ = load_data()

    # build mae model
    mae = build_mae(args)
    mae.train()

    # build optimizer
    optim =  torch.optim.Adam(mae.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs+1):

        # train by epoch
        for idx, (images, targets) in enumerate(data_loader):
            # put images into device
            images = images.to(args.device)
            # forward
            loss = mae(images)
            # back propagation
            optim.zero_grad()
            loss.backward()
            optim.step()

        # evaluate representation quality
        if epoch % args.eval_freq == 0:
            test(mae, args)

        # save checkpoint
        if epoch % args.save_freq == 0:
            save_ckpt(mae, optim, args, epoch, save_file=)

    # save the last checkpoint
    save_ckpt(mae, optim, args, epoch, save_file=)

def test(mae, args):

    # load data
    data_loader, n_class = load_data()

    # restore mae model
    mae = build_mae(args)



if __name__ == '__main__':
    args = ARGS()
    train(args)

