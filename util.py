import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def set_seed(seed=0):
    """
    set seed for torch.
    @param seed: int, defualt 0
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_data(data_dir, data_name, is_train, image_size, batch_size, n_worker):
    """
    load data.
    @param data_dir: data dir, data folder
    @param data_name: data name
    @param is_train: train data or test data
    @param image_size: image size
    @param batch_size: batch size
    @param n_worker: number of workers to load data
    @return: data_loader: loader for train data;
             n_class: number of data classes
    """

    # load data
    if data_name is 'cifar10':
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        data = datasets.CIFAR10(data_dir, transform=transform, train=is_train, download=True)
    elif data_name is 'cifar100':
        MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        data = datasets.CIFAR100(data_dir, transform=transform, train=is_train, download=True)
    elif data_name is 'stl10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor()
        ])
        data = datasets.STL10(data_dir, transform=transform, split='train' if is_train else 'test', download=True)
    elif data_name is 'imagenet':
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # constants in timm.data.constants
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        data = datasets.ImageFolder(os.path.join(data_dir, 'ImageNet1K', 'train' if is_train else 'val'), transform=transform)
    else:
        raise Exception(data_name + ': not supported yet.')

    # obtain class number from test data
    n_class = len(set(data.targets))

    # create data loader
    data_loader = DataLoader(data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=n_worker,
                             pin_memory=True,
                             drop_last=True)

    return data_loader, n_class


def save_ckpt(model, optimizer, args, epoch, save_file):
    '''
    save checkpoint
    :param model: target model
    :param optimizer: used optimizer
    :param args: training parameters
    :param epoch: save at which epoch
    :param save_file: file path
    :return:
    '''
    ckpt = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(ckpt, save_file)
    del ckpt


def load_ckpt(model, load_file):
    '''
    load ckpt to model
    :param model: target model
    :param load_file: file path
    :return: the loaded model
    '''
    ckpt = torch.load(load_file)
    model.load_state_dict(ckpt['model'])
    del ckpt
    return model


def accuracy(y_true, y_pred):
    """
    compute classification accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    assert y_pred.size == y_true.size
    y_true, y_pred = y_true.astype(np.int64), y_pred.astype(np.int64)
    return sum(np.equal(y_true, y_pred)) / y_true.size


class AverageMeter(object):
    '''
    compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count