import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def set_seed(seed=0):
    """
    set seed for torch.
    @param seed: int, defualt 0
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def load_data(data_dir, data_name, is_train, img_size, batch_size, n_worker):
    """
    load data.
    @param data_dir: data dir, data folder
    @param data_name: data name
    @param is_train: train data or test data
    @param img_size: image size
    @param batch_size: batch size
    @param n_worker: number of workers to load data
    @return: data_loader: loader for train data;
             n_class: number of data classes
    """

    # load data
    if data_name is 'cifar10':
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        data = datasets.CIFAR10(data_dir, transform=transform, train=is_train, download=True)
    elif data_name is 'cifar100':
        MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        data = datasets.CIFAR100(data_dir, transform=transform, train=is_train, download=True)
    elif data_name is 'stl10':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        data = datasets.STL10(data_dir, transform=transform, split='train' if is_train else 'test', download=True)
    elif data_name is 'imagenet':
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # constants in timm.data.constants
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        data = datasets.ImageNet(data_dir, transform=transform, split='train' if is_train else 'val', download=True)
    else:
        raise Exception(data_name + ': not supported yet.')

    # obtain class number from test data
    n_class = len(set(data.targets))

    # create data loader
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
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
    :return:
    '''
    ckpt = torch.load(load_file)
    model.load_state_dict(ckpt['model'])
    del ckpt