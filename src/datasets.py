import torchvision.transforms as transforms
import torch
import numpy as np

from third_party.mean_teacher import data
from third_party.mean_teacher import datasets as mt_dataset
from third_party.mean_teacher.utils import export
from third_party.fastswa import datasets as fswa_dataset

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

@export
def cifar10(tnum=2):
    dataset = mt_dataset.cifar10()

    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    dataset['train_transformation'] = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)
    
    dataset['datadir'] = 'third_party/' + dataset['datadir']
    return dataset

@export
def cifar10_cutout(tnum=2):
    dataset = mt_dataset.cifar10()

    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    dataset['train_transformation'] = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats),
            Cutout(n_holes=1, length=16)
        ]), n=tnum)
    
    dataset['datadir'] = 'third_party/' + dataset['datadir']
    return dataset

@export
def cifar100(tnum=2):
    dataset = fswa_dataset.cifar100()

    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    dataset['train_transformation'] = data.TransformNTimes(
        transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]), n=tnum)

    dataset['datadir'] = 'third_party/' + dataset['datadir']
    return dataset


@export
def usps():
    channel_stats = dict(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'third_party/data-local/images/usps',
        'num_classes': 10,
    }


@export
def mnist():
    channel_stats = dict(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'third_party/data-local/images/mnist',
        'num_classes': 10,
    }
