import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from dabs.src.datasets.specs import Input2dSpec


class CIFAR10(Dataset):
    # Dataset information.
    NUM_CLASSES = 10
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    MEAN = [0.49139968, 0.48215827 ,0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'natural_images', 'CIFAR10')
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )
        self.dataset = datasets.cifar.CIFAR10(
            root=self.root,
            train=train,
            download=download,
        )

    def __getitem__(self, index):
        img, label = self.dataset.data[index], int(self.dataset.targets[index])
        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)
        return index, img, label

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def num_classes():
        return CIFAR10.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=CIFAR10.INPUT_SIZE, patch_size=CIFAR10.PATCH_SIZE, in_channels=CIFAR10.IN_CHANNELS),
        ]

    @staticmethod
    def normalize(imgs):
        mean = torch.tensor(CIFAR10.MEAN, device=imgs.device)
        std = torch.tensor(CIFAR10.STD, device=imgs.device)
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    @staticmethod
    def unnormalize(imgs):
        mean = torch.tensor(CIFAR10.MEAN, device=imgs.device)
        std = torch.tensor(CIFAR10.STD, device=imgs.device)
        imgs = (imgs * std[None, :, None, None]) + mean[None, :, None, None]
        return imgs

class CIFAR10Small(CIFAR10):
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    MEAN =[0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=CIFAR10Small.INPUT_SIZE,
                patch_size=CIFAR10Small.PATCH_SIZE,
                in_channels=CIFAR10Small.IN_CHANNELS,
            ),
        ]


class CIFAR10Small80(CIFAR10Small):
    classes_to_mask = [1,6]
    NUM_CLASSES = 10 - len(classes_to_mask)
    
    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__( base_root, download, train)

        self.root = os.path.join(base_root, 'natural_images', 'CIFAR10')
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )
        self.dataset = datasets.cifar.CIFAR10(
            root=self.root,
            train=train,
            download=download,
        )

        if CIFAR10Small80.classes_to_mask is not None:
            # 80% classes
            targets = np.array(self.dataset.targets)
            maskout = [ targets == ctmo for ctmo in CIFAR10Small80.classes_to_mask]
            maskout = ~ (np.stack(maskout,axis=0).sum(0).astype(bool))
            targets = targets[maskout]
            self.dataset.data = self.dataset.data[maskout]
            shift = np.stack([ targets > ctmo for ctmo in CIFAR10Small80.classes_to_mask],axis=0).sum(0)
            self.dataset.targets = (targets - shift).tolist()

    @staticmethod
    def num_classes():
        return CIFAR10Small80.NUM_CLASSES