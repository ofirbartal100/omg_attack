import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from dabs.src.datasets.specs import Input2dSpec


class MNIST(Dataset):
    # Dataset information.
    NUM_CLASSES = 10
    IN_CHANNELS = 1
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)
    MEAN = [0.1307]
    STD = [0.3081]

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'natural_images') 
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.dataset = datasets.mnist.MNIST(
            root=self.root,
            train=train,
            download=download,
        )

    def __getitem__(self, index):
        img, label = self.dataset.data[index], int(self.dataset.targets[index])
        # img = Image.fromarray(img)
        # img = self.transforms(img)
        return index, img.float().unsqueeze(0)/255.0, label

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def num_classes():
        return MNIST.NUM_CLASSES


    @staticmethod
    def normalize(imgs):
        mean = torch.tensor(MNIST.MEAN, device=imgs.device)
        std = torch.tensor(MNIST.STD, device=imgs.device)
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    @staticmethod
    def unnormalize(imgs):
        mean = torch.tensor(MNIST.MEAN, device=imgs.device)
        std = torch.tensor(MNIST.STD, device=imgs.device)
        imgs = (imgs * std[None, :, None, None]) + mean[None, :, None, None]
        return imgs

   


    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=MNIST.INPUT_SIZE,
                patch_size=MNIST.PATCH_SIZE,
                in_channels=MNIST.IN_CHANNELS,
            ),
        ]