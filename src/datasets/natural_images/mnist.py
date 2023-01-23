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
    # MEAN = [0.1307]
    # STD = [0.3081]
    MEAN = [0]
    STD = [1]

    def __init__(self, base_root: str, download: bool = False, train: bool = True, classes_to_mask=None) -> None:
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


        if classes_to_mask is not None:
            # 80% classes
            MNIST.NUM_CLASSES = 10 - len(classes_to_mask)
            # train
            maskout = [ self.dataset.targets == ctmo for ctmo in classes_to_mask]
            maskout = ~ (torch.stack(maskout,axis=0).sum(0).bool())
            self.dataset.targets = self.dataset.targets[maskout]
            self.dataset.data = self.dataset.data[maskout]
            shift = torch.stack([ self.dataset.targets > ctmo for ctmo in classes_to_mask],axis=0).sum(0)
            self.dataset.targets = self.dataset.targets - shift


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