import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets

from dabs.src.datasets.specs import Input2dSpec
from viewmaker.src.datasets.root_paths import DATA_ROOTS


class FFHQ(data.Dataset):

    NUM_CLASSES = 1
    IN_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    MEAN = [0.5202, 0.4252, 0.3803]
    STD = [0.2496, 0.2238, 0.2210]

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.size = size
        self.train = train
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )
        self.dataset = datasets.ImageFolder(base_root, transform=self.transform)
        self.train_len = len(self.dataset) // 10 * 9
        if self.train:
            self.dataset.targets = self.dataset.targets[:self.train_len]
        else:
            self.dataset.targets = self.dataset.targets[self.train_len:]

    def __getitem__(self, index):
        # pick random number
        if self.train:
            offset = 0
        else:
            offset = self.train_len
        neg_index = np.random.choice(np.arange(self.__len__()))
        neg_index += offset
        index += offset
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(),
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        if self.train:
            return len(self.dataset) // 10 * 9
        else:
            return len(self.dataset) // 10

    @staticmethod
    def num_classes():
        return FFHQ.NUM_CLASSES

    @staticmethod
    def normalize(imgs):
        mean = torch.tensor(FFHQ.MEAN, device=imgs.device)
        std = torch.tensor(FFHQ.STD, device=imgs.device)
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    @staticmethod
    def unnormalize(imgs):
        mean = torch.tensor(FFHQ.MEAN, device=imgs.device)
        std = torch.tensor(FFHQ.STD, device=imgs.device)
        imgs = (imgs * std[None, :, None, None]) + mean[None, :, None, None]
        return imgs


class FFHQ32(FFHQ):

    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__(root=root, size=32, train=train, image_transforms=image_transforms)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=FFHQ32.INPUT_SIZE, patch_size=FFHQ32.PATCH_SIZE, in_channels=FFHQ32.IN_CHANNELS),
        ]

class FFHQ64(FFHQ):

    INPUT_SIZE = (64, 64)
    PATCH_SIZE = (4, 4)

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__(root=root, size=64, train=train, image_transforms=image_transforms)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=FFHQ64.INPUT_SIZE, patch_size=FFHQ64.PATCH_SIZE, in_channels=FFHQ64.IN_CHANNELS),
        ]

class FFHQ112(FFHQ):

    INPUT_SIZE = (112, 112)
    PATCH_SIZE = (8, 8)

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__(root=root, size=112, train=train, image_transforms=image_transforms)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=FFHQ112.INPUT_SIZE, patch_size=FFHQ112.PATCH_SIZE, in_channels=FFHQ112.IN_CHANNELS),
        ]