import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms
import os
from dabs.src.datasets.specs import Input2dSpec
from viewmaker.src.datasets.root_paths import DATA_ROOTS


class LFW(data.Dataset):
    NUM_CLASSES = 1
    INPUT_SIZE = (128, 128)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    MEAN = [0.4321, 0.3748, 0.3333]
    STD = [0.2819, 0.2563, 0.2500]

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        base_root = DATA_ROOTS['lfw']
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )

        # seperate test images from training data
        # self.test_subset = [13,14,15,16,70,78,102,126,167,283,308,334,359,377,386,381,388,397,424,411,448,486,502,569,550,598,590,622,626,646,667,670,690,651,655,792,796,788,763,768,822,823,907,956,998,999,1052,1016,1028,1146,1553,1552,1561,1521,1522,1603,1678,1752,1753,1756,1872,1830,1916,1919,1939,2483,2484,2653,2696,2603,2609,2815]
        # def set_test_as_invalid(x):
        #     head, tail = os.path.split(x)
        #     name = tail.split('.')[0]
        #     if int(name) in self.test_subset:
        #         return False
        #     else:
        #         return True


        self.dataset = datasets.ImageFolder(base_root, transform=self.transforms) #, is_valid_file=set_test_as_invalid)
        

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
        # neg_index = np.random.choice(np.arange(self.__len__()))
        # neg_index += offset
        # index += offset
        img_data, label = self.dataset.__getitem__(index+offset)
        return index , img_data, label
        # img2_data, _ = self.dataset.__getitem__(index)
        # neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        # data = [index, img_data.float(), img2_data.float(),
                # neg_data.float(), label]
        # return tuple(data)

    def __len__(self):
        if self.train:
            return len(self.dataset) // 10 * 9
        else:
            return len(self.dataset) // 10

    @staticmethod
    def num_classes():
        return LFW.NUM_CLASSES

    @staticmethod
    def normalize(imgs):
        mean = torch.tensor(LFW.MEAN, device=imgs.device)
        std = torch.tensor(LFW.STD, device=imgs.device)
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    @staticmethod
    def unnormalize(imgs):
        mean = torch.tensor(LFW.MEAN, device=imgs.device)
        std = torch.tensor(LFW.STD, device=imgs.device)
        imgs = (imgs * std[None, :, None, None]) + mean[None, :, None, None]
        return imgs


class LFW32(LFW):

    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)


    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=LFW32.INPUT_SIZE, patch_size=LFW32.PATCH_SIZE, in_channels=LFW32.IN_CHANNELS),
        ]

class LFW64(LFW):

    INPUT_SIZE = (64, 64)
    PATCH_SIZE = (4, 4)


    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=LFW64.INPUT_SIZE, patch_size=LFW64.PATCH_SIZE, in_channels=LFW64.IN_CHANNELS),
        ]

class LFW112(LFW):

    INPUT_SIZE = (112, 112)
    PATCH_SIZE = (8, 8)


    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=LFW112.INPUT_SIZE, patch_size=LFW112.PATCH_SIZE, in_channels=LFW112.IN_CHANNELS),
        ]