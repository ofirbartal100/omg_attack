import os
from glob import glob
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from dabs.src.datasets.specs import Input2dSpec
import imgaug.augmenters as iaa

# https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
TRAFFICSIGN_RESOURCES = {
    'traffic_sign': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
}


class TrafficSign(Dataset):
    # Dataset information.
    NUM_CLASSES = 43
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    MEAN = [0.3337, 0.3064, 0.3171]
    STD = [0.2672, 0.2564, 0.2629]

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        self.root = os.path.join(base_root, 'natural_images', 'traffic_sign')
        self.transforms = transforms.Compose(
            [transforms.Resize(self.INPUT_SIZE),
             transforms.CenterCrop(self.INPUT_SIZE),
             transforms.ToTensor()]
        )

        # self.transforms = transforms.Compose(
        #     [iaa.Sequential([
        #         iaa.Resize(32),
        #         iaa.pillike.Equalize(),
        #         ]).augment_image,
        #      transforms.ToTensor()]
        # )


        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def _is_downloaded(self) -> bool:
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the meta dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        filename = TRAFFICSIGN_RESOURCES['traffic_sign'].rpartition('/')[2]
        download_and_extract_archive(TRAFFICSIGN_RESOURCES['traffic_sign'], download_root=self.root, filename=filename)
        print('Done!')

    def load_images(self):
        rs = np.random.RandomState(42)
        all_filepaths, all_labels = [], []
        for class_i in range(self.NUM_CLASSES):
            class_dir_i = join(self.root, 'GTSRB', 'Final_Training', 'Images', '{:05d}'.format(class_i))
            image_paths = glob(join(class_dir_i, '*.ppm'))
            # train test splitting
            image_paths = np.array(image_paths)
            num = len(image_paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            image_paths = image_paths[indexer].tolist()
            if self.train:
                image_paths = image_paths[:int(0.8 * num)]
            else:
                image_paths = image_paths[int(0.8 * num):]
            labels = [class_i] * len(image_paths)
            all_filepaths.extend(image_paths)
            all_labels.extend(labels)

        return all_filepaths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        image = self.transforms(image)
        # image = self.transforms(np.array(image))
        return index, image, label

    @staticmethod
    def num_classes():
        return TrafficSign.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=TrafficSign.INPUT_SIZE,
                patch_size=TrafficSign.PATCH_SIZE,
                in_channels=TrafficSign.IN_CHANNELS,
            ),
        ]

    @staticmethod
    def normalize(imgs):
        mean = torch.tensor(TrafficSign.MEAN, device=imgs.device)
        std = torch.tensor(TrafficSign.STD, device=imgs.device)
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    @staticmethod
    def unnormalize(imgs):
        mean = torch.tensor(TrafficSign.MEAN, device=imgs.device)
        std = torch.tensor(TrafficSign.STD, device=imgs.device)
        imgs = (imgs * std[None, :, None, None]) + mean[None, :, None, None]
        return imgs


class TrafficSignSmall(TrafficSign):
    # Dataset information.
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=TrafficSignSmall.INPUT_SIZE,
                patch_size=TrafficSignSmall.PATCH_SIZE,
                in_channels=TrafficSignSmall.IN_CHANNELS,
            ),
        ]


class TrafficSignSmall80Percent(TrafficSignSmall):
    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super(TrafficSignSmall,self).__init__(base_root, download, train)

        classes_to_mask = [23,36,24,17,4,31,42,10]
        if classes_to_mask is not None:
            # 80% classes
            TrafficSignSmall80Percent.NUM_CLASSES = TrafficSign.NUM_CLASSES - len(classes_to_mask)
            # train
            # maskout = [ torch.Tensor(self.labels) == ctmo for ctmo in classes_to_mask]
            # maskout = ~ (torch.stack(maskout,axis=0).sum(0).bool())
            # self.labels = self.labels[maskout]
            # self.paths = self.paths[maskout]
            # shift = torch.stack([ self.labels > ctmo for ctmo in classes_to_mask],axis=0).sum(0)
            # self.labels = self.labels - shift
            maskout = [ np.array(self.labels) == ctmo for ctmo in classes_to_mask]
            maskout = (~ (np.stack(maskout,axis=0).sum(0).astype(bool))).tolist()
            self.labels = [self.labels[i] for i in range(len(maskout)) if maskout[i]]
            self.paths = [self.paths[i] for i in range(len(maskout)) if maskout[i]]
            shift = np.stack([ np.array(self.labels) > ctmo for ctmo in classes_to_mask],axis=0).sum(0).tolist()
            self.labels = [self.labels[i] - shift[i] for i in range(len(shift))]

    @staticmethod
    def num_classes():
        return TrafficSignSmall80Percent.NUM_CLASSES