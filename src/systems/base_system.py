import os
from abc import abstractmethod

import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, SubsetRandomSampler,Sampler, Subset
from dabs.src.datasets.utils import fraction_db

#from dabs.src.datasets.catalog import DATASET_DICT
from dabs.src.datasets.natural_images import lfw,traffic_sign, cu_birds,mnist , cifar
from dabs.src.models import transformer, resnet , jit_model , traffic_model , birds_model , mnist_model , cifar_model
#from dabs.src.models import resnet, jit_model
from viewmaker.src.utils.utils import load_json, save_json
from dotmap import DotMap
import os

class Subset_Index(Subset):
    def __getitem__(self, idx):
        index, img, label = self.dataset[self.indices[idx]]
        return idx,  img, label


def get_model(config: DictConfig, **kwargs):
    '''Retrieves the specified model class, given the dataset class.'''
    if config.model.name == 'transformer':
        model_class = transformer.DomainAgnosticTransformer
        dataset_class = None
    elif "resnet" in config.model.name:
        model_class = resnet.ResNetDabs
        dataset_class = None
    elif "jit" in config.model.name:
        model_class = jit_model.JitModel
        dataset_class = lfw.LFW112 
    elif "traffic" == config.model.name:
        model_class = traffic_model.TrafficModel
        dataset_class = traffic_sign.TrafficSignSmall 
    elif "traffic_model_80" == config.model.name:
        model_class = traffic_model.TrafficModel80
        dataset_class = traffic_sign.TrafficSignSmall80Percent
    elif "cu_birds" == config.model.name:
        model_class = birds_model.BirdsModel
        dataset_class = cu_birds.CUBirds
    elif "cu_birds_80" == config.model.name:
        model_class = birds_model.BirdsModel80
        dataset_class = cu_birds.CUBirds80Percent
    elif "mnist_model" in config.model.name:
        model_class = mnist_model.MnistModel
        dataset_class = mnist.MNIST
    elif "cifar_model" in config.model.name:
        model_class = cifar_model.CifarModel
        dataset_class = cifar.CIFAR10Small
    else:
        raise ValueError(f'Encoder {config.model.name} doesn\'t exist.')
    # Retrieve the dataset-specific params.
    kwargs.update(config.model.kwargs)
    spec = dataset_class.spec()
    encoder_model = model_class(
        input_specs=spec,
        **kwargs, )
    return encoder_model , dataset_class


class BaseSystem(pl.LightningModule):

    def __init__(self, config: DictConfig):
        '''An abstract class that implements some shared functionality for training.

        Args:
            config: a hydra config
        '''
        super().__init__()
        self.config = config
        self.model, self.dataset = get_model(config)
        self.low_data = config.dataset.get("low_data", 1.0)

    @abstractmethod
    def objective(self, *args):
        '''Computes the loss and accuracy.'''
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def setup(self, stage):
        '''Called right after downloading data and before fitting model, initializes datasets with splits.'''
        self.train_dataset = self.dataset(base_root=self.config.data_root, download=True, train=True )
        self.val_dataset = self.dataset(base_root=self.config.data_root, download=True, train=False )

        self.train_loader_indices = fraction_db(self.train_dataset, self.low_data)[0]
        self.val_loader_indices = fraction_db(self.val_dataset, self.low_data)[0]


        self.train_dataset_ss = Subset_Index(self.train_dataset,self.train_loader_indices)
        self.val_dataset_ss = Subset_Index(self.val_dataset,self.val_loader_indices)

        try:
            print(
                '\033[94m' + f'{len(self.train_dataset)} train examples, {len(self.val_dataset)} val examples' + '\033[0m')
        except TypeError:
            print('Iterable/streaming dataset- undetermined length.')

        # save config to experiment dir
        vm_config = DotMap(self.config)
        vm_config.train_dataset.IN_CHANNELS = self.train_dataset.IN_CHANNELS
        config_out = os.path.join(vm_config.exp.base_dir, vm_config.exp.name, 'vm_config.yaml')
        os.makedirs(os.path.dirname(config_out), exist_ok=True)
        conf = OmegaConf.create(vm_config.toDict())
        with open(config_out, 'w') as fp:
            OmegaConf.save(config=conf, f=fp.name)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_ss,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            # sampler=SubsetRandomSampler(self.train_loader_indices),
            # shuffle=not isinstance(self.train_dataset, IterableDataset),
        )

    def val_dataloader(self):
        if not self.val_dataset:
            raise ValueError('Cannot get validation data for this dataset')

        return DataLoader(
            self.val_dataset_ss,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            # sampler=SubsetRandomSampler(self.val_loader_indices),
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.config.optim.name == 'adam':
            optim = torch.optim.AdamW(params, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.name == 'sgd':
            optim = torch.optim.SGD(
                params,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
                nesterov=self.config.optim.get("nesterov", False),
            )
        else:
            raise ValueError(f'{self.config.optim.name} optimizer unrecognized.')
        return optim

    def on_train_end(self):
        model_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'model.ckpt')
        torch.save(self, model_path)
        print(f'Pretrained model saved to {model_path}')
