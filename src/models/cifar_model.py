from types import MethodType
from typing import List, Dict, Any, Tuple, Sequence

import torch
import torch.nn as nn
from dabs.src.models.base_model import BaseModel
import torch.nn.functional as F
from omegaconf import OmegaConf
# from pytorch_image_classification.pytorch_image_classification.models.cifar.vgg import Network
import sys
sys.path.insert(0,'/workspace/pytorch_image_classification_repo')
from fvcore.common.checkpoint import Checkpointer

class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.vgg
        self.use_bn = model_config.use_bn
        n_channels = model_config.n_channels
        n_layers = model_config.n_layers

        self.stage1 = self._make_stage(config.dataset.n_channels,
                                       n_channels[0], n_layers[0])
        self.stage2 = self._make_stage(n_channels[0], n_channels[1],
                                       n_layers[1])
        self.stage3 = self._make_stage(n_channels[1], n_channels[2],
                                       n_layers[2])
        self.stage4 = self._make_stage(n_channels[2], n_channels[3],
                                       n_layers[3])
        self.stage5 = self._make_stage(n_channels[3], n_channels[4],
                                       n_layers[4])

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)

        # initialize weights
        # initializer = create_initializer(config.model.init_mode)
        # self.apply(initializer)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            if index == 0:
                conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                conv = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            stage.add_module(f'conv{index}', conv)
            if self.use_bn:
                stage.add_module(f'bn{index}', nn.BatchNorm2d(out_channels))
            stage.add_module(f'relu{index}', nn.ReLU(inplace=True))
        stage.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        return stage

    def _forward_conv(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CifarModel(BaseModel):

    def __init__(self, cifar_model_path, vgg_config_path ,input_specs: List[Dict[str, Any]]):
        super(CifarModel, self).__init__(None)
        self.cifar_model_path = cifar_model_path
        config = OmegaConf.load(vgg_config_path)
        self.cifar_model = Network(config)
        # scope(self.jit_model,input_size=(3,112,112))
        checkpointer = Checkpointer(self.cifar_model)
        ckpt = checkpointer.load(cifar_model_path)
        # self.cifar_model.load_state_dict(torch.load(cifar_model_path))
        self.cifar_model.eval()
        for p in self.cifar_model.parameters():
            p.requires_grad = False

        def forward(self, x):
            x = self._forward_conv(x)
            y = x.view(x.size(0), -1)
            x = self.fc(y)
            return x , y # dim 128

        self.cifar_model.forward = MethodType(forward, self.cifar_model)
        


    def embed(self, inputs: Sequence[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        if self.cifar_model.training:
            self.cifar_model.eval()
            
        return self.cifar_model(x)[1]

