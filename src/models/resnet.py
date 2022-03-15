from typing import List, Dict, Any, Tuple, Sequence

import torch
import torchvision
from torch import nn

from dabs.src.models.base_model import BaseModel
from viewmaker.src.models import resnet_small
# from flowlp.flpert.models.resnet import resnet18


class ResNetWrapper(BaseModel):

    def __init__(self, resnet):
        self.resnet = resnet

    def embed(self, inputs: Sequence[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        return self.resnet(x)


class ResNetDabs(BaseModel):

    def __init__(self,
                 input_specs: List[Dict[str, Any]],
                 resnet_type="resnet18",
                 dim=128,
                 out_dim=128,
                 projection_head=False):
        # super(ResNetDabs, self).__init__(input_specs)
        super(ResNetDabs, self).__init__(None)
        self.emb_dim = out_dim
        if resnet_type == 'resnet_small':
            # ResNet variant for smaller inputs (e.g. CIFAR-10).
            model_class = resnet_small.ResNet18
            encoder_model = model_class(out_dim,
                                        num_channels=input_specs[0].in_channels,
                                        input_size=input_specs[0].input_size[0])
        # elif resnet_type == "resnet18_flowlp":
        #     model_class = resnet18
        #     encoder_model = model_class(num_classes=out_dim,
        #                                 in_channels=input_specs[0].in_channels)
        else:
            model_class = getattr(torchvision.models, resnet_type)
            encoder_model = model_class(
                pretrained=False,
                num_classes=out_dim,
            )
            encoder_model.conv1 = nn.Conv2d(in_channels=input_specs[0].in_channels,
                                            out_channels=encoder_model.conv1.out_channels,
                                            kernel_size=encoder_model.conv1.kernel_size,
                                            stride=encoder_model.conv1.stride,
                                            padding=encoder_model.conv1.padding,
                                            bias=False)
            if projection_head:
                mlp_dim = encoder_model.fc.weight.size(1)
                encoder_model.fc = nn.Sequential(
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.ReLU(),
                    encoder_model.fc,
                )
        self.resnet = encoder_model

    def embed(self, inputs: Tuple[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        return self.resnet(x)
