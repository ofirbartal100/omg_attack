from typing import List, Dict, Any, Tuple, Sequence

import torch
import torch.nn as nn
from dabs.src.models.base_model import BaseModel
import torch.nn.functional as F
# from MNIST.example import MnistNet
from MNIST.example import Model_C, MnistNet
from types import MethodType

class MnistModel(BaseModel):

    def __init__(self, mnist_model_path,input_specs: List[Dict[str, Any]]):
        super(MnistModel, self).__init__(None)
        self.mnist_model_path = mnist_model_path

        type_net = 'c'
        type_net = 'm'
        
        if type_net == 'c':
            # self.mnist_model = Model_C(1,8)
            self.mnist_model = Model_C(1,10)
            def forward(self, x):
                x = F.relu(self.conv1_1(x))
                x = F.relu(self.conv1_2(x))
                x = self.maxpool1(x)
                x = F.relu(self.conv2_1(x))
                x = F.relu(self.conv2_2(x))
                x = self.maxpool2(x)
                x = x.view(x.size(0), -1)
                y = F.relu(self.fc1(x))
                x = self.fc2(y)
                output = F.log_softmax(x, dim=1)
                return output , y # dim 200
        
        else:
            self.mnist_model = MnistNet()
            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                y = self.dropout2(x)
                x = self.fc2(y)
                output = F.log_softmax(x, dim=1)
                return output , y # dim 128

        self.mnist_model.load_state_dict(torch.load(mnist_model_path))
        self.mnist_model.eval()
        for p in self.mnist_model.parameters():
            p.requires_grad = False
        

        self.mnist_model.forward = MethodType(forward, self.mnist_model)
        


    def embed(self, inputs: Sequence[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        if self.mnist_model.training:
            self.mnist_model.eval()
            
        return self.mnist_model(x)[1]

