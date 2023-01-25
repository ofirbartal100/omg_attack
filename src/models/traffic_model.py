from typing import List, Dict, Any, Tuple, Sequence

import torch
import torch.nn as nn
from dabs.src.models.base_model import BaseModel
from gtsrb_pytorch.model import Net

class TrafficModel(BaseModel):

    def __init__(self, traffic_model_path ,input_specs: List[Dict[str, Any]]):
        super(TrafficModel, self).__init__(None)
        self.traffic_model_path = traffic_model_path
        self.traffic_model = Net()
        self.traffic_model.load_state_dict(torch.load(traffic_model_path))
        for p in self.traffic_model.parameters():
            p.requires_grad = False
        self.traffic_model.eval()


    def embed(self, inputs: Sequence[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        if self.traffic_model.training:
            self.traffic_model.eval()
            
        return self.traffic_model(x)


class TrafficModel80(BaseModel):

    def __init__(self, traffic_model_path ,input_specs: List[Dict[str, Any]]):
        super(TrafficModel80, self).__init__(None)
        self.traffic_model_path = traffic_model_path
        self.traffic_model = Net()
        classes_to_mask = [23,36,24,17,4,31,42,10]
        self.traffic_model.fc2 = nn.Linear(350, 43-len(classes_to_mask))

        self.traffic_model.load_state_dict(torch.load(traffic_model_path))
        for p in self.traffic_model.parameters():
            p.requires_grad = False
        self.traffic_model.eval()


    def embed(self, inputs: Sequence[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        if self.traffic_model.training:
            self.traffic_model.eval()
            
        return self.traffic_model(x)

