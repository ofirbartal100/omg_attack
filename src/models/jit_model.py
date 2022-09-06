from typing import List, Dict, Any, Tuple, Sequence

import torch
import torch.nn as nn
from dabs.src.models.base_model import BaseModel
from MobileFaceNet.mobilefacenet import MobileFaceNet
from torchscope import scope

class JitModel(BaseModel):

    def __init__(self, jit_model_path ,input_specs: List[Dict[str, Any]]):
        super(JitModel, self).__init__(None)
        self.jit_model_path = jit_model_path
        self.jit_model = MobileFaceNet()
        scope(self.jit_model,input_size=(3,112,112))
        self.jit_model.load_state_dict(torch.load(jit_model_path))
        for p in self.jit_model.parameters():
            p.requires_grad = False


    def embed(self, inputs: Sequence[torch.Tensor]):
        x = torch.cat(inputs, dim=0)
        return x

    def encode(self, x: torch.Tensor, **kwargs):
        return self.jit_model(x)

