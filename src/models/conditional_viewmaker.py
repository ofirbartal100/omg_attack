
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as init
import torch_dct as dct
from original_vm.viewmaker.src.models.viewmaker import Viewmaker,UpsampleConvLayer


class ConditionalViewmaker(Viewmaker):
    def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu', clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=3):
        super().__init__(num_channels, distortion_budget, activation,clamp, frequency_domain, downsample_to, num_res_blocks)
        self.deconv1 = UpsampleConvLayer(128 + self.num_res_blocks + 1, 64, kernel_size=3, stride=1, upsample=2)

    def basic_net(self, y,conditions_indicies=None, num_res_blocks=5, bound_multiplier=1,shared_seeds_generator=None):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier,shared_seeds_generator=shared_seeds_generator) # y-> (b,4,32,32)
        y = self.act(self.in1(self.conv1(y))) # y-> (b,32,32,32)
        y = self.act(self.in2(self.conv2(y))) # y-> (b,64,16,16)
        y = self.act(self.in3(self.conv3(y))) # y-> (b,128,8,8)
        
        x_j = y.mean(1,keepdim=True)[conditions_indicies] # add conditional generation with x_j encoding

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))
                

        # add conditional generation with x_j encoding
        y = [self.res1, self.res2, self.res3, self.res4, self.res5][num_res_blocks](torch.cat([y,x_j],dim=1))
        

        y = self.act(self.in4(self.deconv1(y))) # y-> (batch,64,16,16)
        y = self.act(self.in5(self.deconv2(y))) # y-> (batch,32,32,32)
        y = self.deconv3(y)

        return y, features

    def forward(self, x, conditions_indicies=None , shared_seeds_generator=None):
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')
        y = x
        
        if self.frequency_domain:
            # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
            # Uses the Discrete Cosine Transform.
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        if conditions_indicies is None: # roll 1 right
            conditions_indicies = [ i for i in range(len(y))]
            conditions_indicies = conditions_indicies[1:] + conditions_indicies[0:1]

        y_pixels, features = self.basic_net(y, conditions_indicies, self.num_res_blocks, bound_multiplier=1,shared_seeds_generator=shared_seeds_generator)
        delta = self.get_delta(y_pixels)
        if self.frequency_domain:
            # Compute inverse DCT from frequency domain to time domain.
            delta = dct.idct_2d(delta)
        if self.downsample_to:
            # Upsample.
            x = x_orig
            delta = torch.nn.functional.interpolate(delta, size=x_orig.shape[-2:], mode='bilinear')

        # Additive perturbation
        result = x + delta
        if self.clamp:
            result = torch.clamp(result, 0, 1.0)

        return result , conditions_indicies


