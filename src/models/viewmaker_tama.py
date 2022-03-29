'''Core architecture and functionality of the viewmaker network.

Adapted from the transformer_net.py example below, using methods proposed in Johnson et al. 2016

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''
import torch
import torch.nn as nn
from torch.nn import functional as init
import torch_dct as dct
from original_vm.viewmaker.src.models.viewmaker import Viewmaker, ConvLayer

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}


class ViewmakerTAMA38(Viewmaker):
    '''Viewmaker network that stochastically maps a multichannel 2D input to an output of the same size.'''
    def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu',  
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=3):
        '''Initialize the Viewmaker network.

        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        super().__init__()
        
        # Upsampling Layers
        self.deconv2_2 = ConvLayer(64+64, 64, kernel_size=3, stride=1)
        self.deconv3_2 = ConvLayer(32+32, 32, kernel_size=3, stride=1)
        self.in4_2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.in5_2 = torch.nn.InstanceNorm2d(32, affine=True)


    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1,shared_seeds_generator=None):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier,shared_seeds_generator=shared_seeds_generator) # y-> (b,4,32,32)
        y = self.act(self.in1(self.conv1(y))) # y-> (b,32,32,32)
        skip1 = y
        y = self.act(self.in2(self.conv2(y))) # y-> (b,64,16,16)
        skip2 = y
        y = self.act(self.in3(self.conv3(y))) # y-> (b,128,8,8)

        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))

        y = self.act(self.in4(self.deconv1(y))) # y-> (batch,64,16,16)
        y = self.act(self.in4_2(self.deconv2_2(torch.cat([y, skip2],dim=1))))
        y = self.act(self.in5(self.deconv2(y)))
        y = self.act(self.in5_2(self.deconv3_2(torch.cat([y, skip1],dim=1))))
        y = self.deconv3(y)

        return y, features
    
