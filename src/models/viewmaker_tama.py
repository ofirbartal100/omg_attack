'''Core architecture and functionality of the viewmaker network.

Adapted from the transformer_net.py example below, using methods proposed in Johnson et al. 2016

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''
import torch
import torch.nn as nn
from torch.nn import functional as init
import torch_dct as dct
from dabs.src.datasets.specs import Input2dSpec
from original_vm.viewmaker.src.models.viewmaker import Viewmaker
from dabs.src.models.transformer import AttentionBlock

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'silu': torch.nn.SiLU
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

        # y = self.add_noise_channel(y, bound_multiplier=bound_multiplier,shared_seeds_generator=shared_seeds_generator) # y-> (b,4,32,32)
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
    


class ViewmakerTAMA38_2(Viewmaker):
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

        # Initial convolution layers
        self.conv1 = ConvLayer(self.num_channels, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)

        self.block1 = nn.ModuleList([self.conv1,self.in1])
        self.block2 = nn.ModuleList([self.conv2,self.in2])
        self.block3 = nn.ModuleList([self.conv3,self.in3])
        
        # Residual layers have +N for added random channels
        self.res1 = ResidualBlock(128, activation='silu')
        self.res2 = ResidualBlock(128, activation='silu')
        self.res3 = ResidualBlock(128, activation='silu')
        self.res4 = ResidualBlock(128, activation='silu')
        self.res5 = ResidualBlock(128, activation='silu')

        self.bottleneck = nn.ModuleList([self.res1,self.res2,self.res3,self.res4,self.res5])
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, self.num_channels, kernel_size=9, stride=1)

        self.up1 = nn.ModuleList([self.deconv1 , self.in4 ])
        self.up2 = nn.ModuleList([self.deconv2 , self.in5 ])
        self.up3 = nn.ModuleList([self.deconv3])
        
        # Upsampling Layers
        # self.deconv2_2 = ConvLayer(64+64, 64, kernel_size=3, stride=1)
        # self.deconv3_2 = ConvLayer(32+32, 32, kernel_size=3, stride=1)
        # self.in4_2 = torch.nn.InstanceNorm2d(64, affine=True)
        # self.in5_2 = torch.nn.InstanceNorm2d(32, affine=True)

        # self.attn = AttentionBlock(64,8)
        # self.attn2 = AttentionBlock(128,8)


    # def get_delta(self, y_pixels, eps=1e-4):
    #     '''Constrains the input perturbation by projecting it onto an L1 sphere'''
    #     # distortion_budget = self.distortion_budget
    #     # delta = torch.tanh(y_pixels) # Project to [-1, 1]
    #     # avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
    #     # max_magnitude = distortion_budget
    #     # delta = delta * max_magnitude / (avg_magnitude + eps)
    #     return y_pixels

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1,shared_seeds_generator=None):

        for b1 in self.block1:# y-> (b,32,32,32)
            y = b1(y)
        y = init.silu(y)
        skip1 = y

        for b2 in self.block2:# y-> (b,64,16,16)
            y = b2(y)
        y = init.silu(y)
        skip2 = y

        for b3 in self.block3:# y-> (b,128,8,8)
            y = b3(y)
        y = init.silu(y)

        for btl in self.bottleneck:# y-> (b,128,8,8)
            y = btl(y)

        y = self.deconv1(y)
        y = y + torch.normal(torch.zeros_like(y),torch.ones_like(y))
        y = self.in4(y) #+ skip2
        y = init.silu(y)

        y = self.deconv2(y)
        y = y  + torch.normal(torch.zeros_like(y),torch.ones_like(y))
        y = self.in5(y) #+ skip1
        y = init.silu(y)

        y = self.deconv3(y)

        return y , None
    

class BottleNeck(torch.nn.Module):
    def __init__(self, channels, activation='silu'):
        super(BottleNeck, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.attn = AttentionBlock(64,8)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv3 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.act = ACTIVATIONS[activation]()


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out