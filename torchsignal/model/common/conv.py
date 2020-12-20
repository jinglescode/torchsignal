# -*- coding: utf-8 -*-
"""Common convolutions
"""
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from typing import List
import math


class Conv2d(nn.Module):
    """
    Input: 4-dim tensor
        Shape [batch, in_channels, H, W]
    Return: 4-dim tensor
        Shape [batch, out_channels, H, W]
        
    Args:
        in_channels : int
            Should match input `channel`
        out_channels : int
            Return tensor with `out_channels`
        kernel_size : int or 2-dim tuple
        stride : int or 2-dim tuple, default: 1
        padding : int or 2-dim tuple or True
            Apply `padding` if given int or 2-dim tuple. Perform TensorFlow-like 'SAME' padding if True
        dilation : int or 2-dim tuple, default: 1
        groups : int or 2-dim tuple, default: 1
        w_in: int, optional
            The size of `W` axis. If given, `w_out` is available.
    
    Usage:
        x = torch.randn(1, 22, 1, 256)
        conv1 = Conv2dSamePadding(22, 64, kernel_size=17, padding=True, w_in=256)
        y = conv1(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, groups=1, w_in=None):
        super().__init__()
        
        padding = padding
        self.kernel_size = kernel_size = kernel_size
        self.stride = stride = stride
        self.dilation = dilation = dilation
        
        self.padding_same = False
        if padding == "SAME":
            self.padding_same = True
            padding = (0,0)
        
        if isinstance(padding, int):
            padding = (padding, padding)
            
        if isinstance(kernel_size, int):
            self.kernel_size = kernel_size = (kernel_size, kernel_size)
            
        if isinstance(stride, int):
            self.stride = stride = (stride, stride)
        
        if isinstance(dilation, int):
            self.dilation = dilation = (dilation, dilation)
            
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0 if padding==True else padding, 
            dilation=dilation, 
            groups=groups
        )
        
        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )
        if self.padding_same == "SAME": # if SAME, then replace, w_out = w_in, obviously
            self.w_out = w_in
            
    def forward(self, x):
        if self.padding_same == True:
            x = self.pad_same(x, self.kernel_size, self.stride, self.dilation)
        return self.conv(x)
    
    # Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
    def get_same_padding(self, x: int, k: int, s: int, d: int):
        return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

    # Dynamically pad input x with 'SAME' padding for conv with specified args
    def pad_same(self, x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
        ih, iw = x.size()[-2:]
        pad_h, pad_w = self.get_same_padding(ih, k[0], s[0], d[0]), self.get_same_padding(iw, k[1], s[1], d[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
        return x


class Conv2dBlockELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, activation=nn.ELU, w_in=None):
        super(Conv2dBlockELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

        if w_in is not None:
            self.w_out = int( ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1) / 1) + 1 )

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=1, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
            
        if isinstance(kernel_size, tuple):
            padding = (
                kernel_size[0]//2 if kernel_size[0]-1 != 0 else 0,
                kernel_size[1]//2 if kernel_size[1]-1 != 0 else 0
            )
            
        self.depthwise = DepthwiseConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SelfAttention(nn.Module):
    """Self attention layer
    
    Inputs :
        x : input feature maps( B X C X W X H)
    Returns :
        out : self attention value + input feature 
        attention: B X N X N (N is Width*Height)
    Usage:
        selfattn = SelfAttention(11)
        x = torch.randn(2, 11, 128, 128)
        print("Input shape:", x.shape)
        y, attention = selfattn(x)
        print("Output shape:", y.shape)
        print("Attention shape:", attention.shape)
    """
    def __init__(self, in_dim, activation=nn.ReLU):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention

    
##### 
# Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

    
class TemporalConvNet(nn.Module):
    """
    TCN layer
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    https://arxiv.org/abs/1803.01271
    https://github.com/locuslab/TCN
    
    Usage: 
        tcn = TemporalConvNet(
            num_channels=11
        )
        x = torch.randn(2, 11, 250)
        print("Input shape:", x.shape)
        y = tcn(x)
        print("Output shape:", y.shape)

    """
    def __init__(self, num_channels, kernel_size=7, dropout=0.1, nhid=32, levels=8):
        super(TemporalConvNet, self).__init__()
        
        channel_sizes = [nhid] * levels
        
        layers = []
        num_levels = len(channel_sizes)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels if i == 0 else channel_sizes[i-1]
            out_channels = channel_sizes[i]
            layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
