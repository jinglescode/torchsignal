import torch
from torch import nn
from .common.conv import DepthwiseConv2d, SeparableConv2d, Conv2d


class MIEEGNet(nn.Module):
    """
    Mobile Inception EEGNet model
    MI-EEGNET: A novel Convolutional Neural Network for motor imagery classification
    https://www.sciencedirect.com/science/article/abs/pii/S016502702030460X
    
    Usage:
        from torchsignal.model import MIEEGNet
        model = MIEEGNet(num_channel=22, num_classes=12, signal_length=256)
        x = torch.randn(1, 22, 256)
        print("Input shape:", x.shape) # torch.Size([1, 22, 256])
        y = model(x)
        print("Output shape:", y.shape) # torch.Size([1, 12])
        
    Note:
        1. My implementation, I did not get the same number of parameters as stated in the paper. This model has 106304 params instead of 162564 as stated in the paper.
        2. Somehow this kind of architecture only support `signal_length` with length of ^2 (128, 256, 512, etc).
        3. Also some certain `num_classes` might have issues.
    """
    def __init__(self, num_channel=22, num_classes=4, signal_length=256, depth=4, first_filter_size=64):
        super().__init__()
        
        self.num_classes = num_classes
        
        filter_size = [first_filter_size, first_filter_size*depth, first_filter_size*depth*4]
        
        self.conv1 = nn.Sequential(
            Conv2d(1, filter_size[0], kernel_size=(1, 16), padding="SAME"),
            nn.BatchNorm2d(filter_size[0]),
            DepthwiseConv2d(filter_size[0], filter_size[0]//depth, kernel_size=(num_channel, 1), depth=4, bias=True),
            nn.BatchNorm2d(filter_size[0]),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        
        self.convtower1 = nn.Sequential(
            Conv2d(filter_size[0], filter_size[0], kernel_size=(1, 1), padding="SAME"),
            SeparableConv2d(filter_size[0], filter_size[0], kernel_size=(1,7), bias=True),
            nn.BatchNorm2d(filter_size[0]),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            SeparableConv2d(filter_size[0], filter_size[0], kernel_size=(1,7), bias=True),
            nn.AvgPool2d(kernel_size=(1,2)),
        )
        
        self.convtower2 = nn.Sequential(
            Conv2d(filter_size[0], filter_size[0], kernel_size=(1, 1), padding="SAME"),
            SeparableConv2d(filter_size[0], filter_size[0], kernel_size=(1,9), bias=True),
            nn.BatchNorm2d(filter_size[0]),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            SeparableConv2d(filter_size[0], filter_size[0], kernel_size=(1,9), bias=True),
            nn.AvgPool2d(kernel_size=(1,2)),
        )
        
        self.convtower3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,2)),
            Conv2d(filter_size[0], filter_size[0], kernel_size=(1, 1), padding="SAME"),
        )
        
        self.convtower4 = nn.Sequential(
            Conv2d(filter_size[0], filter_size[0], kernel_size=(1, 1), stride=(1,2), padding="SAME"),
        )
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(filter_size[1]),
            nn.ELU(inplace=True),
            SeparableConv2d(filter_size[1], filter_size[1], kernel_size=(1,5), bias=True),
            nn.BatchNorm2d(filter_size[1]),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.AvgPool3d(kernel_size=(filter_size[1]//self.num_classes, 1, signal_length//4))
        )
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        
        x1 = self.convtower1(x)
        x2 = self.convtower2(x)
        x3 = self.convtower3(x)
        x4 = self.convtower4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        x = self.conv2(x)
        x = x.view(x.size()[0],-1)
        
        return x
