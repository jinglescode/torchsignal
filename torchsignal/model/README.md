# Models

## Multitask Model

Using multi-task learning to capture signals simultaneously from the fovea efficiently and the neighboring targets in the peripheral vision generate a visual response map. A calibration-free user-independent solution, desirable for clinical diagnostics. A stepping stone for an objective assessment of glaucoma patients’ visual field. Learn more about this model at https://jinglescode.github.io/ssvep-multi-task-learning/.

```
from torchsignal.model import MultitaskSSVEP

model = MultitaskSSVEP(num_channel=10,
    num_classes=40,
    signal_length=1000,
    filters_n1=4,
    kernel_window_ssvep=59,
    kernel_window=19,
    conv_3_dilation=4,
    conv_4_dilation=4,
)

x = torch.ones((20, 10, 1000))
print("Input shape:", x.shape)
y = model(x)
print("Output shape:", y.shape)
```

## EEGNet (Compact)

EEGNet: Compact Convolutional Neural Network (Compact-CNN) https://arxiv.org/pdf/1803.04566.pdf

```
from torchsignal.model import CompactEEGNet

model = CompactEEGNet(
    num_channel=10,
    num_classes=4,
    signal_length=1000,
)

x = torch.ones((21, 10, 1000))
print("Input shape:", x.shape)
y = model(x)
print("Output shape:", y.shape)
```

## Performer

Rethinking Attention with Performers
https://arxiv.org/abs/2009.14794

```
from torchsignal.model import Performer

model = Performer(
    dim = 11,
    depth = 1,
    heads = 1,
    causal = True
)

x = torch.randn(1, 1000, 11)
print("Input shape:", x.shape) # torch.Size([1, 1000, 11])
y = model(x)
print("Output shape:", y.shape) # torch.Size([1, 1000, 11])
```

## WaveNet

WaveNet: A Generative Model for Raw Audio
https://arxiv.org/abs/1609.03499.

```
from torchsignal.model import WaveNet

model = WaveNet(
    layers=6,
    blocks=3,
    dilation_channels=32,
    residual_channels=32,
    skip_channels=1024,
    classes=9,
    end_channels=512, 
    bias=True
)

x = torch.randn(2, 9, 250)
print("Input shape:", x.shape) # torch.Size([2, 9, 250])
y = model(x)
print("Output shape:", y.shape) # torch.Size([2, 9, 128])
```

## MI-EEGNet
MI-EEGNET: A novel Convolutional Neural Network for motor imagery classification
https://www.sciencedirect.com/science/article/abs/pii/S016502702030460X

```
from torchsignal.model import MIEEGNet

model = MIEEGNet(num_channel=22, num_classes=12, signal_length=256)

x = torch.randn(1, 22, 256)
print("Input shape:", x.shape) # torch.Size([1, 22, 256])
y = model(x)
print("Output shape:", y.shape) # torch.Size([1, 12])
```
