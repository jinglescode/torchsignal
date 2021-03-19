import torch
from torch import nn
from .common.conv import Conv2dBlockELU


class MultitaskSSVEPClassifier(nn.Module):
    """
    Using multi-task learning to capture signals simultaneously from the fovea efficiently and the neighboring targets in the peripheral vision generate a visual response map. A calibration-free user-independent solution, desirable for clinical diagnostics. A stepping stone for an objective assessment of glaucoma patients’ visual field.
    Learn more about this model at https://jinglescode.github.io/ssvep-multi-task-learning/
    This model is a multi-label model. Although it produces multiple outputs, we also used this model to get our multi-class results in our paper.
    
    Usage:
        model = MultitaskSSVEPClassifier(
            num_channel=11,
            num_classes=40,
            signal_length=250,
        )

        x = torch.randn(2, 11, 250)
        print("Input shape:", x.shape) # torch.Size([2, 11, 250])
        y = model(x)
        print("Output shape:", y.shape) # torch.Size([2, 40])
        
    Cite:
        @inproceedings{khok2020deep,
          title={Deep Multi-Task Learning for SSVEP Detection and Visual Response Mapping},
          author={Khok, Hong Jing and Koh, Victor Teck Chang and Guan, Cuntai},
          booktitle={2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
          pages={1280--1285},
          year={2020},
          organization={IEEE}
        }
    """
    
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, filters_n1=4, kernel_window_ssvep=59, kernel_window=19, conv_3_dilation=4, conv_4_dilation=4):
        super().__init__()

        filters = [filters_n1, filters_n1 * 2]
        self.num_classes = num_classes

        self.conv_1 = Conv2dBlockELU(in_channels=1, out_channels=filters[0], kernel_size=(1, kernel_window_ssvep), w_in=signal_length)
        self.conv_2 = Conv2dBlockELU(in_channels=filters[0], out_channels=filters[0], kernel_size=(num_channel, 1))
        self.conv_3 = Conv2dBlockELU(in_channels=filters[0], out_channels=filters[1], kernel_size=(1, kernel_window), padding=(0,conv_3_dilation-1), dilation=(1,conv_3_dilation), w_in=self.conv_1.w_out)
        self.conv_4 = Conv2dBlockELU(in_channels=filters[1], out_channels=filters[1], kernel_size=(1, kernel_window), padding=(0,conv_4_dilation-1), dilation=(1,conv_4_dilation), w_in=self.conv_3.w_out)
        self.conv_mtl = multitask_block(filters[1]*num_classes, num_classes, kernel_size=(1, self.conv_4.w_out))
        self.classify = nn.Conv1d(num_classes, num_classes, kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.unsqueeze(x,1)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.dropout(x)

        x = self.conv_mtl(x)
        x = self.classify(x)
        x = x.view(-1, self.num_classes)
        return x
    
    
class multitask_block(nn.Module):
    def __init__(self, in_ch, num_classes, kernel_size):
        super(multitask_block, self).__init__()
        self.num_classes = num_classes
        self.conv_mtl = nn.Conv2d(in_ch, num_classes*2, kernel_size=kernel_size, groups=num_classes)

    def forward(self, x):
        x = torch.cat(self.num_classes*[x], 1)
        x = self.conv_mtl(x)
        x = x.view(-1, self.num_classes, 2)
        return x
