import torch.nn as nn

from activations import h_sigmoid

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcite, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            h_sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
