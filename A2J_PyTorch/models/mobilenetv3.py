import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import h_sigmoid, h_swish
from blocks import SqueezeExcite, Bottleneck

# input should be of shape [x, 3, 224, 224]
class MobileNetV3(nn.Module):
    def __init__(self, config, index_C4=None, truncated=False, num_classes=1000):
        super(MobileNetV3, self).__init__()

        inp = 16
        self.conv1 = nn.Conv2d(3, inp, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.hs1 = h_swish()

        bneck = []
        for k, s, exp, oup, dil, nl, se in config:
            bneck.append(Bottleneck(k, s, inp, exp, oup, nl, se, dil))
            inp = oup
        if index_C4 is not None:
            # I would love to name this bneck1, but that would break compatibility
            # with pretrained model
            self.bneck = nn.Sequential(*bneck[:index_C4])
            self.bneck2 = nn.Sequential(*bneck[index_C4:])
        else:
            self.bneck = nn.Sequential(*bneck)
            self.bneck2 = None

        inp = config[-1][2]
        self.conv2 = nn.Conv2d(oup, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inp)
        self.hs2 = h_swish()

        if not truncated:
            self.linear3 = nn.Linear(inp, 1280)
            self.bn3 = nn.BatchNorm1d(1280)
            self.hs3 = h_swish()
            self.linear4 = nn.Linear(1280, num_classes)
        self.truncated = truncated

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.bneck(x)
        if self.bneck2 is not None:
            x = self.bneck2(x)
        x = self.hs2(self.bn2(self.conv2(x)))
        if not self.truncated:
            x = F.avg_pool2d(x, 7)
            x = x.view(x.size(0), -1)
            x = self.hs3(self.bn3(self.linear3(x)))
            x = self.linear4(x)
        return x

def mobilenetv3_large(**kwargs):
    config = [
        #k, s, ex, out,  dil,   nl,                    se
        [3, 1, 16,  16,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [3, 2, 64,  24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 72,  24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 72,  40,  1,     nn.ReLU(inplace=True), SqueezeExcite(40)],
        [5, 1, 120, 40,  1,     nn.ReLU(inplace=True), SqueezeExcite(40)],
        [5, 1, 120, 40,  1,     nn.ReLU(inplace=True), SqueezeExcite(40)],
        [3, 2, 240, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 200, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 184, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 184, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 480, 112, 1,     h_swish(), SqueezeExcite(112)],
        [3, 1, 672, 112, 1,     h_swish(), SqueezeExcite(112)],
        [5, 1, 672, 160, 1,     h_swish(), SqueezeExcite(160)],
        [5, 2, 672, 160, 1,     h_swish(), SqueezeExcite(160)],
        [5, 1, 960, 160, 1,     h_swish(), SqueezeExcite(160)]
    ]
    return MobileNetV3(config, **kwargs)

def mobilenetv3_large_backbone():
    config = [
        #k, s, ex, out,  dil,   nl,                    se
        [3, 1, 16,  16,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [3, 2, 64,  24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 72,  24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 72,  40,  1,     nn.ReLU(inplace=True), SqueezeExcite(40)],
        [5, 1, 120, 40,  1,     nn.ReLU(inplace=True), SqueezeExcite(40)],
        [5, 1, 120, 40,  1,     nn.ReLU(inplace=True), SqueezeExcite(40)],
        [3, 2, 240, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 200, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 184, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 184, 80,  1,     h_swish(), nn.Identity()],
        [3, 1, 480, 112, 1,     h_swish(), SqueezeExcite(112)],
        [3, 1, 672, 112, 1,     h_swish(), SqueezeExcite(112)],
        [5, 1, 672, 160, 1,     h_swish(), SqueezeExcite(160)],
        [5, 2, 336, 160, 1,     h_swish(), SqueezeExcite(160)],
        [5, 1, 480, 160, 1,     h_swish(), SqueezeExcite(160)]
    ]
    # C4 layer is the one right after bneck
    # C5 layer is the one just before pooling
    return MobileNetV3(config, index_C4=13, truncated=True)

def mobilenetv3_small(**kwargs):
    config = [
        #k, s, ex, out, dil,   nl,                    se
        [3, 2, 16, 16,  1,     nn.ReLU(inplace=True), SqueezeExcite(16)],
        [3, 2, 72, 24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 88, 24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 96, 40,  1,     h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, 1,     h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, 1,     h_swish(), SqueezeExcite(40)],
        [5, 1, 120, 48, 1,     h_swish(), SqueezeExcite(48)],
        [5, 1, 144, 48, 1,     h_swish(), SqueezeExcite(48)],
        [5, 2, 288, 96, 1,     h_swish(), SqueezeExcite(96)],
        [5, 1, 576, 96, 1,     h_swish(), SqueezeExcite(96)],
        [5, 1, 576, 96, 1,     h_swish(), SqueezeExcite(96)]
    ]
    return MobileNetV3(config, **kwargs)

def mobilenetv3_small_backbone():
    config = [
        #k, s, ex, out, dil,   nl,                    se
        [3, 2, 16, 16,  1,     nn.ReLU(inplace=True), SqueezeExcite(16)],
        [3, 2, 72, 24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 88, 24,  1,     nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 96, 40,  1,     h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, 1,     h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, 1,     h_swish(), SqueezeExcite(40)],
        [5, 1, 120, 48, 1,     h_swish(), SqueezeExcite(48)],
        [5, 1, 144, 48, 1,     h_swish(), SqueezeExcite(48)],
        [5, 2, 288, 96, 1,     h_swish(), SqueezeExcite(96)],
        [5, 1, 288, 96, 1,     h_swish(), SqueezeExcite(96)],
        [5, 1, 288, 96, 1,     h_swish(), SqueezeExcite(96)]
    ]
    # C4 layer is the one right after bneck
    # C5 layer is the one just before pooling
    return MobileNetV3(config, index_C4=9, truncated=True)# C5 layer is the one just before pooling
