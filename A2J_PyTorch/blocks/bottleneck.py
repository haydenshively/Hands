import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, kernel, stride, in_size, expand_size, out_size, nlin_layer, se_layer, dilation=1):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]

        # self.stride = stride
        self.use_shortcut = stride == 1

        """
        (TODO 1)
        Some versions of the MNV3 code only apply conv1, bn1, and nolinear1
        if in_size != out_size. I'm unsure what the official paper says.

        (TODO 2)
        Some versions of the MNV3 code place the SE layer between bn2 and nolinear2,
        whereas here we're applying it after bn3. I'm unsure what the official paper says.

        (TODO 3)
        Some versions of the MNV3 code don't use an extra Conv2d and BatchNorm2d for
        doing the shortcut. I'm unsure what the official paper says.
        """
        # pointwise
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nlin_layer
        # depthwise
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel, stride=stride, padding=kernel//2, dilation=dilation, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nlin_layer
        # pointwise-linear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        # squeeze and excitation
        self.se = se_layer
        # shortcut
        if self.use_shortcut:
            self.shortcut = nn.Identity() if in_size==out_size else nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        if isinstance(x, tuple): x = x[0]
        # pointwise
        exp = self.nolinear1(self.bn1(self.conv1(x)))
        # depthwise
        out = self.nolinear2(self.bn2(self.conv2(exp)))
        # pointwise-linear
        out = self.bn3(self.conv3(out))
        # squeeze and excitation
        out = self.se(out)
        out = out + self.shortcut(x) if self.use_shortcut else out
        return out, exp
