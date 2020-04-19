import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import h_sigmoid, h_swish
from blocks import SqueezeExcite, Bottleneck

# input should be of shape [x, 3, 224, 224]
class MobileNetV3(nn.Module):
    def __init__(self, config, truncated=False, num_classes=1000):
        super(MobileNetV3, self).__init__()

        inp = 16
        self.conv1 = nn.Conv2d(3, inp, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.hs1 = h_swish()

        bneck = []
        for k, s, exp, oup, nl, se in config:
            bneck.append(Bottleneck(k, s, inp, exp, oup, nl, se))
            inp = oup
        self.bneck = nn.Sequential(*bneck)

        inp = config[-1][2]
        self.conv2 = nn.Conv2d(oup, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inp)
        self.hs2 = h_swish()

        if not truncated:
            self.linear3 = nn.Linear(inp, 1280)
            self.bn3 = nn.BatchNorm1d(1280)
            self.hs3 = h_swish()
            self.linear4 = nn.Linear(1280, num_classes)

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
        x = self.hs2(self.bn2(self.conv2(x)))
        if not truncated:
            x = F.avg_pool2d(x, 7)
            x = x.view(x.size(0), -1)
            x = self.hs3(self.bn3(self.linear3(x)))
            x = self.linear4(x)
        return x

def mobilenetv3_large(**kwargs):
    config = [
        #k, s, ex, out,  nl,                    se
        [3, 1, 16,  16,  nn.ReLU(inplace=True), nn.Identity()],
        [3, 2, 64,  24,  nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 72,  24,  nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 72,  40,  nn.ReLU(inplace=True), SqueezeExcite(40)],
        [5, 1, 120, 40,  nn.ReLU(inplace=True), SqueezeExcite(40)],
        [5, 1, 120, 40,  nn.ReLU(inplace=True), SqueezeExcite(40)],
        [3, 2, 240, 80,  h_swish(), nn.Identity()],
        [3, 1, 200, 80,  h_swish(), nn.Identity()],
        [3, 1, 184, 80,  h_swish(), nn.Identity()],
        [3, 1, 184, 80,  h_swish(), nn.Identity()],
        [3, 1, 480, 112, h_swish(), SqueezeExcite(112)],
        [3, 1, 672, 112, h_swish(), SqueezeExcite(112)],
        [5, 1, 672, 160, h_swish(), SqueezeExcite(160)],
        [5, 2, 672, 160, h_swish(), SqueezeExcite(160)],
        [5, 1, 960, 160, h_swish(), SqueezeExcite(160)]
    ]
    return MobileNetV3(config, **kwargs)


def mobilenetv3_small(**kwargs):
    config = [
        #k, s, ex, out, nl,                    se
        [3, 2, 16, 16,  nn.ReLU(inplace=True), SqueezeExcite(16)],
        [3, 2, 72, 24,  nn.ReLU(inplace=True), nn.Identity()],
        [3, 1, 88, 24,  nn.ReLU(inplace=True), nn.Identity()],
        [5, 2, 96, 40,  h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, h_swish(), SqueezeExcite(40)],
        [5, 1, 240, 40, h_swish(), SqueezeExcite(40)],
        [5, 1, 120, 48, h_swish(), SqueezeExcite(48)],
        [5, 1, 144, 48, h_swish(), SqueezeExcite(48)],
        [5, 2, 288, 96, h_swish(), SqueezeExcite(96)],
        [5, 1, 576, 96, h_swish(), SqueezeExcite(96)],
        [5, 1, 576, 96, h_swish(), SqueezeExcite(96)]
    ]
    return MobileNetV3(config, **kwargs)


if __name__ == '__main__':
    net = mobilenetv3_small()
    # from torchsummary import summary
    # summary(net, (3, 256, 256))

    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    saved = torch.load('mbv3_small.pth.tar', map_location=torch.device('cpu'))

    from collections import OrderedDict
    state_dict = OrderedDict()
    for key in saved['state_dict']:
        if key.startswith('module.'):
            state_dict[key[7:]] = saved['state_dict'][key]
        else:
            state_dict[key] = saved['state_dict'][key]

    net.load_state_dict(state_dict)

    from torchvision import transforms
    loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    from PIL import Image
    image = Image.open('/Users/haydenshively/Desktop/watersnake.jpg')
    image = loader(image).float()
    image = image.unsqueeze(0)

    net.eval()
    res = net(image).data.numpy()[0]
    a = res.argmax()
    res[a] = -1
    b = res.argmax()
    res[b] = -1
    c = res.argmax()
    res[c] = -1
    d = res.argmax()

    print((a,b,c,d))
