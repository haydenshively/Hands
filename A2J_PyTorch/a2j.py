import torch.nn as nn

from blocks import Regression
class InPlaneRegression(Regression):
    def __init__(self, in_size, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(in_size, 2, feature_size, num_anchors, num_classes)
class DepthRegression(Regression):
    def __init__(self, in_size, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(in_size, 1, feature_size, num_anchors, num_classes)
class AnchorProposal(Regression):
    def __init__(self, in_size, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(in_size, 1, feature_size, num_anchors, num_classes)


from mobilenetv3 import MobileNetV3
class MNV3Backbone(MobileNetV3):
    def __init__(self, config):
        super().__init__(config, truncated=True)
    def forward(x):
        x = self.hs1(self.bn1(self.conv1(x)))
        x3 = self.bneck(x)
        x4 = self.hs2(self.bn2(self.conv2(x3)))
        return x3, x4


class A2J(nn.Module):
    def __init__(self, backbone, num_classes):
        super(A2J, self).__init__()

        self.backbone = backbone
        self.predict_offset = InPlaneRegression(576, num_classes=num_classes)
        self.predict_depth = DepthRegression(576, num_classes=num_classes)
        self.response_map = AnchorProposal(288, num_classes=num_classes)

    def forward(self, x):
        x3, x4 = self.backbone(x)

        offsets = self.predict_offset(x4)
        depths = self.predict_depth(x4)
        responses = self.response_map(x3)

        return (offsets, depths, responses)


if __name__ == '__main__':
    from activations import h_sigmoid, h_swish
    from blocks import SqueezeExcite

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
    backbone = MNV3Backbone(config)
    a2j = A2J(backbone, num_classes=15)
    # print(a2j)

    print('Total params: %.2fM' % (sum(p.numel() for p in a2j.parameters())/1000000.0))
