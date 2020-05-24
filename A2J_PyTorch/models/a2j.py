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


from .mobilenetv3 import MobileNetV3
class MNV3Backbone(MobileNetV3):
    def __init__(self, config, index_C4):
        super().__init__(config, index_C4, truncated=True)
    def forward(self, x):
        # get depth and clone across channels for compatibility
        n, c, h, w = x.size()
        x = x.expand(n, 3, h, w)
        # now run it through the model
        # C4 layer should be 16x downsampling
        # C5 layer should be 32x downsampling
        x = self.hs1(self.bn1(self.conv1(x)))
        x, c4 = self.bneck(x)
        x = self.bneck2(x)[0]
        c5 = self.hs2(self.bn2(self.conv2(x)))
        return c4, c5


class A2J(nn.Module):
    def __init__(self, backbone, num_classes, is_3D=True):
        super(A2J, self).__init__()
        self.is_3D = is_3D

        self.backbone = backbone
        self.response_map = AnchorProposal(288, num_classes=num_classes)
        self.predict_offset = InPlaneRegression(288, num_classes=num_classes)
        if self.is_3D:
            self.predict_depth = DepthRegression(288, num_classes=num_classes)

    def forward(self, x):
        c4, c5 = self.backbone(x)

        responses = self.response_map(c4)
        offsets = self.predict_offset(c5)
        if self.is_3D:
            depths = self.predict_depth(c5)
            return (responses, offsets, depths)

        return (responses, offsets)
