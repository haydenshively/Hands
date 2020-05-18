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
    def __init__(self, config):
        super().__init__(config, truncated=True)
    def forward(self, x):
        # get depth and clone across channels for compatibility
        n, c, h, w = x.size()
        x = x.expand(n, 3, h, w)
        # now run it through the model
        x = self.hs1(self.bn1(self.conv1(x)))
        x3 = self.bneck(x)
        x4 = self.hs2(self.bn2(self.conv2(x3)))
        return x3, x4


class A2J(nn.Module):
    def __init__(self, backbone, num_classes, is_3D=True):
        super(A2J, self).__init__()
        self.is_3D = is_3D

        self.backbone = backbone
        self.response_map = AnchorProposal(96, num_classes=num_classes)# 288
        self.predict_offset = InPlaneRegression(576, num_classes=num_classes)
        if self.is_3D:
            self.predict_depth = DepthRegression(576, num_classes=num_classes)

    def forward(self, x):
        x3, x4 = self.backbone(x)

        responses = self.response_map(x3)
        offsets = self.predict_offset(x4)
        if self.is_3D:
            depths = self.predict_depth(x4)
            return (responses, offsets, depths)

        return (responses, offsets)
