from tensorflow.keras import Model, layers
from util import *

from regression import Regression

class InPlaneRegression(Regression):
    def __init__(self, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(2, feature_size, num_anchors, num_classes)

class DepthRegression(Regression):
    def __init__(self, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(1, feature_size, num_anchors, num_classes)

class AnchorProposal(Regression):
    def __init__(self, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(1, feature_size, num_anchors, num_classes)


from resnet.bottleneck_block import BottleneckBlock
from resnet.resnet import ResNet

class A2J(Model):
    def __init__(self, num_classes):
        super(A2J, self).__init__()

        self.backbone = ResNet(BottleneckBlock, [3,4,6,3])
        self.in_plane_regression = InPlaneRegression(num_classes=num_classes)
        self.depth_regression = DepthRegression(num_classes=num_classes)
        self.anchor_proposal = AnchorProposal(num_classes=num_classes)

    def call(self, inputs):
        x3, x4 = self.backbone(inputs)
        coords = self.in_plane_regression(x4)
        depth = self.depth_regression(x4)
        anchors = self.anchor_proposal(x3)

        return anchors, coords, depth


# class A2J(object):
#     def __init__(self, input_shape, num_classes, predict_3D = True):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.predict_3D = predict_3D
#
#     def build(self):
#         inputs = layers.Input(shape = self.input_shape)
#         backbone = ResNet(BottleneckBlock, [3,4,6,3])
#         x3, x4 = backbone(inputs)
#
#         if self.predict_3D:
#             anchor_proposal_out = AnchorProposal(num_classes=self.num_classes)(x3)
#             in_plane_regression_out = InPlaneRegression(num_classes=self.num_classes)(x4)
#             depth_regression_out = DepthRegression(num_classes=self.num_classes)(x4)
#
#             self.model = models.Model(inputs = inputs, outputs = [anchor_proposal_out, in_plane_regression_out, depth_regression_out])
#
#         else:
#             anchor_proposal = A2J_AnchorProposal(x3, num_classes=self.num_classes)
#             anchor_proposal.build()
#
#             in_plane_regression = A2J_InPlaneRegression(x4, num_classes=self.num_classes)
#             in_plane_regression.build()
#
#             self.model = models.Model(inputs = inputs, outputs = [anchor_proposal.outputs, in_plane_regression.outputs])
#
#
#     def compile(self):
#         pass
input_shape = (256, 256, 1)
inputs = layers.Input(shape = input_shape)
a2j = A2J(15)
outputs = A2J(inputs)
