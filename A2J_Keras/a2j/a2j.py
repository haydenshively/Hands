from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers
from util import *

from a2j.regression import Regression
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
"""OOP style"""
class A2J(Model):
    def __init__(self, num_classes):
        super(A2J, self).__init__()

        self.backbone = ResNet(BottleneckBlock, [3,4,6,3])
        self.predict_offset = InPlaneRegression(num_classes=num_classes)
        self.predict_depth = DepthRegression(num_classes=num_classes)
        self.response_map = AnchorProposal(num_classes=num_classes)

    def call(self, inputs):
        x3, x4 = self.backbone(inputs)
        offsets = self.predict_offset(x4)
        depths = self.predict_depth(x4)
        responses = self.response_map(x3)

        range = K.range(0, 4, 1)
        anchor_pos = layers.Concatenate(axis=2)(tf.meshgrid(range, range))
        joint_pos = layers.Add()([anchor_pos, offsets])
        joint_pos = layers.Multiply()([joint_pos, responses])

        anchor_heatmap = layers.Multiply()([anchor_pos, responses])

        depths = layers.Multiply()([depths, responses])

        return joint_pos, depths, anchor_heatmap





from a2j.regression import regression
def in_plane_regression(input_tensor, feature_size=256, num_anchors=16, num_classes=15):
    return regression(input_tensor, 2, feature_size, num_anchors, num_classes)
def depth_regression(input_tensor, feature_size=256, num_anchors=16, num_classes=15):
    return regression(input_tensor, 1, feature_size, num_anchors, num_classes)
def anchor_proposal(input_tensor, feature_size=256, num_anchors=16, num_classes=15):
    return regression(input_tensor, 1, feature_size, num_anchors, num_classes)

import tensorflow as tf

from resnet.bottleneck_block import bottleneck_block
from resnet.resnet import resnet
"""Functional style"""
def a2j(input_tensor, num_classes):
    x3, x4 = resnet(input_tensor, bottleneck_block, [3,4,6,3])
    responses = anchor_proposal(x3, num_classes=num_classes)
    offsets = in_plane_regression(x4, num_classes=num_classes)
    depths = depth_regression(x4, num_classes=num_classes)

    return post_process(responses, offsets, depths)
