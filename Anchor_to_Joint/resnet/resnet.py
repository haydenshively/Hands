from tensorflow.keras import Model, layers
import numpy as np
from resnet.util import *
from util import *

"""OOP style"""
class ResNetBlock(layers.Layer):
    def __init__(self, block, block_count, filters_in, filters_out, stride=(1,1), dilation=1):
        super(ResNetBlock, self).__init__()
        self.block1 = block(filters_in, filters_out, stride)
        filters_in[0]*=4

        self.blocks = []
        for _ in range(1, block_count):
            self.blocks.append(block(filters_in[0], filters_out, dilation=dilation))

    def __call__(self, inputs):
        x = self.block1(inputs)
        for block in self.blocks: x = block(x)
        return x

class ResNet(Model):
    def __init__(self, block, block_counts):
        super(ResNet, self).__init__()
        self.pad1 = layers.ZeroPadding2D(padding=(3,3))
        self.conv1 = layers.Conv2D(64, (7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal')
        self.pad2 = layers.ZeroPadding2D(padding=(1,1))
        self.pool1 = layers.MaxPooling2D((3,3), strides=(2,2))

        filters_in = np.array([64], dtype='uint32')

        self.block1 = ResNetBlock(block, block_counts[0], filters_in, 64)
        self.block2 = ResNetBlock(block, block_counts[1], filters_in, 128, stride=(2,2))
        self.block3 = ResNetBlock(block, block_counts[2], filters_in, 256, stride=(2,2))
        self.block4 = ResNetBlock(block, block_counts[3], filters_in, 512, dilation=2)

    def __call__(self, inputs):
        x = self.pad1(inputs)
        x = self.conv1(x)
        x = bn()(x)
        x = relu()(x)
        x = self.pad2(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x3)

        return x3, x4


"""Functional style"""
def resnet_layer(input_tensor, block_type, block_count, filters_in, filters_out, stride=(1,1), dilation=1):
    x = block_type(input_tensor, filters_in, filters_out, stride)
    for _ in range(1, block_count):
        x = block_type(x, filters_in, filters_out, dilation=dilation)
    return x

def resnet(input_tensor, block_type, layer_counts):
    x = layers.ZeroPadding2D(padding=(3,3))(input_tensor)
    x = layers.Conv2D(64, (7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=(1,1))(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)

    x = resnet_layer(x, block_type, layer_counts[0], 64, 64)
    x = resnet_layer(x, block_type, layer_counts[1], 64*4, 128, stride=(2,2))
    x3 = resnet_layer(x, block_type, layer_counts[2], 64*4*4, 256, stride=(2,2))
    x4 = resnet_layer(x3, block_type, layer_counts[3], 64*4*4*4, 512, dilation=2)

    return x3, x4
