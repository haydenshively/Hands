from tensorflow.keras import layers
from resnet.util import *
from util import *

"""OOP style"""
class BasicBlock(layers.Layer):
    def __init__(self, filters_in, filters_out, stride=(1,1), dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(filters_out, stride)
        self.conv2 = conv3x3(filters_out, dilation=dilation)
        self.shortcut = conv1x1(filters_out, stride)

        self.should_use_shortcut = (filters_in != filters_out) or (stride is not (1,1))


    def call(self, inputs):
        x = self.conv1(inputs)
        x = bn()(x)
        x = relu()(x)
        x = self.conv2(x)
        x = bn()(x)

        if self.should_use_shortcut:
            s = self.shortcut(inputs)
            s = bn()(s)
        else:
            s = inputs

        x = layers.add([x, s])
        x = relu()(x)
        return x




"""Functional style"""
def basic_block(input_tensor, filters_in, filters_out, stride=(1,1), dilation=1):
    # conv 1 includes the stride
    x = conv3x3(filters_out, stride)(input_tensor)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    # conv 2 includes the dilation
    x = conv3x3(filters_out, dilation=dilation)(x)
    x = layers.BatchNormalization(axis=3)(x)

    if (stride is not (1,1)) or (filters_in != filters_out):
        shortcut = conv1x1(1*filters_out, stride)(input_tensor)
        shortcut = layers.BatchNormalization(axis=3)(shortcut)
    else:
        shortcut = input_tensor

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x
