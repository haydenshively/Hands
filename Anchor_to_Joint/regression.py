from tensorflow.keras import Model, layers
from util import *

"""OOP style"""
class Regression(Model):
    def __init__(self, output_dims, feature_size=256, num_anchors=16, num_classes=15):
        super(Regression, self).__init__()
        self.output_dims = output_dims
        self.feature_size = feature_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')
        self.conv2 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')
        self.conv3 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')
        self.conv4 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')
        self.conv5 = layers.Conv2D(filters=self.num_anchors*self.num_classes*self.output_dims, kernel_size=3, padding='same', kernel_initializer='glorot_normal')

        self.reshape = layers.Reshape((-1, self.num_anchors, self.num_classes, self.output_dims))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = bn()(x)
        x = relu()(x)
        x = self.conv2(x)
        x = bn()(x)
        x = relu()(x)
        x = self.conv3(x)
        x = bn()(x)
        x = relu()(x)
        x = self.conv4(x)
        x = bn()(x)
        x = relu()(x)

        x = self.conv5(x)
        x = self.reshape(x)
        return x
