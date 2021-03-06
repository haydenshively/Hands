"""
MODEL DEFINITION
Inspiration from https://github.com/zhixuhao/unet/blob/master/model.py
"""
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from model import Model
class UNet(Model):
    def __init__(self, input_shape, num_class):
        super().__init__(input_shape)
        self.num_class = num_class

    def build(self):
        # NOTE: iPhone RGB resolution is 3088x2320
        # iPhone Depth Map resolution is 640x480
        # We should be able to designate variable input size for the network
        # by setting shape = (num_channels, None, None)
        inputs = layers.Input(shape = self.input_shape)#256x256

        conv1 = self._CommonConv(32, "conv1_1")(inputs)#254x254
        conv1 = self._CommonConv(32, "conv1_2")(conv1)#252x252
        pool1 = self._CommonPool()(conv1)#126x126

        conv2 = self._CommonConv(64, "conv2_1")(pool1)#124x124
        conv2 = self._CommonConv(64, "conv2_2")(conv2)#122x122
        pool2 = self._CommonPool()(conv2)#61x61

        conv3 = self._CommonConv(128, "conv3_1")(pool2)#59x59
        conv3 = self._CommonConv(128, "conv3_2")(conv3)#57x57
        drop3 = layers.Dropout(0.5)(conv3)

        up4 = self._CommonDeconv(64, "up4", drop3)
        merge4 = self._CommonMerge(conv2, up4)
        conv4 = self._CommonConv(64, "conv4_1")(merge4)
        conv4 = self._CommonConv(64, "conv4_2")(conv4)

        up5 = self._CommonDeconv(32, "up5", conv4)
        merge5 = self._CommonMerge(conv1, up5)
        conv5 = self._CommonConv(32, "conv5_1")(merge5)
        conv5 = self._CommonConv(32, "conv5_2")(conv5)

        conv6 = layers.Conv2D(filters=self.num_class, kernel_size=1, activation="sigmoid")(conv5)
        outputs = layers.Reshape((self.input_shape[0], self.input_shape[1]))(conv6)

        self.model = models.Model(inputs = inputs, outputs = outputs)
        super().build()

    def compile(self):
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
        super().compile()


    def _CommonConv(self, filters, name, kernel_size = 3):
        return layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same", kernel_initializer="he_normal", name=name)

    def _CommonPool(self):
        return layers.MaxPooling2D(pool_size = (2, 2))

    def _CommonDeconv(self, filters, name, inputs):
        up = layers.UpSampling2D(size = (2, 2))(inputs)
        return self._CommonConv(filters=filters, name=name, kernel_size=2)(up)

    def _CommonMerge(self, a, b):
        return layers.Concatenate(axis = 3)([a, b])
