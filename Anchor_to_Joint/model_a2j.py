from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from model import Model
class A2J_Regression(Model):
    def __init__(self, output_dims, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(None)
        self.output_dims = output_dims
        self.feature_size = feature_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def build(self):
        self.model = models.Model(inputs = self._inputs, outputs = self._outputs)
        super().build()

    def compile(self):
        super().compile()

    def __call__(self, value):
        self.inputs = value
        return self.outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value
        self.input_shape = self._inputs.shape[1:]

        conv1 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')(self._inputs)
        bn1 = layers.BatchNormalization(axis=3)(conv1)# , kernel_initializer='ones', bias_initializer='zeros'
        act1 = layers.ReLU()(bn1)

        conv2 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')(act1)
        bn2 = layers.BatchNormalization(axis=3)(conv2)
        act2 = layers.ReLU()(bn2)

        conv3 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')(act2)
        bn3 = layers.BatchNormalization(axis=3)(conv3)
        act3 = layers.ReLU()(bn3)

        conv4 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='same', kernel_initializer='glorot_normal')(act3)
        bn4 = layers.BatchNormalization(axis=3)(conv4)
        act4 = layers.ReLU()(bn4)

        conv5 = layers.Conv2D(filters=self.num_anchors*self.num_classes*self.output_dims, kernel_size=3, padding='same', kernel_initializer='glorot_normal')(act4)
        # TODO this reshape may not actually work
        self._outputs = layers.Reshape((-1, self.num_classes, self.output_dims))(conv5)

    @property
    def outputs(self):
        return self._outputs


class A2J_InPlaneRegression(A2J_Regression):
    def __init__(self, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(2, feature_size, num_anchors, num_classes)

class A2J_DepthRegression(A2J_Regression):
    def __init__(self, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(1, feature_size, num_anchors, num_classes)

class A2J_AnchorProposal(A2J_Regression):
    def __init__(self, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(1, feature_size, num_anchors, num_classes)


from resnet.bottleneck_block import BottleneckBlock
from resnet.resnet import ResNet

class A2J(Model):
    def __init__(self, input_shape, num_classes, predict_3D = True):
        super().__init__(input_shape)
        self.num_classes = num_classes
        self.predict_3D = predict_3D

    def build(self):
        # backbone = ResNetBackbone(self.input_shape)
        # backbone.build()
        inputs = layers.Input(shape = self.input_shape)
        backbone = ResNet(BottleneckBlock, [3,4,6,3])
        x3, x4 = backbone(inputs)

        if self.predict_3D:
            anchor_proposal_out = A2J_AnchorProposal(num_classes=self.num_classes)(x3)
            in_plane_regression_out = A2J_InPlaneRegression(num_classes=self.num_classes)(x4)
            depth_regression_out = A2J_DepthRegression(num_classes=self.num_classes)(x4)

            self.model = models.Model(inputs = inputs, outputs = [anchor_proposal_out, in_plane_regression_out, depth_regression_out])

        else:
            anchor_proposal = A2J_AnchorProposal(x3, num_classes=self.num_classes)
            anchor_proposal.build()

            in_plane_regression = A2J_InPlaneRegression(x4, num_classes=self.num_classes)
            in_plane_regression.build()

            self.model = models.Model(inputs = inputs, outputs = [anchor_proposal.outputs, in_plane_regression.outputs])

        super().build()

    def compile(self):
        pass


a2j = A2J((256, 256, 1), 15)
a2j.build()
