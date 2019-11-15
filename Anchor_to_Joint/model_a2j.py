from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from model import Model
class A2J_Regression(Model):
    def __init__(self, input_tensor, output_dims, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(input_tensor.shape[1:])
        self.input_tensor = input_tensor
        self.output_dims = output_dims
        self.feature_size = feature_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def build(self):
        # input should be channels last
        conv1 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='valid', kernel_initializer='glorot_normal')(self.input_tensor)
        bn1 = layers.BatchNormalization(axis=3)(conv1)# , kernel_initializer='ones', bias_initializer='zeros'
        act1 = layers.ReLU()(bn1)

        conv2 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='valid', kernel_initializer='glorot_normal')(act1)
        bn2 = layers.BatchNormalization(axis=3)(conv2)
        act2 = layers.ReLU()(bn2)

        conv3 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='valid', kernel_initializer='glorot_normal')(act2)
        bn3 = layers.BatchNormalization(axis=3)(conv3)
        act3 = layers.ReLU()(bn3)

        conv4 = layers.Conv2D(filters=self.feature_size, kernel_size=3, padding='valid', kernel_initializer='glorot_normal')(act3)
        bn4 = layers.BatchNormalization(axis=3)(conv4)
        act4 = layers.ReLU()(bn4)

        conv5 = layers.Conv2D(filters=self.num_anchors*self.num_classes*self.output_dims, kernel_size=3, padding='valid', kernel_initializer='glorot_normal')(act4)
        # TODO this reshape may not actually work
        self.outputs = layers.Reshape((-1, self.num_classes, self.output_dims))(conv5)

        # self.model = models.Model(inputs = self.input_tensor, outputs = self.outputs)
        super().build()

    def compile(self):
        # self.model.compile(optimizer = 'sgd', loss = TODO, metrics = ['accuracy'])
        super().compile()

class A2J_InPlaneRegression(A2J_Regression):
    def __init__(self, input_tensor, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(input_tensor, 2, feature_size=feature_size, num_anchors=num_anchors, num_classes=num_classes)

class A2J_DepthRegression(A2J_Regression):
    def __init__(self, input_tensor, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(input_tensor, 1, feature_size=feature_size, num_anchors=num_anchors, num_classes=num_classes)

class A2J_AnchorProposal(A2J_Regression):
    def __init__(self, input_tensor, feature_size=256, num_anchors=16, num_classes=15):
        super().__init__(input_tensor, 1, feature_size=feature_size, num_anchors=num_anchors, num_classes=num_classes)


from custom_resnet import resnet, bottleneck_block
class ResNetBackbone(Model):# LOTS OF TODOS HERE
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def build(self):
        # input should be channels last
        self.inputs = layers.Input(shape = self.input_shape)
        self.outputs = resnet(self.inputs, bottleneck_block, [3,4,6,3])

        self.model = models.Model(inputs = self.inputs, outputs = self.outputs)
        super().build()

    def compile(self):
        super().compile()

class A2J(Model):
    def __init__(self, input_shape, num_classes, predict_3D = True):
        super().__init__(input_shape)
        self.num_classes = num_classes
        self.predict_3D = predict_3D

    def build(self):
        if self.predict_3D:
            backbone = ResNetBackbone(self.input_shape)
            backbone.build()
            print(backbone.model.summary())
            x3, x4 = backbone.outputs

            anchor_proposal = A2J_AnchorProposal(x3, num_classes=self.num_classes)
            anchor_proposal.build()
            # print(anchor_proposal.model.summary())

            in_plane_regression = A2J_InPlaneRegression(x4, num_classes=self.num_classes)
            in_plane_regression.build()
            # print(in_plane_regression.model.summary())

            depth_regression = A2J_DepthRegression(x4, num_classes=self.num_classes)
            depth_regression.build()
            # print(depth_regression.model.summary())

            self.model = models.Model(inputs = backbone.inputs, outputs = [anchor_proposal.outputs, in_plane_regression.outputs, depth_regression.outputs])
        else:
            print('Not yet implemented')

        super().build()

    def compile(self):
        #TODO
        pass


a2j = A2J((360, 480, 1), 15)
a2j.build()
