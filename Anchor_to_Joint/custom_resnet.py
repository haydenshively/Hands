from tensorflow.keras import Model, layers, utils
import numpy as np

# in_filters = 64

# """Conv Block"""
# def basic_block(input_tensor, filters, stride=(1,1), dilation=1):
#     # conv 1 includes the stride
#     x = conv3x3(filters, stride)(input_tensor)
#     x = layers.BatchNormalization(axis=3)(x)
#     x = layers.ReLU()(x)
#     # conv 2 includes the dilation
#     x = conv3x3(filters, dilation=dilation)(x)
#     x = layers.BatchNormalization(axis=3)(x)
#
#     if (stride is not (1,1)) or (in_filters != 1*filters):
#         shortcut = conv1x1(1*filters, stride)(input_tensor)
#         shortcut = layers.BatchNormalization(axis=3)(shortcut)
#     else:
#         shortcut = input_tensor
#
#     x = layers.add([x, shortcut])
#     x = layers.ReLU()(x)
#     return x

# """Identity Block"""
# def bottleneck_block(input_tensor, filters, stride=(1,1), dilation=1):
#     x = conv1x1(filters)(input_tensor)
#     x = layers.BatchNormalization(axis=3)(x)
#     x = layers.ReLU()(x)
#     # conv 2 includes the stride and the dilation
#     x = conv3x3(filters, stride, dilation=dilation)(x)
#     x = layers.BatchNormalization(axis=3)(x)
#     x = layers.ReLU()(x)
#     # conv 3 has more 4 times as many filters
#     x = conv1x1(4*filters)(x)
#     x = layers.BatchNormalization(axis=3)(x)
#
#     if (stride is not (1,1)) or (in_filters != 4*filters):
#         shortcut = conv1x1(4*filters, stride)(input_tensor)
#         shortcut = layers.BatchNormalization(axis=3)(shortcut)
#     else:
#         shortcut = input_tensor
#
#     x = layers.add([x, shortcut])
#     x = layers.ReLU()(x)
#     return x

# def resnet_layer(input_tensor, block_type, block_count, filters, stride=(1,1), dilation=1):
#     x = block_type(input_tensor, filters, stride)
#
#     global in_filters
#     in_filters *= 4
#
#     for _ in range(1, block_count):
#         x = block_type(x, filters, dilation=dilation)
#     return x

# def resnet(input_tensor, block_type, layer_counts):
#     x = layers.ZeroPadding2D(padding=(3,3))(input_tensor)
#     x = layers.Conv2D(64, (7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal')(x)
#     x = layers.BatchNormalization(axis=3)(x)
#     x = layers.ReLU()(x)
#     x = layers.ZeroPadding2D(padding=(1,1))(x)
#     x = layers.MaxPooling2D((3,3), strides=(2,2))(x)
#
#     x = resnet_layer(x, block_type, layer_counts[0], 64)
#     x = resnet_layer(x, block_type, layer_counts[1], 128, stride=(2,2))
#     x3 = resnet_layer(x, block_type, layer_counts[2], 256, stride=(2,2))
#     x4 = resnet_layer(x3, block_type, layer_counts[3], 512, dilation=2)
#
#     return x3, x4
