from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

import imagenet_utils
from imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape
from imagenet_utils import get_submodules_from_kwargs
preprocess_input = imagenet_utils.preprocess_input

from tensorflow.python.keras import backend, layers, models, utils
keras_utils = utils

def conv1x1(filters, stride=(1,1)):
    return layers.Conv2D(filters=filters, kernel_size=(1,1), strides=stride, kernel_initializer='he_normal')

def conv3x3(filters, stride=(1,1), dilation=1):
    return layers.Conv2D(filters=filters, kernel_size=(3,3), strides=stride, dilation_rate=dilation, padding='same', kernel_initializer='he_normal')

"""Conv Block"""
def basic_block(input_tensor, filters, stride=(1,1), dilation=1):
    x = conv3x3(filters, stride)(input_tensor)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)

    x = conv3x3(filters, dilation=dilation)
    x = layers.BatchNormalization(axis=3)(x)

    shortcut = conv1x1(4*filters, stride)(input_tensor)
    shortcut = layers.BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x, 1

"""Identity Block"""
def bottleneck_block(input_tensor, filters, stride=(1,1), dilation=1):
    x = conv1x1(filters)(input_tensor)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)

    x = conv3x3(filters, stride, dilation=dilation)(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)

    x = conv1x1(4*filters)(x)
    x = layers.BatchNormalization(axis=3)(x)

    shortcut = conv1x1(4*filters, stride)(input_tensor)
    shortcut = layers.BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x, 4

def resnet_layer(input_tensor, block_type, block_count, filters, stride=(1,1), dilation=1):
    # TODO pytorch version has some downsampling stuff here
    x, expansion = block_type(input_tensor, filters, stride)
    for _ in range(1, block_count):
        x, _ = block_type(x, expansion*filters, dilation=dilation)
    return x

def resnet(input_tensor, block_type, layer_counts):
    x = layers.ZeroPadding2D(padding=(3,3))(input_tensor)
    x = layers.Conv2D(64, (7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    x = layers.ZeroPadding2D(padding=(1,1))(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)

    x = resnet_layer(x, block_type, layer_counts[0], 64)
    x = resnet_layer(x, block_type, layer_counts[1], 128, stride=(2,2))
    x3 = resnet_layer(x, block_type, layer_counts[2], 256, stride=(2,2))
    x4 = resnet_layer(x3, block_type, layer_counts[3], 512, dilation=2)

    return x3, x4



def MyResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):

    global backend, layers, models, keras_utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # start A
    x = layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1,1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2))(x)
    # end A
    x = basic_block(x, 64)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    x = basic_block(x, 128)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    x = basic_block(x, 256)#<--change
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)

    x = basic_block(x, 512)
    x = bottleneck_block(x, 512)
    x = bottleneck_block(x, 512)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
