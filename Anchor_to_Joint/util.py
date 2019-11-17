from tensorflow.keras import backend as K
from tensorflow.keras import layers

def bn():
    return layers.BatchNormalization(axis=3)
def relu():
    return layers.ReLU()

import numpy as np
import math
"""TODO generate anchor cluster may be able to do replicate's job as well"""
def generate_anchor_cluster(image_size, feature_map_size, anchor_stride = [4,4]):
    # every pixel in the feature map corresponds to a region of size 'patch' in the input image
    patch = [image_dim/feature_map_dim for image_dim, feature_map_dim in zip(image_size, feature_map_size)]
    coords = []
    for i in range(len(anchor_stride)):
        separation = math.ceil(patch[i]/anchor_stride[i])
        coords.append(np.arange(separation/2.0, patch[i], separation).astype('float'))

    coords = np.meshgrid(*coords, indexing='ij')
    coords = np.vstack([arr.flatten() for arr in coords])
    # subtract 1 so that we have 0 based indexing
    return coords.T - 1

def replicate(anchor_cluster, image_size, feature_map_size):
    # every pixel in the feature map corresponds to a region of size 'patch' in the input image
    patch = [image_dim/feature_map_dim for image_dim, feature_map_dim in zip(image_size, feature_map_size)]

    shifts = [np.arange(0, dim)*dim for dim in patch]
    shifts = np.meshgrid(*shifts)
    shifts = np.vstack([arr.flatten() for arr in shifts]).T

    A = anchor_cluster.shape[0]
    K = shifts.shape[0]

    all_anchors = anchor_cluster.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))
    return all_anchors.reshape((A*K, 2))

def post_process(offsets, depths, responses):
    anchor_cluster = generate_anchor_cluster([256,256], [16,16])
    anchor_coords = replicate(anchor_cluster, [256,256], [16,16])
    anchor_coords = K.constant(np.expand_dims(anchor_coords, axis=1))

    responses = layers.Activation('softmax')(responses)
    anchor_heatmap = layers.Lambda(lambda x: x*anchor_coords)(responses)

    joint_coords = layers.Lambda(lambda x: x+anchor_coords)(offsets)
    joint_coords = layers.Multiply()([joint_coords, responses])

    depths = layers.Multiply()([depths, responses])

    return joint_coords, depths, anchor_heatmap
