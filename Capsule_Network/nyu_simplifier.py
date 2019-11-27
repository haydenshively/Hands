import os
import cv2
import math
import fnmatch
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K


half_res_x = 640/2.0
half_res_y = 480/2.0
coeff_x = 588.036865
coeff_y = 587.075073
# factor_xz = 1.08836710
# factor_yz = 0.817612648

def num_files_in(dir):
    return len(next(os.walk(dir))[2])

def num_pngs_in(dir):
    return len(fnmatch.filter(os.listdir(dirpath), '*.png'))

def characteristics(dir):
    files = os.listdir(dir)
    counts = {}

    for file in files:
        png_type, camera_id = file.split('_')[:2]

        counts.setdefault(png_type, {})
        counts[png_type][camera_id] = counts[png_type].get(camera_id, 0) + 1

    return counts

def pixel2world(x):
    x[:,:,0] = (x[:,:,0] - half_res_x)*x[:,:,2] / coeff_x
    x[:,:,1] = (x[:,:,1] - half_res_y)*x[:,:,2] / coeff_y
    return x

def world2pixel(x):
    x[:,:,0] = coeff_x * x[:,:,0]/x[:,:,2] + half_res_x
    x[:,:,1] = coeff_y * x[:,:,1]/x[:,:,2] + half_res_y
    return x


desired_size=256
dir = '/Users/coppercut/Downloads/datasets/nyu_hand_dataset/train/'
dir_out = '/Users/coppercut/Downloads/datasets/nyu_hand_dataset/train_npy_n/'

if __name__ == '__main__':
    files = os.listdir(dir)
    files.sort()

    joint_data = sio.loadmat(os.path.join(dir, 'joint_data.mat'))
    joint_coords = joint_data['joint_uvd']
    joint_coords = joint_coords.reshape(-1, *joint_coords.shape[2:])

    def imread_tf(path):
        x = tf.io.read_file(path)
        x = tf.io.decode_png(x, channels=3, dtype=tf.dtypes.uint16)
        return K.eval(x)#x.numpy()

    def imread_cv(path):
        x = cv2.imread(path, -1)
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    def resize_tf(img, size):
        img = tf.compat.v2.image.resize(img[:,:,np.newaxis], (size,)*2)
        return K.eval(img)#img.numpy()

    def resize_cv(img, size):
        return cv2.resize(img, (size,)*2)


    successes = 0
    for file in files:
        if file.startswith('depth') and file.endswith('.png'):

            Y = joint_coords[successes]
            # Y_center = Y[:,:2].mean(axis = 0)
            # Y_size = Y[:,:2].std(axis = 0).max(axis = 0)*2.5
            Y_size = Y[:,:2].ptp(axis = 0)/2.0
            Y_topleft = Y[:,:2].min(axis = 0)
            Y_center = Y_topleft + Y_size
            Y_size = Y_size.max(axis = 0) + 10.0

            # depth = imread_tf(os.path.join(dir, depthname))
            depth = imread_cv(os.path.join(dir, file))
            depth = np.left_shift(depth[:,:,1].astype('uint32'), 8) + depth[:,:,2].astype('uint32')
            depth = 1 - depth.astype('float32')/depth.max()

            center = tuple(Y_center.astype('uint32'))
            size = int(Y_size)
            cropped = depth[center[1]-size:center[1]+size, center[0]-size:center[0]+size]

            # resized = resize_tf(cropped, desired_size)
            resized = resize_cv(cropped, desired_size)

            recolored = (resized - resized.mean()) * min(1.2, 1.0/resized.ptp())
            recolored = recolored - recolored.min()

            Y[:,:2] -= Y_center - Y_size
            Y[:,:2] *= desired_size/Y_size/2

            Y[:,2] -= Y[:,2].min()
            Y[:,2] /= Y[:,2].max()

            out_name = file.split('.')[0].split('_')[-1] + '.npy'
            np.save(os.path.join(dir_out, 'X', out_name), recolored)
            np.save(os.path.join(dir_out, 'Y', out_name), Y)

            successes += 1

            # gt_resized = recolored.copy()
            # for j in range(Y.shape[0]):
            #     joint = Y[j].astype('uint16')
            #     cv2.circle(gt_resized, (joint[0], joint[1]), 3, color=(1,))
            # cv2.imshow('gt_res', gt_resized)

            # cv2.imshow('depth', depth)
            # cv2.imshow('cropped', cropped)
            # cv2.imshow('resized', resized)
            # cv2.imshow('recolored', recolored)
            # cv2.waitKey(1)
