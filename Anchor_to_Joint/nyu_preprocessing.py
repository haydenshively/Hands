import os
import cv2
import math
import fnmatch
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import Sequence

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


class NYU(Sequence):
    def __init__(self, dir, desired_size=256, batch_size=8):
        file_counts = characteristics(dir)

        self.depth_counts = [0]
        for i in range(len(file_counts['depth'])):
            self.depth_counts.append(self.depth_counts[-1] + file_counts['depth'][str(i+1)])

        print(file_counts)
        print(file_counts['depth'])

        self.camera_id = 1
        self.desired_size=desired_size

        joint_data = sio.loadmat(os.path.join(dir, 'joint_data.mat'))
        joint_coords = joint_data['joint_uvd']
        self.joint_coords = joint_coords.reshape(-1, *joint_coords.shape[2:])

        self.dir = dir
        self.sample_count = self.joint_coords[0]
        self.batch_size = batch_size

    def image_name_at(self, index):
        if index >= self.depth_counts[self.camera_id]:
            self.camera_id += 1
        index -= self.depth_counts[self.camera_id - 1]
        return 'depth_' + '%d_%07d' % (self.camera_id, index)

    def __len__(self):
        return math.ceil(self.sample_count / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        Y = self.joint_coords[start:end]
        Y_centers = Y.mean(axis = 1)
        Y_sizes = Y.ptp(axis = 1).max(axis = 1)/2.0 + 5.0#5 pixel padding on each side

        Y[:,:,:2] -= (Y_centers - Y_sizes)
        Y[:,:,:2] *= self.desired_size/Y_sizes

        x = cv2.imread(os.path.join(self.dir, self.image_name_at(start)), -1)
        X = np.zeros((self.batch_size, x.shape[0], x.shape[1], 1))

        for i in range(start, end):
            x = cv2.imread(os.path.join(self.dir, self.image_name_at(i)), -1)
            x = x[:,:,2].astype('uint16') + np.left_shift(x[:,:,1].astype('uint16'), 8)

            center = tuple(Y_centers[i].astype('uint32'))
            size = int(Y_sizes[i])
            cropped = x[center[1]-size:center[1]+size, center[0]-size:center[0]+size]
            resized = cv2.resize(cropped, (self.desired_size,)*2)

            cv2.imshow('X', resized.astype('uint8'))
            cv2.waitKey(0)
            print(Y[i])

            X[i,:,:,0] = resized.astype('float')/255.0

        Y_split = Y[:,:,:2], Y[:,:,2], Y[:,:,:2]

        return X, Y_split
