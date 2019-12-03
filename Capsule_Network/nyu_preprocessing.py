import os
# import cv2
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


class NYU(Sequence):
    def __init__(self, dir, desired_size, batch_size=8):
        self.dir_X = os.path.join(dir, 'X')
        self.dir_Y = os.path.join(dir, 'Y')

        self.names = os.listdir(self.dir_X)
        self.names.sort()

        self.desired_size = desired_size

        self.sample_count = len(self.names)
        self.batch_size = batch_size

        # self.dim = [desired_size, desired_size, 1]
        # self.ndim = 3

    def __len__(self):
        return math.ceil(self.sample_count / self.batch_size)

    def load_x_or_y(self, dir, id):
        name = '%07d.npy' % id
        try:
            return np.load(os.path.join(dir, name))
        except:
            return self.load_x_or_y(dir, id - 1)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        end = min(end, self.sample_count)

        X = []
        Y = []

        for i in range(start, end):
            name = self.names[i]
            X.append(np.load(os.path.join(self.dir_X, name)))
            Y.append(np.load(os.path.join(self.dir_Y, name)))


        X = np.expand_dims(np.asarray(X), axis = -1)
        Y = np.asarray(Y)
#        print(Y[:,:2])
#        print(Y[:,:2].shape)
        Y[:,:,:2] /= self.desired_size

        return X, Y[:,:,:2]
