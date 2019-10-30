"""
TRAIN
"""
from util import *
import os
import math
import numpy as np
from tensorflow.keras.utils import Sequence

class OurData(Sequence):
    def __init__(self, x_dir, y_dir, batch_size):
        self.sample_count = num_files_in(x_dir)
        if self.sample_count != num_files_in(y_dir):
            print("OOPS! Every input needs a single output.")

        self.x_dir = x_dir
        self.y_dir = y_dir
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.sample_count / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        first_x = np.load(os.path.join(self.x_dir, '%.5d.npy' % start))
        first_y = np.load(os.path.join(self.y_dir, '%.5d.npy' % start))

        X = np.zeros((self.batch_size, first_x.shape[0], first_x.shape[1], first_x.shape[2]))
        Y = np.zeros((self.batch_size, first_y.shape[0], first_y.shape[1]))

        for i in range(self.batch_size):
            imageX = np.load(os.path.join(self.x_dir, '%.5d.npy' % (start + i)))
            imageX = np.expand_dims(imageX, axis=0)
            X[i] = imageX
            # X.append(np.moveaxis(image, 2, 0))
            imageY = np.load(os.path.join(self.y_dir, '%.5d.npy' % (start + i)))
            imageY = np.expand_dims(imageY, axis=0)
            Y[i] = imageY

        return X, Y


if __name__ == '__main__':
    from unet import UNet

    INPUT_DIR = '/stor/ResearchData/Handtracking/Datasets/RHD_published_v2/training/four_channel'
    OUTPUT_DIR = '/stor/ResearchData/Handtracking/Datasets/RHD_published_v2/training/hands'
    BATCH_SIZE = 12

    generator = OurData(INPUT_DIR, OUTPUT_DIR, BATCH_SIZE)

    unet = UNet((320, 320, 4), 1)
    unet.build()
    unet.compile()

    unet.model.fit(generator,
                   epochs = 1,
                   verbose = 1,
                   steps_per_epoch = generator.__len__(),
                   workers = 4,
                   use_multiprocessing = True)

    unet.model.save('trained_unet.h5')
