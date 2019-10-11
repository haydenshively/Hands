import os
import numpy as np

def load_npy_data(amount = None):
    raw = np.load('Datasets/Raw.npy')
    highlighted = np.load('Datasets/Highlighted.npy')

    if amount is not None: return raw[:amount], highlighted[:amount]
    else: return raw, highlighted


def num_files_in(dir):
    return len(next(os.walk(dir))[2])

def load_MNIST_style(from_dir):
    num_files = num_files_in(from_dir)
    X = np.zeros((num_files, 90, 90))

    directory = os.fsencode(from_dir)
    for i, file in enumerate(os.listdir(directory)):
         filename = os.fsdecode(file)
         if filename.endswith(".npy"):
             X[i] = np.load(os.path.join(from_dir, filename))

    print(X.shape)
    np.save('MNIST_style.npy', X)
    return X
