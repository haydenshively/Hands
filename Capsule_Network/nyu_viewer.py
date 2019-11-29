import os
import cv2
import random
import numpy as np

def num_files_in(dir):
    return len(next(os.walk(dir))[2])

dir = '/Users/coppercut/Downloads/datasets/nyu_hand_dataset/train_npy/'

if __name__ == '__main__':
    num_files = num_files_in(os.path.join(dir, 'X'))

    while True:
        file_name = '2%07d.npy' % int(random.random()*num_files)
        try:
            x = np.load(os.path.join(dir, 'X', file_name))
            y = np.load(os.path.join(dir, 'Y', file_name))

            for i in range(y.shape[0]):
                joint = y[i].astype('uint16')
                cv2.circle(x, (joint[0], joint[1]), 3, color=(1,))

            cv2.imshow(file_name, x)
            if cv2.waitKey(0) == 27: break

        except:
            print('File {} does not exist.'.format(file_name))

    cv2.destroyAllWindows()
