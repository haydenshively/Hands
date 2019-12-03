from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, callbacks, optimizers

from capsulenet import CapsNet
from nyu_preprocessing import NYU

import cv2
import time

if __name__ == '__main__':
    NYU_DIR = '/Users/haydenshively/Downloads/test_npy'

    generator = NYU(NYU_DIR, desired_size=256, batch_size=8)

    model = CapsNet([256,256,1], n_class=36, routings=4)
    model.load_weights('model-20.h5')
    
    for batch in generator:
        X = batch[0]
        Y = batch[1]
        
        start = time.time()
        results = model.predict_on_batch(X)
        end = time.time()
        print('{} FPS'.format(X.shape[0]/(end - start)))
        
        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            result = results[i]
            
            for j in range(y.shape[0]):
                joint = (256*result[j]).astype('uint16')
                cv2.circle(x, (joint[0], joint[1]), 3, color=(1,))
#                joint = (256*y[j]).astype('uint16')
#                cv2.circle(x, (joint[0], joint[1]), 3, color=(0,))
            
            cv2.imshow('x', x)
            cv2.waitKey(1)



                                      
    print(results)
    print(results.shape)
