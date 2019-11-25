from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, callbacks, optimizers

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
#from tensorflow.keras.backend.tensorflow_backend import set_session
K.set_session(sess)


from capsulenet import CapsNet

from nyu_preprocessing import NYU

if __name__ == '__main__':
    NYU_DIR = '/mnt/sshfs_cais/hdd/datasets/hands/nyu_hand_dataset/train_npy'

    generator = NYU(NYU_DIR, desired_size=256, batch_size=8)

    #from tensorflow.keras.optimizers import Adam
    optimizer = optimizers.Adam(lr = 0.001)

    def my_metric(y_true, y_pred):
        return 256*K.mean(K.abs(y_pred - y_true))

    checkpoint = callbacks.ModelCheckpoint('model-{epoch:02d}.h5', verbose=1)

    model = CapsNet([256,256,1], n_class=36, routings=4)
    model.summary()
    model.compile(optimizer, loss = ['mse'], loss_weights = [1.0], metrics = [my_metric])

    model.fit_generator(generator,
                        epochs = 20,
                        verbose = 1,
                        callbacks = [checkpoint],
                        steps_per_epoch = generator.__len__(),
                        shuffle = False)
