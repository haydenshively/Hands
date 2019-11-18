from tensorflow.keras import layers, models

from a2j.a2j import a2j#A2J
from a2j.losses import *

from nyu_preprocessing import NYU

if __name__ == '__main__':
    NYU_DIR = '/mnt/sshfs/hdd/datasets/hands/nyu_hand_dataset/train'

    generator = NYU(NYU_DIR, desired_size=256, batch_size=8)

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(lr = 0.00035)

    inputs = layers.Input(shape = (256, 256, 1))
    outputs = a2j(inputs, 15)
    model = models.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    tau1 = 1.0
    tau2 = 3.0
    losses = [smoothL1(tau1), smoothL1(tau2), smoothL1(tau1)]
    loss_weights = [3.0, 3.0, 1.0]

    model.compile(optimizer, loss = losses, loss_weights = loss_weights, metrics = ['accuracy'])

    model.fit(generator,
              epochs = 34,
              verbose = 1,
              steps_per_epoch = generator.__len__(),
              validation_split = 0.01,
              shuffle = True,
              workers = 4,
              use_multiprocessing = True)
