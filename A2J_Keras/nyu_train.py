from tensorflow.keras import layers, models, callbacks, optimizers

from a2j.a2j import a2j#A2J
from a2j.losses import *

from nyu_preprocessing import NYU

if __name__ == '__main__':
    NYU_DIR = '/home/haydenshively/ssd-datasets/train_npy'
    generator = NYU(NYU_DIR, desired_size=256, batch_size=16)
    optimizer = optimizers.Adam(lr = 0.035)

    checkpoint = callbacks.ModelCheckpoint('model-{epoch:02d}.h5', verbose=1)

    inputs = layers.Input(shape = (256, 256, 1))
    outputs = a2j(inputs, 36)
    model = models.Model(inputs=inputs, outputs=outputs)
    #model.summary()

    tau1 = 1.0/3.0
    tau2 = 1.0
    losses = [smoothL1(tau1), smoothL1(tau2), smoothL1(tau1)]
    loss_weights = [1.0, 1.0, 1.0/3.0]

    model.compile(optimizer, loss = losses, loss_weights = loss_weights, metrics = ['accuracy'])

    model.fit(generator,
              epochs = 34,
              verbose = 1,
              callbacks = [checkpoint],
              steps_per_epoch = generator.__len__(),
              shuffle = False,
              workers = 8,
              use_multiprocessing = True)
