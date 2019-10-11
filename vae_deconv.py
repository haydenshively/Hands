'''Example of VAE on MNIST dataset using CNN
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from util import load_MNIST_style
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    # x_test, y_test = data
    # os.makedirs(model_name, exist_ok=True)
    #
    # filename = os.path.join(model_name, "vae_mean.png")
    # # display a 2D plot of the digit classes in the latent space
    # z_mean, _, _ = encoder.predict(x_test,
    #                                batch_size=batch_size)
    # plt.figure(figsize=(12, 10))
    # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    # plt.colorbar()
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
#x_train = load_MNIST_style('/Users/haydenshively/Desktop/Hand Datasets/RHD_published_v2/training/MNIST-style')
x_train = np.load('MNIST_style.npy')
x_test = x_train[-1000:].copy()
x_train = x_train[:-1000]
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# VAE model = encoder + decoder
class VAE:
    def __init__(self, input_shape, batch_size, kernel_size, filters, latent_dim, epochs):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.latent_dim = latent_dim
        self.epochs = epochs

    def build_encoder(self):
        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        self.x = self.inputs
        for i in range(2):
            self.filters *= 2
            self.x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(self.x)

        # shape info needed to build decoder model
        self.shape = K.int_shape(self.x)

    def build_latent_space(self):
        # generate latent vector Q(z|X)
        self.x = Flatten()(self.x)
        self.x = Dense(16, activation='relu')(self.x)
        self.z_mean = Dense(self.latent_dim, name='z_mean')(self.x)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(self.x)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])

    def instantiate_encoder(self):
        # instantiate encoder model
        self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        self.encoder.summary()

    def build_decoder(self):
        # build decoder model
        self.latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        self.x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation='relu')(self.latent_inputs)
        self.x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(self.x)

        for i in range(2):
            self.x = Conv2DTranspose(filters=self.filters,
                                    kernel_size=self.kernel_size,
                                    activation='relu',
                                    strides=2,
                                    padding='same')(self.x)
            self.filters //= 2

        self.outputs = Conv2DTranspose(filters=1,
                                  kernel_size=self.kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(self.x)

    def instantiate_decoder(self):
        # instantiate decoder model
        self.decoder = Model(self.latent_inputs, self.outputs, name='decoder')
        self.decoder.summary()

    def instantiate_VAE(self):
        # instantiate VAE model
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = Model(self.inputs, self.outputs, name='vae')


# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 30

vae = VAE(input_shape, batch_size, kernel_size, filters, latent_dim, epochs)
vae.build_encoder()
vae.build_latent_space()
vae.instantiate_encoder()
vae.build_decoder()
vae.instantiate_decoder()
vae.instantiate_VAE()


if __name__ == '__main__':
    models = (vae.encoder, vae.decoder)
    #data = (x_test, None)

    # VAE loss = mse_loss or xent_loss + kl_loss
    # if args.mse:
    #     reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    # else:
    reconstruction_loss = binary_crossentropy(K.flatten(vae.inputs), K.flatten(vae.outputs))
    reconstruction_loss *= image_size * image_size


    kl_loss = 1 + vae.z_log_var - K.square(vae.z_mean) - K.exp(vae.z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.vae.add_loss(vae_loss)
    vae.vae.compile(optimizer='rmsprop')
    vae.vae.summary()

    # vae.vae.load_weights('vae_cnn_mnist.h5')

    # train the autoencoder
    vae.vae.fit(x_train,
            epochs=vae.epochs,
            batch_size=vae.batch_size,
            validation_data=(x_test, None))
    vae.vae.save_weights('vae_cnn_hands.h5')

    plot_results(models, None, batch_size=batch_size, model_name="vae_cnn")
