from util import *

x_train = build_dataset_from('/Users/haydenshively/Desktop/Hand Datasets/RHD_published_v2/training/MNIST-style', 'MNIST_style.npy')
x_test = x_train[-1000:].copy()
x_train = x_train[:-1000]


image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


input_shape = (image_size, image_size, 1)
kernel_size = 3
filters = 16
latent_dim = 30


if __name__ == '__main__':
    from vae import VAE

    vae = VAE(input_shape, kernel_size, filters, latent_dim)
    vae.build()
    vae.compile()

    # vae.vae.load_weights('vae_cnn_hands.h5')

    vae.model.fit(x_train, epochs = 50, batch_size = 64, validation_data = (x_test, None))

    vae.model.save_weights('vae_cnn_hands_v2.h5')
