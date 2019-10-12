from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

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


from model import Model
class VAE(Model):
    def __init__(self, input_shape, kernel_size, filters, latent_dim):
        super().__init__(input_shape)
        self.kernel_size = kernel_size
        self.filters = filters
        self.latent_dim = latent_dim

    def build(self):
        self.build_encoder()
        self.build_latent_space()
        self.build_decoder()

        self.encoder = models.Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name = 'encoder')
        self.encoder.summary()
        self.decoder = models.Model(self.decoder_inputs, self.decoder_outputs, name = 'decoder')
        self.decoder.summary()
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.model = models.Model(self.inputs, self.outputs, name = 'vae')

        super().build()

    def build_encoder(self):
        self.inputs = layers.Input(shape = self.input_shape, name = 'encoder_input')

        conv1 = layers.Conv2D(filters=self.filters*2, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(self.inputs)
        conv2 = layers.Conv2D(filters=self.filters*4, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(conv1)
        conv3 = layers.Conv2D(filters=self.filters*8, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(conv2)

        self.encoder_output = conv3
        self.encoder_output_shape = K.int_shape(self.encoder_output)

    def build_latent_space(self):
        # generate latent vector Q(z|X)
        conv3_flattened = layers.Flatten()(self.encoder_output)
        dens1 = layers.Dense(self.latent_dim*8, activation = 'relu')(conv3_flattened)
        self.z_mean = layers.Dense(self.latent_dim, name = 'z_mean')(dens1)
        self.z_log_var = layers.Dense(self.latent_dim, name = 'z_log_var')(dens1)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = layers.Lambda(sampling, output_shape = (self.latent_dim,), name = 'z')([self.z_mean, self.z_log_var])

    def build_decoder(self):
        self.decoder_inputs = layers.Input(shape = (self.latent_dim,), name = 'z_sampling')

        num_neurons = self.encoder_output_shape[1] * self.encoder_output_shape[2] * self.encoder_output_shape[3]

        dens2 = layers.Dense(num_neurons, activation='relu')(self.decoder_inputs)
        dens2_unflattened = layers.Reshape((self.encoder_output_shape[1], self.encoder_output_shape[2], self.encoder_output_shape[3]))(dens2)

        conv4 = Conv2DTranspose(filters=self.filters*8, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(dens2_unflattened)
        conv5 = Conv2DTranspose(filters=self.filters*4, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(conv4)
        conv6 = Conv2DTranspose(filters=self.filters*2, kernel_size=self.kernel_size, activation='relu', strides=2, padding='same')(conv5)

        self.decoder_outputs = Conv2DTranspose(filters=1, kernel_size=self.kernel_size, activation='sigmoid', padding='same', name='decoder_output')(conv6)


    def compile(self):
        reconstruction_loss = binary_crossentropy(K.flatten(self.inputs), K.flatten(self.outputs))
        reconstruction_loss *= image_size * image_size

        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        self.model.add_loss(vae_loss)
        self.model.compile(optimizer = 'rmsprop')

        super().compile()
