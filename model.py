import keras.layers as layers
import keras.models as models
import keras.optimizers as opt
import keras.losses as lss
import numpy as np
import matplotlib.pyplot as plt

z_dim = 100


def discriminator_block(units, kernel=3, padding='same'):
    return models.Sequential([
        layers.Conv2D(units, kernel, padding=padding),
        layers.MaxPool2D(),
        layers.BatchNormalization(),
        layers.LeakyReLU(.01)
    ])


def generator_block(units, kernel=3, padding='same'):
    return models.Sequential([
        layers.UpSampling2D(),
        layers.Conv2D(units, kernel, padding=padding),
        layers.BatchNormalization(),
        layers.LeakyReLU(.01)
    ])


def build_generator(z_dim=100):
    l0 = layers.Input((z_dim,))

    l2 = layers.Dense(7 * 7 * 128, name='Dense_1st_layer')(l0)
    l3 = layers.Reshape((7, 7, 128), name='reshape_layer')(l2)
    l4 = layers.LeakyReLU(.01)(l3)
    l5 = generator_block(64)(l4)  # 14
    l6 = generator_block(32)(l5)  # 28
    l7 = layers.Conv2D(1, 1, activation='tanh')(l6)
    return models.Model(l0, l7)


def build_discriminator(input_shape=(28, 28, 1)):
    l0 = layers.Input(input_shape)
    l1 = discriminator_block(32)(l0)  # 14
    l2 = discriminator_block(64)(l1)  # 7
    l3 = discriminator_block(128)(l2)
    l4 = layers.Flatten()(l3)
    l5 = layers.Dense(1, activation='sigmoid')(l4)
    return models.Model(l0, l5)


def compile_gan():
    gener = build_generator()
    discr = build_discriminator()
    gan = models.Sequential([gener, discr])
    # compiling the discriminator
    discr.compile(opt.adam(), lss.binary_crossentropy)
    # compiling the GAN
    gan.layers[1].trainable = False
    gan.compile(opt.adam(), lss.binary_crossentropy)
    return gan, gener, discr


def load_model(path='gan.h5'):
    return models.load_model(path)


if __name__ == '__main__':
    mo = build_generator()
    z_sample = np.random.normal(1, 0, (1, z_dim))
    gen_img = mo.predict(z_sample)
    plt.imshow(np.reshape(gen_img[0], (28, 28)), cmap='gray');
    plt.axis('off')
    dmo = build_discriminator();
    pr = dmo.predict(gen_img)
    print(pr)
