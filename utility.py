import keras.datasets as datasets
import matplotlib.pyplot as plt
import keras.utils as utils
import numpy as np

def normalize_reshape(X):
    X = X / 127.5 - 1
    X = X[..., np.newaxis]
    return X

def prepare_data(img, lb):
    return [normalize_reshape(img),
            utils.to_categorical(lb, 10)]

def get_mnist():
    (X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
    return [prepare_data(X_train, Y_train),
            prepare_data(X_test, Y_test)]

def plot(generator, z_dim= 100, row=5, col=5):
    num = row*col
    z_sample = np.random.normal(0,1,(num,z_dim))
    imgs = generator.predict(z_sample)
    for i in range(num):
        img = np.reshape(imgs[i], (28,28))
        plt.subplot(row, col, i+1)
        plt.imshow(img,cmap = 'gray')
        plt.axis('off')


