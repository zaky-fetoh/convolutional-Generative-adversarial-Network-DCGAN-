import numpy as np
import keras.models as models
import utility as util
import model


def gan_train(modls, epochs= 5,
              training=util.get_mnist,
              batch_size=128, z_dim=100):

    (train_imgs,_),(_,_)= training()
    gan, gener, discr = modls
    epoch_step = int(train_imgs.shape[0]/batch_size)
    real_label = np.ones((batch_size,1),dtype=np.float)
    fake_label = np.zeros((batch_size,1), dtype= np.float)
    for itr in range(epoch_step*epochs):
        print(itr)
        #train discr
        indx = np.random.randint(0,train_imgs.shape[0],batch_size)
        real_imgs = train_imgs[indx]
        z_samples = np.random.normal(0,1,(batch_size,z_dim))
        fake_imgs = gener.predict(z_samples)
        discr.train_on_batch(real_imgs,real_label)
        discr.train_on_batch(fake_imgs,fake_label)
        #train generator
        z_samples = np.random.normal(0, 1, (batch_size, z_dim))
        gan.train_on_batch(z_samples,real_label)


if __name__ == '__main__':
    gan, gen, disc= model.compile_gan()
    gan_train([gan, gen, disc])
    util.plot(gen)
    gan.save('gan.h5')



