import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten,Reshape
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
import matplotlib.pyplot as plt

import numpy as np

#Generator model

def build_generator(z_dim): #28*28
    model= Sequential()
    model.add(Dense(128, input_dim=z_dim))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(28*28*1, activation='tanh')) #[-1,1]
    model.add(Reshape((28, 28,1)))
    return model

#Discriminator model

def build_discriminator(img_shape):
    model= Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha= 0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

#Gan model(combined Generator and Discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable= False
    model= Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

#hyperparameters
z_dim = 100
img_shape = (28, 28,1)

#build and compile the discriminator
discriminator =build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',optimizer= Adam(0.0002, 0.5),metrics= ['accuracy'])

#build the generator
generator= build_generator(z_dim)

#build and compile the GAN
gan= build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy',optimizer= Adam(0.0002, 0.5))

#training loop
epochs=1000
batch_size= 64
half_batch= batch_size// 2

X_train= np.random.randn(1000, 28, 28, 1)

for epoch in range(epochs):
    idx= np.random.randint(0, X_train.shape[0], half_batch)
    real_imgs= X_train[idx]

noise= np.random.randn(half_batch, z_dim)
fake_imgs= generator.predict(noise)

d_loss_real= discriminator.train_on_batch(real_imgs, np.ones((half_batch,1)))
d_loss_fake= discriminator.train_on_batch(fake_imgs, np.zeros((half_batch,1)))
d_loss = 0.5 *np.add(d_loss_real, d_loss_fake)

noise= np.random.randn(batch_size, z_dim)
g_loss= gan.train_on_batch(noise, np.ones((batch_size,1)))


if epoch % 100 ==0:
    print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}][G loss:{g_loss} ]")

    generated_image= generator.predict(np.random.randn(1, z_dim))
    generated_image_display= (generated_image+ 1) / 2 #[-1,1] => [0,1]

    clear_output(wait=True)
    plt.imshow(generated_image_display[0], cmap='gray')
    plt.axis('off')
    plt.show()