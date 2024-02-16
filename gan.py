#rectGAN: https://github.com/xjdeng/RectGAN/blob/master/aspect_ratio.py
#artGAN: https://github.com/cs-chan/ArtGAN/blob/master/ArtGAN/Genre128GANAE.py

# To fix:
# 1) Clean up code 
# 2) Make code work on smaller dataset?
# 3) Currently in progress: implementing discriminator/generator, understanding the different types of neural net layers,
#    adjusting for best performance. I do not understand how layers work but most GANs seem to follow
#    Conv2D -> LeakyReLU -> Batchnorm for a few rounds of up/downscaling?

#imports
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np
from matplotlib import pyplot as plt
import os
import PIL
import time
import glob
import imageio
from IPython import display
from sklearn.model_selection import train_test_split

#image source, dimensions

image_dimensions = (192, 256, 1)
batch_size = 4
batch_length = 100
data_directory = './images_V2I/output_training_set' #output file name

strides = [(4,4), (3,4), (2,2)] #from rectGAN aspect_ratio
noise = tf.random.normal([1, 100])

alpha_Discriminator = 0.2
alpha_Generator = 0.2
momentum_BatchNormalization = 0.8 #rectgan suggests 0.3


def add_generator_layer(model, filters, kernel_size, num_strides): #from rectgan
    model.add(layers.Conv2DTranspose(filters, kernel_initializer=kernel_size, strides=num_strides, padding='same', use_bias=False))
    model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
    model.add(layers.LeakyReLU())

# NEW TUTORIAL: https://www.tensorflow.org/tutorials/generative/dcgan
# https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
def make_generator():
    model = tf.keras.Sequential() #add layers to generator model

    model.add(layers.Dense(7*7*256, input_dim=128, use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape(7, 7, 128))

    add_generator_layer(model, 128, (5, 5), strides[0])
    add_generator_layer(model, 64, (5, 5), strides[1])
    add_generator_layer(model, 1, (5, 5), strides[2])

    return model

#https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
def add_discriminator_layer(model, filters, kernel_size, num_strides): #from rectgan/nets.py and https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
    # model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
    model.add(layers.Conv2D(filters, kernel_initializer=kernel_size, strides=num_strides, padding='same'))
    model.add(layers.LeakyReLU(alpha_Discriminator))

#other tutorial: https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
def make_discriminator(): #same tutorial, https://github.com/nicknochnack/GANBasics/blob/main/FashionGAN-Tutorial.ipynb
    model = tf.keras.Sequential() #FIGURE OUT STRIDES for a rectangular image!!!

    add_discriminator_layer(model, 128, (3, 3), strides[0])
    add_discriminator_layer(model, 256, (3, 3), strides[1])
    add_discriminator_layer(model, 512, (3, 3), strides[2])
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid')) #may be unnecessary

    return model

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) #real_output = 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) #fake_output = 0
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)


generator = make_generator()
gen_image = generator(noise, training=False)
plt.imshow(gen_image[0, :, :, 0], cmap='gray')


# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
#                                  discriminator_optimizer=discriminator_optimizer, 
#                                  generator=generator, discriminator=discriminator)
                                        

# # https://pylessons.com/gan-introduction
# def train_step(epochs):
#     for epoch in range(epochs):
#         for _ in range(X_train.shape[0]):
#             # Generate random noise as an input to the generator
#             noise = np.random.normal(0, 1, (batch_size, latent_dim))
#             # Generate fake images using the generator
#             generated_images = generator.predict(noise)
#             # Select a random batch of real images
#             idx = np.random.randint(0, X_train.shape[0], batch_size)
#             real_images = X_train[idx]
#             # Labels for real and fake images
#             real_labels = np.ones((batch_size, 1))
#             fake_labels = np.zeros((batch_size, 1))
#             # Train the discriminator on real and fake images
#             d_loss_real = discriminator.train_on_batch(real_images, real_labels)
#             d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
#             # Train the generator
#             noise = np.random.normal(0, 1, (batch_size, latent_dim))
#             valid_labels = np.ones((batch_size, 1))
#             g_loss = gan.train_on_batch(noise, valid_labels)

#             print(f"Epoch {epoch+1}, [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

