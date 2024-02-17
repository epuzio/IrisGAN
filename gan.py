#rectGAN: https://github.com/xjdeng/RectGAN/blob/master/aspect_ratio.py
#artGAN: https://github.com/cs-chan/ArtGAN/blob/master/ArtGAN/Genre128GANAE.py

# To fix:
# 1) Clean up code 
# 2) Make code work on smaller dataset?
    # Implement K-Fold cross validation if possible for a GAN
# 3) Currently in progress: implementing discriminator/generator, understanding the different types of neural net layers,
#    adjusting for best performance. I do not understand how layers work but most GANs seem to follow
#    Conv2D -> LeakyReLU -> Batchnorm for a few rounds of up/downscaling?
# 4) Use SGD!

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


batch_length = 100
data_directory = './images_V2I/output_training_set' #output file name
EPOCHS = 50
EXAMPLES_TO_GENERATE = 6
IMAGE_DIMENSIONS = (192, 256, 1)
BATCH_SIZE = 4

strides = [(4,4), (3,4), (2,2)] #from rectGAN aspect_ratio
noise_dim = 100

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
    real_loss = tf.keras.losses.BinaryCrossEntropy(from_logits=True)(tf.ones_like(real_output), real_output) #real_output = 1
    fake_loss = tf.keras.losses.BinaryCrossEntropy(from_logits=True)(tf.zeros_like(fake_output), fake_output) #fake_output = 0
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossEntropy(from_logits=True)(tf.ones_like(fake_output), fake_output) #array of 1s


generator = make_generator()
discriminator = make_discriminator()


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = 'gan_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                 discriminator_optimizer=discriminator_optimizer, 
                                 generator=generator, discriminator=discriminator)

#https://www.tensorflow.org/tutorials/generative/dcgan
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#https://www.tensorflow.org/tutorials/generative/dcgan
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  
  
  
  
  
#https://www.tensorflow.org/tutorials/generative/dcgan
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()