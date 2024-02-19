#rectGAN: https://github.com/xjdeng/RectGAN/blob/master/aspect_ratio.py
#artGAN: https://github.com/cs-chan/ArtGAN/blob/master/ArtGAN/Genre128GANAE.py
#super helpful tensorflow tutorial: https://www.tensorflow.org/tutorials/generative/dcgan

# To fix:
# 1) Clean up code 
# 2) Make code work on smaller dataset?
    # Implement K-Fold cross validation if possible for a GAN
# 3) Currently in progress: implementing discriminator/generator, understanding the different types of neural net layers,
#    adjusting for best performance. I do not understand how layers work but most GANs seem to follow
#    Conv2D -> LeakyReLU -> Batchnorm for a few rounds of up/downscaling?
# 4) Use SGD!
# TFRecord: https://pyimagesearch.com/2022/08/08/introduction-to-tfrecords/

#imports
import tensorflow as tf
import sys
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import losses
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from IPython import display
# from sklearn.model_selection import train_test_split


'''
Arguments for GAN
'''
EPOCHS = 3
EXAMPLES_TO_GENERATE = 6
IMAGE_DIMENSIONS = (192, 256, 1)
BATCH_SIZE = 4
BUFFER_SIZE = 80 #should be around the size of the dataset! Shouldn't be hardcoded
TF_RECORD_PATH = "output.tfrecord"

strides = [(4,4), (3,4), (2,2)] #from rectGAN aspect_ratio
noise_dim = 100
seed = tf.random.normal([EXAMPLES_TO_GENERATE, noise_dim])

alpha_Discriminator = 0.2
alpha_Generator = 0.2
momentum_BatchNormalization = 0.8 #rectgan suggests 0.3






################# Building Generator #################

def add_generator_layer(model, filters, kernel, num_strides): #from rectgan
  '''
  Adding Conv2DTranspose -> BatchNormalization -> LeakyReLU layers to the model
  '''
  model.add(layers.Conv2DTranspose(filters, kernel_size=kernel, strides=num_strides, padding='same', use_bias=False))
  model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
  model.add(layers.LeakyReLU())

# NEW TUTORIAL: https://www.tensorflow.org/tutorials/generative/dcgan
# https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
def make_generator():
  model = tf.keras.Sequential() #add layers to generator model

  model.add(layers.Dense(7*7*256, input_dim=128, use_bias=False))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Reshape((7, 7, 256)))

  add_generator_layer(model, 128, (5, 5), strides[0])
  add_generator_layer(model, 64, (5, 5), strides[1])
  add_generator_layer(model, 1, (5, 5), strides[2])

  return model

#https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
def add_discriminator_layer(model, filters, kernel, num_strides): #from rectgan/nets.py and https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
  # model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
  model.add(layers.Conv2D(filters, kernel_size=kernel, strides=num_strides, padding='same'))
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







######## Running the model ########

#https://www.tensorflow.org/tutorials/generative/dcgan
@tf.function #turns into a graph, for faster execution
def train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: #compute gradients for discriminator and generator using two different GradientTapes
      generated_images = generator(noise, training=True)

      real_output = discriminator(image_batch, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
#https://www.tensorflow.org/tutorials/generative/dcgan
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
  
  
  
  
# def _parse_function(): 
#   '''
#   Decode TFRecord binary to build dataset - images are stored as a TFRecord binary for space efficiency
#   '''
#   feature_description = {
#       'image': tf.io.FixedLenFeature([], tf.string),
#   }
#   example = tf.io.parse_single_example(example_proto, feature_description)
#   image = tf.io.decode_jpeg(example['image'])
#   image = tf.cast(image, tf.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
#   return image
  
  
  
#https://www.tensorflow.org/tutorials/generative/dcgan
#https://github.com/asahi417/CycleGAN/blob/master/cycle_gan/cycle_gan.py
def train(): #Run to train set
  print("INITIALIZING SETUP...")
  print("Creating dataset...")
  dataset = tf.data.TFRecordDataset(TF_RECORD_PATH, compression_type='GZIP') #hopefully builds dataset from output.tfrecord
  # dataset = dataset.map(_parse_function)
  # num_samples = tf.data.experimental.cardinality(dataset).numpy()
  # print("Total number of samples in the dataset:", num_samples) 
  # dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  
  print("Tensorflow dataset:", dataset)
  
  print("Creating generator model...")
  generator = make_generator()
  
  print("Creating discriminator model...")
  discriminator = make_discriminator()

  #Optimize losses and performance:
  print("Creating optimizers...")
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

  print("Creating directory for model checkpoints...")
  checkpoint_dir = 'GAN_checkpoints'
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  checkpoint_prefix = os.path.join(checkpoint_dir, "checkpt")
  
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                  discriminator_optimizer=discriminator_optimizer, 
                                  generator=generator, discriminator=discriminator)
  
  print("SETUP COMPLETE.\n\n")
  
  
  
  
  
  
  
  
  
  
  
  
  print("Beginning training loop.", end="\r")
  for epoch in range(EPOCHS):
    start = time.time()

    for image_batch in dataset:
      print("iterating.")
      print("current batch:", image_batch)
  #     train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

  #   # Produce images for the GIF as you go
  #   display.clear_output(wait=True)
  #   generate_and_save_images(generator,
  #                            epoch + 1,
  #                            seed)

  #   # Save the model every 15 epochs
  #   if (epoch + 1) % 2 == 0:
  #     checkpoint.save(file_prefix = checkpoint_prefix)
  #     print("Checkpoint saved successfully to file:", checkpoint_dir)

  #   print ('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

  # # Generate after the final epoch
  # display.clear_output(wait=True)
  # generate_and_save_images(generator,
  #                          EPOCHS,
  #                          seed)
  
def main():  
  train()

if __name__ == "__main__":
  main()

  
  