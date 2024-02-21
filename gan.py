#rectGAN: https://github.com/xjdeng/RectGAN/blob/master/aspect_ratio.py
#artGAN: https://github.com/cs-chan/ArtGAN/blob/master/ArtGAN/Genre128GANAE.py
#super helpful tensorflow tutorial: https://www.tensorflow.org/tutorials/generative/dcgan
#Another very helpful tensorflow tutorial: https://www.tensorflow.org/tutorials/generative/cyclegan
#incredibly straightforward explanation of what a GAN is: https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d
# Tips and tricks: https://medium.com/intel-student-ambassadors/tips-on-training-your-gans-faster-and-achieve-better-results-9200354acaa5#:~:text=Batch%20Size%3A&text=While%20training%20your%20GAN%20use,a%20negative%20effect%20on%20training.

# To fix:
# 1) Clean up code 
# 2) Make code work on smaller dataset?
    # Implement K-Fold cross validation if possible for a GAN
# 3) Currently in progress: implementing discriminator/generator, understanding the different types of neural net layers,
#    adjusting for best performance. I do not understand how layers work but most GANs seem to follow
#    Conv2D -> LeakyReLU -> Batchnorm for a few rounds of up/downscaling?
# 4) Use SGD!
# 5) Cropping is such a non-issue for right now, should stick to nxn images until this works.
# 6) Batch size = 1 for CycleGAN but 256 for DCGAN. 
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
# IMAGE_HEIGHT = 192
# IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256 #for now
IMAGE_WIDTH = 256 #for now
BATCH_SIZE = 8 #From tips and tricks article
BUFFER_SIZE = 80 #should be around the size of the dataset! Shouldn't be hardcoded
TF_RECORD_PATH = "output.tfrecord"

# strides = [(4,4), (3,4), (2,2)] #from rectGAN aspect_ratio
strides = [(1, 1), (2, 2), (2, 2)] #for now, for a square image
noise_dim = 100
seed = tf.random.normal([EXAMPLES_TO_GENERATE, noise_dim])

alpha_Discriminator = 0.2
alpha_Generator = 0.2
momentum_BatchNormalization = 0.8 #rectgan suggests 0.3






################# Building Generator #################

def add_generator_layer(model, num_filters, kernel, num_strides): #from rectgan
  '''
  Adding Conv2DTranspose -> BatchNormalization -> LeakyReLU layers to the model
  '''
  model.add(layers.Conv2DTranspose(num_filters, kernel_size=kernel, strides=num_strides, padding='same', use_bias=False))
  model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
  model.add(layers.LeakyReLU())

# NEW TUTORIAL: https://www.tensorflow.org/tutorials/generative/dcgan
# https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
def make_generator():
  '''
  Create noise for the generator. Experiment with the number of layers!
  '''
  model = tf.keras.Sequential() #add layers to generator model
  # model.add(layers.Dense(7*7*256, input_dim=128, use_bias=False))
  model.add(layers.Dense(16*16*256, input_shape=(100,), use_bias=False))
  model.add(layers.LeakyReLU(alpha=0.2))
  model.add(layers.Reshape((16, 16, 256)))
  # assert model.output_shape == (None, 16, 16, 256)  

  add_generator_layer(model, 256, (5, 5), strides[0])
  add_generator_layer(model, 128, (5, 5), strides[1])
  add_generator_layer(model, 64, (5, 5), strides[1])
  add_generator_layer(model, 32, (5, 5), strides[1])
  add_generator_layer(model, 3, (5, 5), strides[2]) #3-channel output?
  # model.add(layers.Conv2DTranspose(, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid')) #change to 3-channel output?

  return model

#https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/
def add_discriminator_layer(model, filters, kernel, num_strides): #from rectgan/nets.py and https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
  # model.add(layers.BatchNormalization(momentum=momentum_BatchNormalization))
  model.add(layers.Conv2D(filters, kernel_size=kernel, strides=num_strides, padding='same', input_shape=(256, 256, 3)))
  model.add(layers.LeakyReLU(alpha_Discriminator))

#other tutorial: https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
def make_discriminator(): #same tutorial, https://github.com/nicknochnack/GANBasics/blob/main/FashionGAN-Tutorial.ipynb
  model = tf.keras.Sequential()
  add_discriminator_layer(model, 64, (3, 3), strides[2])
  add_discriminator_layer(model, 128, (3, 3), strides[1])
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(3, activation='sigmoid')) #3 channel output

  return model

def discriminator_loss(real_output, fake_output):
  real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output) #real_output = 1
  fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output) #fake_output = 0
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output) #array of 1s

######## Running the model ########

#https://www.tensorflow.org/tutorials/generative/dcgan
@tf.function #turns into a graph, for faster execution
def train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    
    print("Image batch shape:", image_batch.get_shape())
    
    
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
  for i in range(predictions.shape[0]):
    plt.imshow(predictions.shape[i, :, :, 0] * 127.5 + 127.5)
    plt.axis('off')
    plt.savefig(f"GAN_output_images/img{i}/image{i}_epoch_{epoch:04d}.png")
    plt.clf()  # Clear the current figure to prevent overlapping plots
    
  ##Works, but not what I want:  
  # predictions = model(test_input, training=False)
  # fig = plt.figure(figsize=(4, 4))

  # for i in range(predictions.shape[0]):
  #     plt.subplot(4, 4, i+1)
  #     plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
  #     plt.axis('off')

  # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()
  
  
def make_output_directories(examples_to_generate):
  '''
  Make one image output directory for each example to generate.
  This saves all work-in-progress images to each directory.
  '''
  if not os.path.exists('GAN_output_images'):
    os.makedirs('GAN_output_images')
  for i in range(examples_to_generate):
    new_dir = 'GAN_output_images/img' + str(i)
    if not os.path.exists(new_dir):
      os.makedirs(new_dir)

#https://www.kaggle.com/code/drzhuzhe/monet-cyclegan-tutorial  
#decodes into tensor
def decode_image(image):
  '''
  Map the image to the [-1, 1] range
  '''
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  image = (tf.cast(image, tf.float32) / 127.5) - 1 #Normalize images to [-1, 1]
  return image
  # return tf.data.Dataset.from_tensor_slices(image)

#decode
#https://www.kaggle.com/code/drzhuzhe/monet-cyclegan-tutorial
def read_tfrecord(example):
  '''
  Decode the image from the tfrecord file using decode_image
  '''
  tfrecord_format = {
    "image": tf.io.FixedLenFeature([], tf.string)
  }
  example = tf.io.parse_single_example(example, tfrecord_format)
  image = decode_image(example['image'])
  return image
  
#https://www.tensorflow.org/tutorials/generative/dcgan
#https://github.com/asahi417/CycleGAN/blob/master/cycle_gan/cycle_gan.py
def train(): 
  #Set up dataset
  print("BEGINNING SETUP...")
  print("Creating dataset from TFRecord...")
  dataset = tf.data.TFRecordDataset(TF_RECORD_PATH) #Do not use compression, https://github.com/shahrukhqasim/TIES-2.0/issues/14
  dataset = dataset.map(read_tfrecord) #Unpacks from string to a float32 with correct dims under shape
  
  print("Shuffling dataset, creating batches...")
  train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  print("Dataset:", dataset)
  print("Train dataset:", train_dataset)
  
  #Create generator and discriminator models:
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
  
  print("Creating directories for output images...")
  make_output_directories(EXAMPLES_TO_GENERATE)
  print("SETUP COMPLETE.\n\n")
  

  print("Beginning training loop...")
  for epoch in range(EPOCHS):
    start = time.time()
    print("Epoch:", epoch)
    for image_batch in train_dataset:
      print("Iterating for image batch...")
      train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

    print("Saving images...")
    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    print("Saving checkpoint of model...")
    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      print("Checkpoint saved successfully to file:", checkpoint_dir)

    print ('Time for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           EPOCHS,
                           seed)
  
def main():  
  train()

if __name__ == "__main__":
  main()

  
  
  
  
  # #Test from DCGAN tutorial:
# #generator: works for nxn
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

# #discriminator:
# decision = discriminator(generated_image)
# print (decision)
