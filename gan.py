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
import numpy as np
from matplotlib import pyplot as plt

#image source, dimensions
data_directory = './images_V2I/output_training_set' #output file name
width = 192
height = 256
strides = [(4,4), (3,4), (2,2)] #from rectGAN aspect_ratio

alpha_Discriminator = 0.2
alpha_Generator = 0.2
momentum_BatchNormalization = 0.8 #rectgan suggests 0.3


batch_size = 8
batch_length = 100
noise_shape = (1, 1, 8)
samples = tf.placeholder(tf.float32, [None, 100])
noise = tf.placeholder(tf.float32, [None, 100])

#Parameters (from Genre128GANAE)
init_iter, max_iter = 0, 50000 
display_iter = 100
eval_iter = 100
store_img_iter = 100
save_iter = 1000

lr_init = 0.0002
batch_size = 100
zdim = 100
n_classes = 10
dropout = 0.2
im_size = [64, 64]
dname, gname = 'd_', 'g_'
tf.set_random_seed(1234)

data_set = '.datase'


# #Load from images
# (X_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# train_images = train_images.astype('float32') / 255.0


def add_generator_layer(model): #from rectgan
    model.add(Conv2DTranspose( filters, shape, padding='same',
        strides=mystrides, kernel_initializer=Args.kernel_initializer))

    model.add(BatchNormalization(momentum=momentum_BatchNormalization))
    model.add(LeakyReLU(alpha=alpha_Generator))
    return x


# Generator model (https://www.youtube.com/watch?v=AALBGpLbj6Q)
def generator():
    model = tf.keras.Sequential() #add layers to generator model
    #todo: edit reshape for 3:4 ratio

    #First layer
    model.add(tf.layers.Dense(7*7*256, input_dim=128, activation='relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape(7, 7, 128))

    #Second layer (upsampling?)
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    #Third layer (upsampling)
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    #Fourth layer (downsampling)
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    #Fifth layer (downsampling)
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    #Sixth layer (conv layer)
    model.add(Conv2D(1, kernel_size=4, activation='sigmoid', padding='same'))

    return model


def add_discriminator_layer(model, oc): #from rectgan/nets.py and https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
    model.add(BatchNormalization(momentum=momentum_BatchNormalization))
    model.add(Conv2D(out_channels=oc, kernel_size=3, strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha_Discriminator))

#other tutorial: https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
def discriminator(): #same tutorial, https://github.com/nicknochnack/GANBasics/blob/main/FashionGAN-Tutorial.ipynb
    model = Sequential() #FIGURE OUT STRIDES for a rectangular image!!!

    #First layer (conv layer) without batchnorm
    model.add(Conv2D(32, kernel_size=3, strides=strides[0], input_shape=painting_dimensions, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))#drop 25% of input units

    #Add conv layers?
    add_discriminator_layer(model, 128)
    add_discriminator_layer(model, 256)
    add_discriminator_layer(model, 512)
    model.add(Dropout(0.25))

    return model

# Build and compile the Discriminator
def build_discriminator(img_shape):
    discriminator = build_discriminator(img_shape)
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    return discriminator

# Build and compile the GAN (GDP)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')
    return model

# Combine the generator and discriminator
def build_gan_model(latent_dim, img_shape):
    generator = build_generator(latent_dim)
    discriminator = build_discriminator_model(img_shape)
    gan = build_gan(generator, discriminator)
    return generator, discriminator, gan

#tutorial: https://pylessons.com/gan-introduction
def discriminator_loss(real_output, fake_output):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
    fake_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(fake_output), logits=fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#tutorial: https://pylessons.com/gan-introduction
def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

#gdp
def train_GAN(epochs):
    for epoch in range(epochs):
        for _ in range(X_train.shape[0]):
            # Generate random noise as an input to the generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            # Generate fake images using the generator
            generated_images = generator.predict(noise)
            # Select a random batch of real images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]
            # Labels for real and fake images
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            # Train the discriminator on real and fake images
            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

            print(f"Epoch {epoch+1}, [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

