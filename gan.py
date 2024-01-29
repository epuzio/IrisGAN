#rectGAN: https://github.com/xjdeng/RectGAN/blob/master/aspect_ratio.py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt



painting_dimensions = (192, 256, 1) #we want 3:4 ratio for final paintings
strides = [(4,4), (3,4), (2,2)] #from rectGAN

samples = tf.placeholder(tf.float32, [None, 100])
noise = tf.placeholder(tf.float32, [None, 100])

#Load from images
(X_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0



# Generator model
def generator():
    model = tf.keras.Sequential() #add layers to generator model
    #todo: edit reshape for 3:4 ratio

    #First layer (https://www.youtube.com/watch?v=AALBGpLbj6Q)
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

#other tutorial: https://github.com/vmvargas/GAN-for-Nuclei-Detection/blob/master/model-MNIST-cross-validation.py
def discriminator(): #same tutorial, https://github.com/nicknochnack/GANBasics/blob/main/FashionGAN-Tutorial.ipynb
    model = Sequential() #FIGURE OUT STRIDES for a rectangular image!!!

    #First layer (conv layer)
    model.add(Conv2D(32, kernel_size=3, strides=strides[0], input_shape=painting_dimensions, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))#drop 25% of input units

    #Second layer (conv layer)
    model.add(Conv2D(64, kernel_size=3, strides=strides[1], padding='same'))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=strides[2], padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=strides[2], padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    #Second layer (conv layer)
    # model.add(Conv2D(128, 5))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))

    # #Third layer (conv layer)
    # model.add(Conv2D(256, 5))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))

    # model.add(Flatten())
    # model.add(Dropout(0.4))
    # model.add(Dense(1, activation='sigmoid'))

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

