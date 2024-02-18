
# class Args:
#     OUTPUT_DIRECTORY = './images_V2I/images_GAN' #output file name
#     TRAINING_DIRECTORY = './images_V2I/output_training_set' #output file name
   
#     EPOCHS = 50
#     EXAMPLES_TO_GENERATE = 6
#     IMAGE_DIMENSIONS = (192, 256, 1) #3:4 aspect ratio
    
#     BATCH_SIZE = 6
#     BATCH_LENGTH = 100

#     strides = [(4,4), (3,4), (2,2)] #from rectGAN aspect_ratio
#     noise_dim = 100
#     # seed = tf.random.normal([EXAMPLES_TO_GENERATE, noise_dim])

#     ALPHA_DISCRIMINATOR = 0.2
#     ALPHA_GENERATOR = 0.2
#     MOMENTUM_BATCH_NORMALIZATION = 0.8 #rectgan suggests 0.3
#     ADAM_BETA = 0.5 #rectgan also suggests 0.5