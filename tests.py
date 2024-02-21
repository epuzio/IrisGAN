import tensorflow as tf
import os

TF_RECORD_PATH = 'output.tfrecord'
EPOCHS = 3
EXAMPLES_TO_GENERATE = 6
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256
BATCH_SIZE = 8 #From tips and tricks article
BUFFER_SIZE = 80 #should be around the size of the dataset! Shouldn't be hardcoded
TF_RECORD_PATH = "output.tfrecord"

if os.path.exists(TF_RECORD_PATH):
    print("TFRecord file exists.")
else:
    print("TFRecord file does not exist.")
    







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




dataset = tf.data.TFRecordDataset(TF_RECORD_PATH, compression_type='GZIP')
dataset = dataset.map(read_tfrecord) #Unpacks from string to a float32 with correct dims under shape

print("Shuffling dataset, creating batches...")
train_dataset = dataset.shuffle(60).batch(8)
print("Dataset:", dataset)
print("Train dataset:", train_dataset)

print(tf.compat.v1.WholeFileReader(train_dataset))

# # Open the TFRecord file
# dataset = tf.data.TFRecordDataset(TF_RECORD_PATH, compression_type='GZIP')

# # Iterate over the dataset to inspect its contents
# for raw_record in dataset.take(1):  # Take the first 5 records for inspection
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)  # Print or inspect the parsed example