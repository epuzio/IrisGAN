import tensorflow as tf
import os

TF_RECORD_PATH = 'output.tfrecord'

if os.path.exists(TF_RECORD_PATH):
    print("TFRecord file exists.")
else:
    print("TFRecord file does not exist.")
    
dataset = tf.data.TFRecordDataset(TF_RECORD_PATH, compression_type='GZIP')
print(dataset)
dataset = dataset.shuffle(60).batch(10)
print("Tensorflow dataset:", dataset)

# # Open the TFRecord file
# dataset = tf.data.TFRecordDataset(TF_RECORD_PATH, compression_type='GZIP')

# # Iterate over the dataset to inspect its contents
# for raw_record in dataset.take(1):  # Take the first 5 records for inspection
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)  # Print or inspect the parsed example