"""
This will test out some TensorFlow stuff. Currently it's a sandbox but maybe
I'll make it better. Look at the following scripts, which are run for MNIST:

tensorflow/tensorflow/examples/how_tos/reading_data/convert_to_records.py
tensorflow/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

PS: Github has copy path feature!! Yay!!

(c) November 2016 by Daniel Seita
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

# Daniel: mostly copied, except I don't need the validation flag.
tf.app.flags.DEFINE_string('directory', 'tmp',
                           'Directory to download data files and write the '
                           'converted result')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
    """ 
    Daniel: copied from
        https://github.com/tensorflow/tensorflow/blob/r0.11/
        tensorflow/examples/how_tos/reading_data/convert_to_records.py 
    with a few minor (obvious) changes.
    """
    num_examples = images.shape[0]
    assert num_examples == len(labels)

    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        if index % 1000 == 0:
            print("on image {}".format(index))
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    d = "final_data/breakout/"
    train_data   = np.load(d+ "train.data.npy")
    train_labels = np.load(d+ "train.labels.npy")
    valid_data   = np.load(d+ "valid.data.npy")
    valid_labels = np.load(d+ "valid.labels.npy")
    test_data    = np.load(d+ "test.data.npy")
    test_labels  = np.load(d+ "test.labels.npy")
    convert_to(train_data, train_labels, 'train')
    convert_to(valid_data, valid_labels, 'validation')
    convert_to(test_data, test_labels, 'test')
