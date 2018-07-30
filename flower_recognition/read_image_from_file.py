import os
from datetime import datetime as dt
from pdb import set_trace as bp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_SIZE = 256
IMG_WIDTH = 64
IMG_HEIGHT = 64


def read_labeled_image_list(
        directory="./data/flowers/",
):
    """ Read all files in a directory
    Args:
        directory: location where pictures are stored
    Returns:
       List with all filenames in directory
    """
    file_names = []
    labels = []

    for label in os.listdir(directory):
        sub_directory = directory + label + '/'
        images = os.listdir(sub_directory)
        images = [
            sub_directory + file_name for file_name in images[:10]
            if file_name[-4:] == '.jpg'
        ]
        file_names += images
        labels += [label] * len(images)

    return file_names, labels


def read_images_from_disk(file_name, label):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    file_contents = tf.read_file(file_name)
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example = tf.image.resize_images(example, [IMG_HEIGHT, IMG_WIDTH])
    return example, label


def run(
        num_epochs=2,
        batch_size=16,
):
    image_list, label_list = read_labeled_image_list()
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.string)

    image, label = tf.train.slice_input_producer(
        tensor_list=[images, labels],
        num_epochs=num_epochs,
    )
    image, label = read_images_from_disk(image, label)
    bp()

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    # image = preprocess_image(image)
    # label = preprocess_label(label)

    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
    )
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    )

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.start_queue_runners()
        bp()
        results = sess.run(image_batch)

 
if __name__ == "__main__":
    run()
