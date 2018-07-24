import os
from datetime import datetime as dt
from pdb import set_trace as bp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_SIZE = 256


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
        images = os.listdir(directory + '/' + label)
        images = [
            file_name for file_name in images[:10]
            if file_name[-4:] == '.jpg'
        ]
        file_names += images
        labels += [label] * len(images)

    return file_names, labels


def read_images_from_disk(file_name, label): # input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    # label = input_queue[1]
    file_contents = tf.read_file(file_name)  # input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


def run(
        num_epochs=2,
        batch_size=16,
):
    image_list, label_list = read_labeled_image_list()

    sess = tf.InteractiveSession()
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(read_images_from_disk)

    bp()
    # Makes an input queue
    # input_queue = tf.train.slice_input_producer(
    #     tensor_list=[images, labels],
    #     num_epochs=num_epochs,
    # )

    # image, label = read_images_from_disk(input_queue)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    # image = preprocess_image(image)
    # label = preprocess_label(label)

    # Optional Image and Label Batching
    # image_batch, label_batch = tf.train.batch(
    #     [image, label],
    #     batch_size=batch_size,
    # )
 
    # with tf.Session() as sess:
    #     sess.run(tf.local_variables_initializer())
    #     bp()
    #     # results = sess.run([image, label])
    #     results = sess.run(dataset)
 
if __name__ == "__main__":
    run()
