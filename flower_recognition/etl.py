import os
from pdb import set_trace as bp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_image_list(
        directory="./data/flowers/",
):
    """ Read all files in a directory
    Args:
        directory: location where pictures are stored
    Returns:
       List with all filenames in directory and list with labels for files
    """
    images = tf.matching_files(
        pattern=tf.constant(directory + '*/*.jpg', dtype=tf.string)
    )
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = images.map(
        lambda x: tf.size(x)
    )
    return images, labels


def run():
    sess = tf.InteractiveSession()
    images = load_image_list()

    images, labels = sess.run(images)
    tmp = images.eval()
    tmp2 = labels.eval()
    sess.close()


if __name__ == '__main__':
    run()
