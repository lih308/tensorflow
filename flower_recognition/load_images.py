import os
from pdb import set_trace as bp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_images(
        directory='./data/flowers/',
        num_epochs=2,
        batch_size=128,
):
    """ Read all files in a directory
    Args:
        directory: location where pictures are stored
        num_epochs: number of epochs
        batch_size: batch size

    Returns:
        Two tensors: one for image file names, one for image labels
    """
    image_list, label_list = read_labeled_image_list()
    train_image, train_label, test_image, test_label = split_train_and_test(
        image_list,
        label_list,
    )
    train_data = process_data(
        train_image,
        train_label,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    test_data = process_data(
        test_image,
        test_label,
        batch_size=1,
    )
    return train_data, test_data


def read_labeled_image_list(
        directory="./data/flowers/",
):
    """ Read all files in a directory
    Args:
        directory: location where pictures are stored
    Returns:
       List with all filenames in directory and list with labels for files
    """
    file_names = []
    labels = []

    for label, flower_name in enumerate(sorted(os.listdir(directory))):
        sub_directory = directory + flower_name + '/'
        images = os.listdir(sub_directory)
        images = [
            sub_directory + file_name for file_name in images
            if file_name.endswith('.jpg')
        ]
        file_names += images
        labels += [label] * len(images)
    return np.array(file_names), convert_to_one_hot(np.array(labels))


def split_train_and_test(
        image_list,
        label_list,
        train_ratio=0.9,
):
    assert len(image_list) == len(label_list)
    num = len(image_list)
    rand_num = np.arange(num)
    np.random.shuffle(rand_num)

    train_num = int(num * train_ratio)
    train_image = image_list[rand_num[: train_num]]
    train_label = label_list[rand_num[: train_num]]
    test_image = image_list[rand_num[train_num:]]
    test_label = label_list[rand_num[train_num:]]

    return train_image, train_label, test_image, test_label


def process_data(
        images,
        labels,
        num_epochs=1,
        batch_size=1,
):
    """ Process image/label list

    Args:
        images: list of images files
        labels: list of corresponding labels for images

    Returns:
        image: tensor of image data
        label: tensor of label data
    """
    image_paths = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int16)

    images = tf.map_fn(
        lambda image: read_images_from_disk(image),
        image_paths,
        dtype=tf.float32,
    )

    data_set = tf.data.Dataset.from_tensor_slices(
        (images, labels),
    )
    return data_set

    
def read_images_from_disk(file_name):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    file_contents = tf.read_file(file_name)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    return image


def convert_to_one_hot(
        labels,
        num_classes=None,
):
    if num_classes is None:
        num_classes = np.unique(labels).shape[0]
    labels = np.eye(num_classes)[labels.astype(int)]
    return labels


if __name__ == "__main__":
    pass
