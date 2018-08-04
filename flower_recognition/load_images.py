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
    train_image_batch, train_label_batch = process_image(
        train_image,
        train_label,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    test_image, test_label = process_image(
        test_image,
        test_label,
        batch_size=1,
    )
    return train_image_batch, train_label_batch, test_image, test_label


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


def process_image(
        images,
        labels,
        num_epochs=1,
        batch_size=None,
):
    """ Process image/label list

    Args:
        images: list of images files
        labels: list of corresponding labels for images

    Returns:
        image: tensor of image data
        label: tensor of label data
    """
    images = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int16)

    image, label = tf.train.slice_input_producer(
        tensor_list=[images, labels],
        num_epochs=num_epochs,
    )
    image, label = read_images_from_disk(image, label)
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
    )
    return image_batch, label_batch
    
    
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


def convert_to_one_hot(
        Y,
        C=None,
):
    if C is None:
        C = np.unique(Y).shape[0]
    Y = np.eye(C)[Y.astype(int)]
    return Y


def run():
    train_image_batch, train_label_batch, test_image, test_label = load_images()
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    )

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(
            sess=sess,
            coord=coord,
        )

        bp()
        pass


if __name__ == "__main__":
    run()
