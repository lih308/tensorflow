import os
from pdb import set_trace as bp

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def test():
    sess = tf.InteractiveSession()

    directory = './data/flowers/tulip/'
    tulip_file = directory + os.listdir(directory)[4]
    
    tulip_tensor = tf.convert_to_tensor(tulip_file, dtype=tf.string)
    tulip_image = tf.read_file(tulip_file)
    tulip_data = tf.image.decode_jpeg(tulip_image, channels=3)
    tulip_resized = tf.image.resize_images(
        tulip_data,
        size=[128, 128],
    )
    return tulip_data.eval(), tulip_resized.eval()


if __name__ == '__main__':
    test()
