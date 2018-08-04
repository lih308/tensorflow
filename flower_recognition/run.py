from pdb import set_trace as bp

import numpy as np
from tensorflow.python.framework import ops

from model import *
from load_images import load_images


def run():
    ops.reset_default_graph()
    train_image_batch, train_label_batch, test_image, test_label = load_images(num_epochs=1)
    results = model(
        X_train=train_image_batch,
        Y_train=train_label_batch,
        X_test=test_image,
        Y_test=test_label,
        num_epochs=100,
    )


if __name__ == '__main__':
    run()
