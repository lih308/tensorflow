from pdb import set_trace as bp

import numpy as np
from tensorflow.python.framework import ops

from model import *
from load_images import load_images


def run():
    train_data, test_data = load_images(num_epochs=1)
    train_and_test(train_data, test_data)


if __name__ == '__main__':
    run()
