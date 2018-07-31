from pdb import set_trace as bp

import numpy as np

from model import *
from util import *


def run():
    x_train, y_train, x_test, y_test = load_dataset()
    results = model(
        X_train=x_train,
        Y_train=y_train,
        X_test=x_test,
        Y_test=y_test,
        num_epochs=100,
    )


def test_split_train_test_data():
    data = {
        'daisy': np.random.rand(100, 64, 64, 3),
        'lily': np.random.rand(100, 64, 64, 3),
        'rose': np.random.rand(100, 64, 64, 3),
    }
    split_train_test_data(data)


if __name__ == '__main__':
    run()
