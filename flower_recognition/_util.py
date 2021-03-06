import math
from pdb import set_trace as bp

import numpy as np
import h5py
import tensorflow as tf

from preprocess_data import load_images


def load_dataset():
    full_dataset = load_images()
    train_x, train_y, test_x, test_y = split_train_test_data(full_dataset)
    train_y = convert_to_one_hot(train_y)
    test_y = convert_to_one_hot(test_y)
    return train_x, train_y, test_x, test_y


def split_train_test_data(
        full_dataset,
        test_ratio=0.1,
):
    train, test = [], []
    for flower in sorted(full_dataset.keys()):
        all_images = full_dataset[flower]
        num_images = all_images.shape[0]

        rand_num = np.arange(num_images)
        np.random.shuffle(rand_num)
        _test = np.array(all_images)[rand_num[:int(num_images * test_ratio)]]
        _train = np.array(all_images)[rand_num[int(num_images * test_ratio):250]]
        train.append(_train)
        test.append(_test)

    train_x = np.concatenate(train, axis=0)
    test_x = np.concatenate(test, axis=0)

    train_y = np.concatenate([
        np.ones([values.shape[0]]) * i
        for i, values in enumerate(train)
    ])
    test_y = np.concatenate([
        np.ones([values.shape[0]]) * i
        for i, values in enumerate(test)
    ])
    return train_x, train_y, test_x, test_y


def random_mini_batches(
        X,
        Y,
        mini_batch_size=64,
        seed=0,
):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    input_size = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(input_size))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = math.floor(input_size/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if input_size % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size :, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(
        Y,
        C=None,
):
    if C is None:
        C = np.unique(Y).shape[0]
    Y = np.eye(C)[Y.astype(int)]
    return Y


def predict(X, parameters):
    params = {
        key: tf.convert_to_tensor(value)
        for key, value in parameters.items()
    }

    # (m, n_H0, n_W0, n_C0) = X.shape             
    x = tf.placeholder("float", [1, 64, 64, 3])  # [None, n_H0, n_W0, n_C0])
    
    z3 = forward_propagation_for_predict(x, params)
    # p = tf.argmax(z3)
    
    with tf.Session() as sess:
        prediction = sess.run(
            z3, # p,
            feed_dict = {x: X},
        )
    return prediction
