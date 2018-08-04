import math
from pdb import set_trace as bp

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(
        tf.float32,
        shape=[None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL],
        name='input_X',
    )
    Y = tf.placeholder(
        tf.float32,
        shape=[None, n_y],
        name='input_Y',
    )
    return X, Y


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
    W1 : [4, 4, 3, 8]
    W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    W1 = tf.get_variable(
        name="W1",
        shape=[4, 4, 3, 8],
        initializer=tf.contrib.layers.xavier_initializer(seed = 0),
    )
    W2 = tf.get_variable(
        name="W2",
        shape=[2, 2, 8, 16],
        initializer=tf.contrib.layers.xavier_initializer(seed = 0),
    )
    parameters = {
        "W1": W1,
        "W2": W2,
    }
    return parameters

                                                                                                                    
def forward_propagation(X, parameters, n_y):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters['W1']
    Z1 = tf.nn.conv2d(
        input=X,
        filter=W1,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='Z1',
    )
    A1 = tf.nn.relu(
        features=Z1,
        name='A1',
    )
    P1 = tf.nn.max_pool(
        value=A1,
        ksize=[1, 8, 8, 1],
        strides=[1, 8, 8, 1],
        padding='SAME',
        name='P1',
    )
    tf.summary.histogram('W1', W1)
    tf.summary.histogram('Z1', Z1)
    tf.summary.histogram('A1', A1)
    tf.summary.histogram('P1', P1)

    W2 = parameters['W2']
    Z2 = tf.nn.conv2d(
        input=P1,
        filter=W2,
        strides=[1, 1, 1, 1],
        padding='SAME',
        name='Z2',
    )
    A2 = tf.nn.relu(
        features=Z2,
        name='A2',
    )
    P2 = tf.nn.max_pool(
        value=A2,
        ksize=[1, 4, 4, 1],
        strides=[1, 4, 4, 1],
        padding='SAME',
        name='P2',
    )
    P2 = tf.contrib.layers.flatten(P2)
    tf.summary.histogram('W2', W2)
    tf.summary.histogram('Z2', Z2)
    tf.summary.histogram('A2', A2)
    tf.summary.histogram('P2', P2)

    Z3 = tf.contrib.layers.fully_connected(
        inputs=P2,
        num_outputs=n_y,
        activation_fn=None,
    )
    tf.summary.histogram('Z3', Z3)
    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    cost_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=Z3,
        labels=Y,
        name='item_cost',
    )
    cost = tf.reduce_mean(
        input_tensor=cost_entropy,
        name='cost',
    )
    tf.summary.scalar('cost', cost)

    return cost


def model(
        X_train,
        Y_train,
        X_test,
        Y_test,
        learning_rate=0.009,
        num_epochs=100,
        minibatch_size=64,
        print_cost=True,
):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    # ops.reset_default_graph()
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape.as_list() if isinstance(X_train, tf.Tensor) else X_train.shape
    n_y = Y_train.shape.as_list()[1] if isinstance(Y_train, tf.Tensor) else Y_train.shape[1]
    costs = []
    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters, n_y)
    cost = compute_cost(Z3, Y)

    merged_summary = tf.summary.merge_all()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver()
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    )
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init_op)
        threads = tf.train.start_queue_runners(
            sess=sess,
            coord=coord,
        )
        ckpt = tf.train.get_checkpoint_state('./model/')
        writer = tf.summary.FileWriter('./model/tmp/tensorboard/3')

        try:
            count = 0
            while not coord.should_stop():
                _cost = X_train.eval(session=sess).mean()
                # _, _cost = sess.run(
                #     [optimizer, cost],
                #     feed_dict={
                #         X: X_train.eval(session=sess),
                #         Y: Y_train.eval(session=sess),
                #     },
                # )
                count += 1
                if count % 1 == 0:
                    print(count, 'th minibatch with cost', _cost)
        except:
            coord.request_stop()
            coord.join(threads)

        writer.add_graph(sess.graph)
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({
            X: X_train.eval(session=sess),
            Y: Y_train.eval(session=sess),
        })
        test_accuracy = accuracy.eval({
            X: X_test.eval(session=sess),
            Y: Y_test.eval(session=sess),
        })
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return train_accuracy, test_accuracy, parameters


def predict(X, parameters):
    params = {
        key: tf.convert_to_tensor(value)
        for key, value in parameters.items()
    }

    # (m, n_H0, n_W0, n_C0) = X.shape             
    x = tf.placeholder("float", [1, 64, 64, 3])  # [None, n_H0, n_W0, n_C0])
    
    z3 = forward_propagation(x, params)
    # p = tf.argmax(z3)
    
    with tf.Session() as sess:
        prediction = sess.run(
            z3, # p,
            feed_dict = {x: X},
        )
    return prediction
