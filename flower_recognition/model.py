from pdb import set_trace as bp

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

from load_images import load_images


def generate_flower_model(
        x_input_shape,
        y_num_classes,
):
    """
    Models to identify flowers

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    input = Input(x_input_shape)

    x_input = ZeroPadding2D((3, 3))(input)
    x_input = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(x_input)
    x_input = BatchNormalization(axis = 3, name = 'bn0')(x_input)
    x_input = Activation('relu')(x_input)
    x_input = MaxPooling2D((2, 2), name='max_pool')(x_input)
    x_input = Flatten()(x_input)
    x_input = Dense(y_num_classes, activation='softmax', name='fc')(x_input)

    model = Model(
        inputs=input,
        outputs=x_input,
        name='Flower Model',
    )
    return model


def train_and_test(
        train_data,
        test_data,
        learning_rate=0.009,
        buffer_size=10,
        batch_size=128,
        num_epochs=10,
        print_cost=True,
):
    flower_model = generate_flower_model(
        train_data.output_shapes[0].as_list(),
        train_data.output_shapes[1].as_list()[0],
    )
    flower_model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    )


    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(iterator.initializer)

        for num_epoch in range(num_epoch):
            # train_data = train_data.repeat(num_epochs)
            train_data = train_data.shuffle(buffer_size)
            train_data = train_data.batch(batch_size)
            iterator = train_data.make_initializable_iterator()

            for num_batch in range(something_number):
                images, labels = iterator.get_next()

                fit = flower_model.fit(
                    x=images,  # .eval(),
                    y=labels,  # .eval(),
                    # epochs=num_epochs,
                    # batch_size=batch_size,
                )
    #  1. It seems I have to pass in data to flower_model.fit, not tensors. So I used .eval for tensors. Is this the right way?
    #  2. I want to run for several epochs, and with mini batches.
    #     But right now, the code is runnig num_epoch iteration, with batch_size for each epoch.
    #     How to properly iterate through all images with minibatches?
    # https://cedar.buffalo.edu/~srihari/CSE676/1.4.2%20Fizzbuzz.pdf
    # http://adventuresinmachinelearning.com/keras-lstm-tutorial/

if __name__ == '__main__':
    pass
