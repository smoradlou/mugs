#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf

def get_training_steps():
    """Returns the number of batches that will be used to train your solution.
    It is recommended to change this value."""
    return 50


def get_batch_size():
    """Returns the batch size that will be used by your solution.
    It is recommended to change this value."""
    return 100


def solution(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using the solution that you have to write below."""
    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])
    # input_layer = input_layer/255.0

    # TODO: Code of your solution

    # define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    logits = model(input_layer, training=False)

    # mymodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # mymodel.summary()
    # est_model = tf.keras.estimator.model_to_estimator(keras_model=mymodel)


    if mode == tf.estimator.ModeKeys.PREDICT:
        # return tf.estimator.EstimatorSpec with prediction values of all classes
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    optimizer = tf.train.AdamOptimizer()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / get_batch_size())

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: Let the model train here
        #mymodel.fit(input_layer, labels, epochs=5)
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss, 
            train_op=optimizer.minimize(loss)
            )
    if mode == tf.estimator.ModeKeys.EVAL:
        # The classes variable below exists of an tensor that contains all the predicted classes in a batch
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops = eval_metric_ops)
