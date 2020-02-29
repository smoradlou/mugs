#!/usr/bin/env python
"""This file trains the model upon the training data and evaluates it with the test data.
It uses the arguments it got via the gcloud command."""

import argparse
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datetime
# import tensorflow
from tensorflow.contrib.training.python.training import hparam
# from tensorflow.compat.v1.contrib.training.python.training import hparam
# import tensorboard.plugins.hparams as hparam
# from tensorboard.plugins.hparams import api as hp

import data as data
import model as model

# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.INFO)
def train_model(params):
    """The function gets the training data from the training folder,
    the evaluation data from the test folder and trains your solution from the model.py file with it."""
    (train_data, train_labels) = data.create_data_with_labels("data/train/")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=model.get_batch_size(),
        num_epochs=None,
        shuffle=True)

    (eval_data, eval_labels) = data.create_data_with_labels("data/test/")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    estimator = tf.estimator.Estimator(model_fn=model.solution)

    steps_per_eval = int(model.get_training_steps() / params.eval_steps)

    for _ in range(params.eval_steps):
        estimator.train(train_input_fn, steps=steps_per_eval)
        estimator.evaluate(eval_input_fn)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=1,
        type=int
    )

    ARGS = PARSER.parse_args()
    tf.logging.set_verbosity('INFO')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(logger.__dict__[] / 10)

    HPARAMS = hparam.HParams(**ARGS.__dict__)
    train_model(HPARAMS)