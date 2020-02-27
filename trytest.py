#!/usr/bin/env python
"""This file trains the model upon the training data and evaluates it with the test data.
It uses the arguments it got via the gcloud command."""
print("aaa")
import argparse
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow as tf

from tensorflow.contrib.training.python.training import hparam
#from tensorflow.compat.v1.contrib.training.python.training import hparam
print("a1111")
import trainer.data as data
import trainer.model as model
print("bbb")
PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    '--eval-steps',
    help='Number of steps to run evaluation for at each checkpoint',
    default=1,
    type=int
)

ARGS = PARSER.parse_args(["--eval-steps", "5"])
tf.logging.set_verbosity('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)
print("ccc")
HPARAMS = hparam.HParams(**ARGS.__dict__)
params = HPARAMS
print("before data ingestion")

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
print("bbb")
for _ in range(params.eval_steps):
    estimator.train(train_input_fn, steps=steps_per_eval)
    estimator.evaluate(eval_input_fn)
