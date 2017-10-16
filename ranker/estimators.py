import tensorflow as tf
import features
import numpy as np

"""
This file will contain the code for the neural networks that will estimate
reward signals: both short-term signal (bot response being up- or down- voted)
and long term signal (average conversation score between 1 and 5).
"""

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


class FullyConnectedFeedForward(object):
    # TODO Fully connected feed forward neural net using tensorflow

    def __init__(self, features, activation, hidden_sizes, output_size, optimizer, drop_rate=0.0):
        self.features = features
        self.activation = activation
        self.n_hidden_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.optimizer = optimizer
        self.drop_rate = drop_rate

    def build(self, mode):
        # build the list of objects that will hold the input features, but article, context and candidate = None for now
        self.feature_objects = features.get(None, None, None, self.features)
        self.input_size = np.sum([f.dim for f in self.feature_objects])

        # x and y are each a placeholders: a value that we'll input when we ask TensorFlow to run a computation.
        # None indicates that the first dimension, corresponding to the batch size, can be of any size.
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='input')   # x dim: batch_size x input
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size], name='output')  # y dim: batch_size x output

        # Forward prediction: dense layers with activation - dropout - dense w/o activation
        dense = self.x  # (bs, in)
        for idx, hidd in enumerate(self.hidden_sizes):
            dense = tf.layers.dense(inputs=dense, units=hidd,
                                    activation=self.activation,
                                    name='dense_layer_%d' % idx)  # (bs, h_i)

        if self.drop_rate > 0.0:
            dense = tf.layers.dropout(inputs=dense, rate=self.drop_rate,
                                      training=mode == tf.estimator.ModeKeys.TRAIN,
                                      name='dropout_layer')  # (bs, h_last)

        self.logits = tf.layers.dense(inputs=dense, units=self.output_size, name='logits_layer')  # (bs, out)

        # Get predictions:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=self.logits, axis=1, name='argmax_predictions'),  # (out)
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`
            "probabilities": tf.nn.softmax(self.logits, name='softmax_tensor')  # (bs, out)
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        # onehot_labels = tf.one_hot(indices=tf.cast(self.y, tf.int32), depth=self.output_size)
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.logits)

        # TODO: continue from: https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0
        # TODO: and from: https://www.tensorflow.org/tutorials/layers#calculate_loss




