import tensorflow as tf
import numpy as np
import cPickle as pkl
import argparse
import sys

import features


FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def model1_fn(features, labels, mode, params):
    """
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object

    :param features: A dict containing the features passed to the model via `input_fn`
    :param labels: A Tensor containing the labels passed to the model via `input_fn` -- (bs,)
        Will be empty for predict() calls, as these are the values the model will infer
    :param model: indicating the context in which the model_fn was invoked:
        tf.estimator.ModeKeys.TRAIN -- in training mode, namely via a `train()` call
        tf.estimator.ModeKeys.EVAL  -- in evaluation mode, namely via an `evaluate()` call
        tf.estimator.ModeKeys.PREDICT -- in predict mode, namely via a `predict()` call
    :param params: additional parameters passed as a dictionary:
        + 'input_size' --> int ; size of input layer
        + 'hidden_sizes' --> array of ints ; size of each hidden layers
        + 'activation' --> tf function (ex: tf.nn.swish, tf.nn.relu, tf.sigmoid, ...)
        + 'drop_rate' --> float ; probability of dropping a dimension
        + 'optimizer' --> tf Optimizer Object used for training (ex: tf.train.AdamOptimizer())

    :return: a tf.estimator.EstimatorSpec object, which contains the following values:
        + mode (required) -- mode in which the model was run
        + predictions (required in PREDICT mode) -- dict that maps key names to prediction Tensors from the model
        + loss (required in EVAL and TRAIN mode) -- Tensor containing model's loss function calculated over all input examples
        + train_op (required only in TRAIN mode) -- Op that runs one step of training
        + eval_metric_ops (optional) -- dict of name/value pairs specifying the metrics that will be calculated when the model runs in EVAL mode.
    """
    for p in ['input_size', 'hidden_sizes', 'activation', 'drop_rate', 'optimizer']
    assert p in params

    # Create input layer tensor
    input_layer = tf.zeros(
        shape = (None, params['input_size']),  # (bs, input_dim)
        dtype = tf.float32,
        name  = "input_layer"
    )
    # fill in the input layer with the values
    prev_idx = 0
    for f in features:
        f_dim = features[f].shape[1]
        input_layer[:, prev_idx: prev_idx+f_dim] = features[f]  # (bs, f_dim)
        prev_idx += f_dim

    # input_layer = tf.reshape(features["x"], [None, params['input_size']], "input_layer")  # (batch_size, input_dim)

    # Forward prediction:
    # (1) dense layers with activation
    dense = input_layer  # (bs, in)
    for idx, hidd in enumerate(params['hidden_sizes']):
        dense = tf.layers.dense(inputs     = dense,
                                units      = hidd,
                                activation = params['activation'],
                                name       = 'dense_layer_%d' % idx)  # (bs, h_i)
    # (2) dropout
    if params['drop_rate'] > 0.0:
        dense = tf.layers.dropout(inputs   = dense,
                                  rate     = params['drop_rate'],
                                  training = (mode == tf.estimator.ModeKeys.TRAIN),
                                  name     = 'dropout_layer')  # (bs, h_last)
    # (3) dense w/o activation
    logits = tf.layers.dense(inputs = dense,
                             units  = 2,  # binary classification: predict +1/-1
                             # no activation: linear activation
                             name   = 'logits_layer')  # (bs, 2)

    # The logits layer of our model returns our predictions as raw values in a [batch_size, 2]-dimensional tensor.
    # Let's convert these raw values into two different formats that our model function can return:
    # - The predicted class for each example: either 1 or 0
    # - The probabilities for each possible target class for each example
    predictions = {
        "classes"       : tf.argmax(logits, axis=1),  # (bs,)
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")  # (bs, 2)
    }
    # RETURN HERE IN PREDICT MODE
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # convert list of labels [0, 1, ..., 1] ~ (bs,) to one-hot encodings [[1,0], [0,1], ..., [0,1]] ~ (bs, 2)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)  # (bs, 2)
    # define the cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # RETURN HERE IN TRAINING MDE
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = params['optimizer'].minimize(
            loss        = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # RETURN HERE IN EVAL MODE
    # add the accuracy metric:
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def model2_fn():
    pass


def main(unused_argv):
    """
    tuto: https://www.tensorflow.org/tutorials/layers#evaluate_the_model
    """
    # TODO: load datasets
    #
    # map from feature_name to numpy array of size (data, feature_length)
    # x_train = {
    #     <name of feature 1>: np.array( [[f1_dim],...,[f1_dim]] ),
    #     <name of feature 2>: np.array( [[f2_dim],...,[f2_dim]] ),
    #     ...
    # ]
    # y_train = [0, 1, 0, ..., 1]
    # input_size = feat_1.dim + ... + feat_n.dim

    # TODO: parse arguments!
    FLAGS['hidden_sizes_1']
    if FLAGS['activation_1'] == 'swish':
        activation_1 = tf.nn.swish
    FLAGS['dropout_rate_1']

    # Build first RNN to predict immediate reward: +1 / -1
    nn1 = tf.estimator.Estimator(
        model_fn  = model1_fn,
        params    = {
            'input_size'  : input_size,
            'hidden_sizes': [500, 100, 10],
            'activation'  : tf.nn.swish,  # try tf.nn.swish | tf.nn.relu | tf.sigmoid
            'drop_rate'   : 0.1,  # try 0.1 | 0.3 | 0.5 | 0.7 | 0.9
            'optimizer'   : tf.train.AdamOptimizer()
        },
        model_dir = "./models/vote_estimator_model"
    )

    # Set up logging for predictions
    tensors_to_log = {  # key = string to be printed && value = name of tensor to print
        "probabilities": "softmax_tensor"
    }
    # log probabilities after every 50 steps of training
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    # Define the training inputs: use tf.estimator.inputs.numpy_input_fn to produce the input pipeline
    # Returns input function that would feed dict of numpy arrays into the model.
    # This returns a function outputting features and target based on the dict of numpy arrays.
    train_input_fn1 = tf.estimator.inputs.numpy_input_fn(
        x          = x_train,  # map from 'feature_name' to numpy array ~ (data, feat_dim)
        # x = {"x": train_data},
        y          = y_train,  # list of labels (0 for downvote / 1 for upvote)
        batch_size = 128,      # number of examples per training step
        num_epochs = None,     # train until specified number of train steps is reached
        shuffle    = True      # shuffle the training data
    )
    # Train the model
    nn1.train(
        input_fn = train_input_fn1,
        steps    = 1000,  # train 1000 steps (*batch_size = number of seen examples) 
        hooks    = [logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x          = x_test,
        y          = y_test,
        num_epochs = 1,  # evaluates the metrics over one epoch of data only
        shuffle    = False  # iterate through the data sequentially
    )
    eval_results = nn1.evaluate(input_fn=eval_input_fn)  # return eval_metric_ops
    print(eval_results)


    ###
    # SECOND PREDICTOR
    ###
    # nn2 = tf.estimator.Estimator(
    #     model_fn  = model2_fn,
    #     params    = {},
    #     model_dir = "./models/finalscore_model"
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("-h1", "--hidden_sizes_1", nargs='+', type=int, default=[500, 100, 10], help="List of hidden sizes for first network")
    parser.add_argument("-h2", "--hidden_sizes_2", nargs='+', type=int, default=[500, 100, 10], help="List of hidden sizes for second network")
    parser.add_argument("-a1", "--activation_1", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for first network")
    parser.add_argument("-a2", "--activation_2", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for second network")
    parser.add_argument("-d1", "--dropout_rate_1", type=float, default=0.1, help="Probability of dropout layer in first network")
    parser.add_argument("-d2", "--dropout_rate_2", type=float, default=0.1, help="Probability of dropout layer in second network")
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

