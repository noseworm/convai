import tensorflow as tf
import numpy as np
import cPickle as pkl

import argparse
import pyprind
import random
import copy
import sys
import os

import features


TARGET_TO_FEATURES = {
    'r': ['AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_Article',
          'GreedyScore_CandidateUser', 'AverageScore_CandidateUser', 'ExtremaScore_CandidateUser',
          'EntityOverlap', 'BigramOverlap', 'TrigramOverlap', 'WhWords', 'LastUserLength', 'CandidateLength'],
    'R': ['AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK', 'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
          'GreedyScore_CandidateUser', 'AverageScore_CandidateUser', 'ExtremaScore_CandidateUser',
          'GreedyScore_CandidateLastK', 'AverageScore_CandidateLastK', 'ExtremaScore_CandidateLastK',
          'GreedyScore_CandidateLastK_noStop', 'AverageScore_CandidateLastK_noStop', 'ExtremaScore_CandidateLastK_noStop',
          'GreedyScore_CandidateKUser', 'AverageScore_CandidateKUser', 'ExtremaScore_CandidateKUser',
          'GreedyScore_CandidateKUser_noStop', 'AverageScore_CandidateKUser_noStop', 'ExtremaScore_CandidateKUser_noStop',
          'EntityOverlap','BigramOverlap','TrigramOverlap','WhWords','DialogLength','LastUserLength','ArticleLength','CandidateLength']
}


# def weight_variable(name, shape, mean=0., stddev=0.1):
#     """
#     Build a tensorflow variable oject initialized with the normal distribution
#     :param name: name of the variable
#     :param shape: shape of the variable
#     :param mean: the mean of the normal distribution to sample from
#     :param stddev: the standard deviation of the normal distribution to sample from
#     :return: tf.Variable() object to update during training
#     """
#     initial = tf.truncated_normal(shape, mean=mean, stddev=stddev, name=name)
#     return tf.Variable(initial)


# def bias_variable(name, shape, cst=0.1):
#     """
#     Build a tensorflow variable oject initialized to some value
#     :param name: name of the variable
#     :param shape: shape of the variable
#     :param cst: the initial value of the variable
#     :return: tf.Variable() object to update during training
#     """
#     initial = tf.constant(cst, shape=shape, name=name)
#     return tf.Variable(initial)


def build_model1(input_dim, hidden_dims, activation, optimizer):
    """
    Build the model to estimate immediate reward: +1 or -1
    :param input_dim: the number of neurons in the input layer
    :param hidden_dims: list of ints specifying the size of each hidden layer
    :param activation: tensor activation function to use at each layer
    :param optimizer: tensorflow optimizer object to train the network
    :return: predictions (dict of predicted classes and probabilities),
             loss, training operator, and accuracy
    """

    x = tf.placeholder(tf.float32, shape=[None, input_dim], name="input_layer")  # (bs, feat_size)
    y = tf.placeholder(tf.float32, shape=[None, 1], name="labels")  # (bs, 1)

    # Fully connected dense layers
    h_fc = x  # (bs, in)
    for idx, hidd in enumerate(hidden_dims):
        h_fc = tf.layers.dense(inputs     = h_fc,
                               units      = hidd,
                               activation = activation,
                               name       = 'dense_layer_%d' % idx+1)  # (bs, hidd)
    # Dropout layer
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # proba of keeping the neuron
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
    # Output layer
    logits = tf.layers.dense(inputs = h_fc_drop,
                             units  = 2,
                             # no activation for the logits
                             name   = 'logits_layer')  # (bs, 2)

    # Define prediction: class label (0,1) and the class probabilities:
    predictions = {
        "classes"       : tf.argmax(logits, axis=1, name="pred_classes"),  # (bs,)
        "probabilities" : tf.nn.softmax(logits, name="pred_probas")  # (bs, 2)
    }

    # Loss tensor:
    # create one-hot labels
    onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=2)  # (bs, 2)
    # define the cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Train operator:
    train_step = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Accuracy tensor:
    accuracy = tf.metrics.accuracy(labels=y, predictions=predictions["classes"], name="accuracy")
    
    return predictions, loss, train_step, accuracy


def main(args):
    # Load datasets
    input_size, (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data('r', voted_only=True)

    # parse arguments
    if args.activation_1 == 'swish':
        activation_1 = lambda x: x*tf.sigmoid(x)
    elif args.activation_1 == 'relu':
        activation_1 = tf.nn.relu
    elif args.activation_1 == 'sigmoid':
        activation_1 = tf.sigmoid
    else:
        print "ERROR: unknown activation 1: %s" % args.activation_1
        return

    # TODO: save model when early stop
    model1_dir = "./models/vote_estimator_model"
    # TODO: use batch size that make sense according to data_size
    steps = len(y_train) / FLAGS.batch_size

    predictions, loss, train_step, accuracy = build_model1(
        input_size,
        args.hidden_sizes_1,
        activation_1,
        tf.train.AdamOptimizer()
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize model
        for epoch in range(20000):  # will probably stop before 20k epochs due to early stop
            batch = data.train.next_batch(args.batch_size)  # TODO: get data batch loader
            while batch is not None:
                # train 
                train_step.run(
                    feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0-args.dropout_rate_1}
                )
                # evaluate (so no dropout) train accuracy
                train_acc = accuracy.eval(
                    feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0}
                )
                print "epoch %d: train accuracy: %g" % (epoch, train_acc)
                # get next training batch
                batch = data.train.next_batch(args.batch_size)
            # TODO: Evaluate on validation set and early stop & save!



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs='+', type=str, help="List of files to consider for training")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    # training parameters:
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during training")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Number of training steps to wait before stoping when validatiaon accuracy doesn't increase")
    # network architecture:
    parser.add_argument("-h1", "--hidden_sizes_1", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for first network")
    parser.add_argument("-h2", "--hidden_sizes_2", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for second network")
    parser.add_argument("-a1", "--activation_1", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for first network")
    parser.add_argument("-a2", "--activation_2", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for second network")
    parser.add_argument("-d1", "--dropout_rate_1", type=float, default=0.1, help="Probability of dropout layer in first network")
    parser.add_argument("-d2", "--dropout_rate_2", type=float, default=0.1, help="Probability of dropout layer in second network")
    args = parser.parse_args()
    print args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    main(args)

