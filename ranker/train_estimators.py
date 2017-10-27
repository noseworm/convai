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


def get_data(files, target, voted_only=False, val_prop=0.1, test_prop=0.1):
    """
    Load data to train ranker
    :param files: list of data files to load
    :param target: field of each dictionary instance to estimate
        either 'R' for final dialog score, or 'r' for immediate reward
    :param voted_only: load messages which have been voted only
    :param val_prop: proportion of data to consider for validation set
    :param test_prop: proportion of data to consider for test set
    :return: input size (int), train x & y, test x & y
        x = numpy array of size (data, feature_length)
          ie: np.array( [[f1, f2, ..., fn], ..., [f1, f2, ..., fn]] )
        y = array of label values to predict
        input_size = feat_1.dim + ... + feat_n.dim
    """
    assert target in ['R', 'r'], "Unknown target: %s" % target

    print "\nLoading data..."
    data = []
    file_ids = []
    for data_file in files:
        if voted_only and data_file.startswith('voted_data_'):
            with open(data_file, 'rb') as handle:
                data.extend(pkl.load(handle))
                # get the time id of the data
                file_ids.append(data_file.split('_')[-1].replace('pkl', ''))
        elif (not voted_only) and data_file.startswith('full_data_'):
            with open(data_file, 'rb') as handle:
                data.extend(pkl.load(handle))
                # get the time id of the data
                file_ids.append(data_file.split('_')[-1].replace('pkl', ''))

    # if didn't get train/valid/test data, build your own
    if len(data) > 3:
        print "got %d examples" % len(data)

        # shuffle data TODO: make sure same article not present in train, valid, test set
        random.shuffle(data)

        train_val_idx = int(len(data) * (1-val_prop-test_prop))  # idx to start validation
        val_test_idx = int(len(data) * (1-test_prop))  # idx to start test set
        train_data = data[:train_val_idx]
        valid_data = data[train_val_idx: val_test_idx]
        test_data = data[val_test_idx:]
        print "train: %d" % len(train_data)
        print "valid: %d" % len(valid_data)
        print "test: %d" % len(test_data)

        # create list of Feature instances
        feature_objects = features.get(None, None, None, TARGET_TO_FEATURES[target])
        input_size = np.sum([f.dim for f in feature_objects])

        # construct data to save & return
        train_x = np.zeros((len(train_data), input_size))
        train_y = []
        valid_x = np.zeros((len(valid_data), input_size))
        valid_y = []
        test_x = np.zeros((len(test_data), input_size))
        test_y = []

        # TODO: continue corrections below!!
        print "building data..."
        bar = pyprind.ProgBar(len(data), monitor=False, stream=sys.stdout)  # show a progression bar on the screen
        for (x, y, data) in [(train_x, train_y, train_data), (valid_x, valid_y, valid_data), (test_x, test_y, test_data)]:
            for idx, msg in enumerate(data):
                # set x for each feature for that msg
                for f in feature_objects:
                    f.set(msg['article'], msg['context'], msg['candidate'])
                    x[f.__class__.__name__][idx, :] = np.array(copy.deepcopy(f.feat), dtype=np.float32)
                # set y labels
                if target == 'r':
                    if int(msg[target]) == -1:  y.append(0)
                    elif int(msg[target]) == 1: y.append(1)
                    else: print "ERROR: unknown immediate reward value: %s" % msg[target]
                else:
                    y.append(msg[target])
                bar.update()
        train_y = np.array(train_y)
        valid_y = np.array(valid_y)
        test_y = np.array(test_y)

        for f in feature_objects:
            assert train_x[f.__class__.__name__].shape[0] == len(train_y)
            assert test_x[f.__class__.__name__].shape[0] == len(test_y)

        # save train/val/test data to pkl file
        file_name = ""
        if voted_only:
            file_name += "voted_data_"
        else:
            file_name += "full_data_"
        file_name += "train-val-test_"
        for file_id in file_ids:
            file_name += file_id
        print "saving in %spkl..." % file_name
        with open(file_name+'pkl', 'wb') as handle:
            pkl.dump(
                [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)],
                handle,
                pkl.HIGHEST_PROTOCOL
            )
        print "done."

    # got train/valid/test data
    elif len(data) == 3:
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data
        input_size = np.sum([feat.shape[1] for feat in train_x.values()])
        print "train: %d" % len(train_y)
        print "valid: %d" % len(valid_y)
        print "test: %d" % len(test_y)

    else:
        print "Unknown data format: data length is less than 3."
        return

    return input_size, (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


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
    # model1_dir = "./models/vote_estimator_model"
    # TODO: use batch size that make sense according to data_size
    # steps = len(y_train) / FLAGS.batch_size

    predictions, cross_entropy, train_step, accuracy = build_model1(
        input_size,
        args.hidden_sizes_1,
        activation_1,
        tf.train.AdamOptimizer()
    )

    train_accuracies = []
    valid_accuracies = []
    best_valid_acc = 0.0
    patience = args.patience
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initialize model
        for epoch in range(20000):  # will probably stop before 20k epochs due to early stop
            # do 1 epoch: go through all training_batches
            for idx in range(0, len(y_train), args.batch_size):
                _, loss = sess.run(
                    [train_step, cross_entropy],
                    feed_dict={x: x_train[idx: idx+args.batch_size],
                               y: y_train[idx: idx+args.batch_size],
                               keep_prob: 1.0 - args.dropout_rate_1}
                )
                step = idx / args.batch_size
                if step % 10 == 0:
                    print "epoch %d - step %d - training loss: %g" % (epoch, step, loss)
            print "------------------------------"
            # Evaluate (so no dropout) on training set
            train_acc = accuracy.eval(
                feed_dict={x: x_train, y: y_train, keep_prob: 1.0}
            )
            print "epoch %d: train accuracy: %g" % (epoch, train_acc)
            train_accuracies.append(train_acc)

            # Evaluate (so no dropout) on validation set
            valid_acc = accuracy.eval(
                feed_dict={x: x_valid, y: y_valid, keep_prob: 1.0}
            )
            print "epoch %d: valid accuracy: %g" % (epoch, train_acc)
            valid_accuracies.append(valid_acc)

            # early stop & save
            if valid_acc > best_valid_acc:
                bes_valid_acc = valid_acc # set best acc
                patience = args.patience  # reset patience to initial value bcs score improved
                # TODO: save current model!
            else:
                patience -= 1
            print "epoch %d: patience: %d\n" % (epoch, patience)
            if patience == 0:
                break


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

