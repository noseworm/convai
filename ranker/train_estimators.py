import tensorflow as tf
import numpy as np
import cPickle as pkl

import argparse
import pyprind
import random
import copy
import time
import sys
import os

import features


ALL_FEATURES = [
    'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK',
    'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
    'Similarity_CandidateUser',
    'Similarity_CandidateLastK', 'Similarity_CandidateLastK_noStop',
    'Similarity_CandidateKUser', 'Similarity_CandidateKUser_noStop',
    'Similarity_CandidateArticle', 'Similarity_CandidateArticle_noStop',
    'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
    'GenericTurns',
    'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
    'DialogLength', 'LastUserLength', 'CandidateLength', 'ArticleLength',
    'DialogActCandidate', 'DialogActLastUser',
    'SentimentScoreCandidate', 'SentimentScoreLastUser'
]

TARGET_TO_FEATURES = {
    'r': [
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_Article',
        'Similarity_CandidateUser',
        'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
        'GenericTurns',
        'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
        'LastUserLength', 'CandidateLength',
        'DialogActCandidate', 'DialogActLastUser',
        'SentimentScoreCandidate', 'SentimentScoreLastUser'
    ],
    'R': [
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK',
        'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
        'Similarity_CandidateUser',
        'Similarity_CandidateLastK', 'Similarity_CandidateLastK_noStop',
        'Similarity_CandidateKUser', 'Similarity_CandidateKUser_noStop',
        'Similarity_CandidateArticle', 'Similarity_CandidateArticle_noStop',
        'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
        'GenericTurns',
        'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
        'DialogLength', 'LastUserLength', 'CandidateLength', 'ArticleLength',
        'DialogActCandidate', 'DialogActLastUser',
        'SentimentScoreCandidate', 'SentimentScoreLastUser'
    ]
}


ACTIVATIONS = {
    'swish': lambda x: x * tf.sigmoid(x),
    'relu': tf.nn.relu,
    'sigmoid': tf.sigmoid
}


OPTIMIZERS = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer
}


def get_data(files, target, feature_list=None, voted_only=False, val_prop=0.1, test_prop=0.1):
    """
    Load data to train ranker
    :param files: list of data files to load
    :param target: field of each dictionary instance to estimate
        either 'R' for final dialog score, or 'r' for immediate reward
    :param feature_list: list of feature names (str) to load.
        if none will take the ones defined in TARGET_TO_FEATURES.
    :param voted_only: load messages which have been voted only
    :param val_prop: proportion of data to consider for validation set
    :param test_prop: proportion of data to consider for test set
    :return: train x & y, valid x & y, test x & y
        x = numpy array of size (data, feature_length)
          ie: np.array( [[f1, f2, ..., fn], ..., [f1, f2, ..., fn]] )
        y = array of label values to predict
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
        else:
            print "Warning: will not consider file %s because voted_only=%s" % (data_file, voted_only)

    # if didn't get train/valid/test data, build your own
    if len(data) > 4:
        print "got %d examples" % len(data)

        # Build map from article to data_idx to avoid having overlap between train/valid/test sets
        article2id = {}
        for idx, msg in enumerate(data):
            article = unicode(msg['article'])
            if article in article2id:
                article2id[article].append(idx)
            else:
                article2id[article] = [idx]
        print "got %d unique articles" % len(article2id)

        n_train = int(len(data) * (1-val_prop-test_prop))  # size of training data
        n_valid = int(len(data) * val_prop)  # size of valid data
        n_test  = int(len(data) * test_prop) # size of test data
        train_data, valid_data, test_data = [], [], []
        for article, indices in article2id.iteritems():
            # add to training set
            if len(train_data) < n_train:
                train_data.extend([data[idx] for idx in indices])
            # add to validation set when train set is full
            elif len(valid_data) < n_valid:
                valid_data.extend([data[idx] for idx in indices])
            # add to test set when train & valid are full
            else:
                test_data.extend([data[idx] for idx in indices])
        # shuffle data
        random.shuffle(train_data)
        random.shuffle(valid_data)
        random.shuffle(test_data)

        # create list of Feature instances
        if feature_list is None:
            feature_list = TARGET_TO_FEATURES[target]
        feature_objects = features.get(article=None, context=None, candidate=None, feature_list=feature_list)
        input_size = np.sum([f.dim for f in feature_objects])

        # construct data to save & return
        train_x = np.zeros((len(train_data), input_size))
        train_y = []
        valid_x = np.zeros((len(valid_data), input_size))
        valid_y = []
        test_x = np.zeros((len(test_data), input_size))
        test_y = []
        print "train: %s" % (train_x.shape,)
        print "valid: %s" % (valid_x.shape,)
        print "test: %s" % (test_x.shape,)

        print "building data..."
        bar = pyprind.ProgBar(len(data), monitor=False, stream=sys.stdout)  # show a progression bar on the screen
        for (x, y, data) in [(train_x, train_y, train_data), (valid_x, valid_y, valid_data), (test_x, test_y, test_data)]:
            for idx, msg in enumerate(data):
                # create input features for this msg:
                tmp = []
                for f in feature_objects:
                    # Set x for each feature for that msg
                    f.set(msg['article'], msg['context'], msg['candidate'])
                    tmp.extend(copy.deepcopy(f.feat))
                x[idx, :] = np.array(tmp, dtype=np.float32)
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
                [(train_x, train_y), (valid_x, valid_y), (test_x, test_y), feature_list],
                handle,
                pkl.HIGHEST_PROTOCOL
            )
        print "done."

    # got train/valid/test data
    elif len(data) == 4:
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y), feature_list = data
        print "train: %s" % (train_x.shape,)
        print "valid: %s" % (valid_x.shape,)
        print "test: %s" % (test_x.shape,)
        # make sure train, valid, test have the same amount of features.
        assert train_x.shape[1] == valid_x.shape[1] == test_x.shape[1]

    else:
        print "Unknown data format: data length is less than 4."
        return

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y), feature_list


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


class ShortTermEstimator(object):
    def __init__(self, data, hidden_dims, activation, optimizer, model_id=None):
        """
        Build the estimator for short term reward: +1 / -1
        :param data: train, valid, test data to use
        :param hidden_dims: list of ints specifying the size of each hidden layer
        :param activation: tensor activation function to use at each layer
        :param optimizer: tensorflow optimizer object to train the network
        """
        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test), self.feature_list = data
        self.n, self.input_dim = self.x_train.shape

        self.hidden_dims = hidden_dims
        self.activation = activation
        self.optimizer = optimizer

        self.model_path = "models"
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = str(time.time())
        self.model_file = "VoteEstimator"

        self.build()

    def build(self):
        """
        Build the actual neural net, define the predictions, loss, train operator, and accuracy
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="input_layer")  # (bs, feat_size)
        self.y = tf.placeholder(tf.int64, shape=[None, ], name="labels")  # (bs,)

        # Fully connected dense layers
        h_fc = self.x  # (bs, in)
        for idx, hidd in enumerate(self.hidden_dims):
            h_fc = tf.layers.dense(inputs=h_fc,
                                   units=hidd,
                                   # kernel_initializer = Initializer function for the weight matrix.
                                   # bias_initializer: Initializer function for the bias.
                                   activation=ACTIVATIONS[self.activation],
                                   name='dense_layer_%d' % (idx + 1))  # (bs, hidd)
        # Dropout layer
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # proba of keeping the neuron
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
        # Output layer
        self.logits = tf.layers.dense(inputs=h_fc_drop,
                                      units=2,
                                      # no activation for the logits
                                      name='logits_layer')  # (bs, 2)

        # Define prediction: class label (0,1) and the class probabilities:
        self.predictions = {
            "classes": tf.argmax(self.logits, axis=1, name="pred_classes"),  # (bs,)
            "probabilities": tf.nn.softmax(self.logits, name="pred_probas")  # (bs, 2)
        }

        # Loss tensor:
        # create one-hot labels
        onehot_labels = tf.one_hot(indices=tf.cast(self.y, tf.int32), depth=2)  # (bs, 2)
        # define the cross-entropy loss
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.logits)

        # Train operator:
        optimizer = OPTIMIZERS[self.optimizer](learning_rate=0.001)
        self.train_step = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # Accuracy tensor:
        correct_predictions = tf.equal(
            self.predictions['classes'], self.y
        )
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32)
        )
        # self.accuracy, _ = tf.metrics.accuracy(labels=self.y, predictions=self.predictions["classes"], name="accuracy")

        # Once graph is built, create a saver for the model:
        # Add an op to initialize the variables.
        self.init_op = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def train(self, patience, batch_size, dropout_rate):
        self.train_accuracies = []
        self.valid_accuracies = []

        best_valid_acc = 0.0
        p = patience
        with tf.Session() as sess:
            sess.run(self.init_op)  # initialize model variables
            for epoch in range(20000):  # will probably stop before 20k epochs due to early stop
                # do 1 epoch: go through all training_batches
                for idx in range(0, self.n, batch_size):
                    _, loss = sess.run(
                        [self.train_step, self.loss],
                        feed_dict={self.x: self.x_train[idx: idx + batch_size],
                                   self.y: self.y_train[idx: idx + batch_size],
                                   self.keep_prob: 1.0 - dropout_rate}
                    )
                    step = idx / batch_size
                    if step % 10 == 0:
                        print "epoch %d - step %d - training loss: %g" % (epoch, step, loss)
                print "epoch %d - step %d - training loss: %g" % (epoch, step, loss)
                print "------------------------------"
                # Evaluate (so no dropout) on training set
                train_acc = self.evaluate(self.x_train, self.y_train)
                print "epoch %d: train accuracy: %g" % (epoch, train_acc)
                self.train_accuracies.append(train_acc)

                # Evaluate (so no dropout) on validation set
                valid_acc = self.evaluate(self.x_valid, self.y_valid)
                print "epoch %d: valid accuracy: %g" % (epoch, valid_acc)
                self.valid_accuracies.append(valid_acc)

                # early stop & save
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc  # set best acc
                    p = patience  # reset patience to initial value bcs score improved
                    self.save(sess, save_args=False, save_timings=False)
                else:
                    p -= 1
                print "epoch %d: patience: %d\n" % (epoch, p)
                if p == 0:
                    break
            self.save(sess, save_model=False)

    def test(self, x=None, y=None):
        """
        evaluate model on test set
        specify `x` and `y` if we want to evaluate on different set than self.test
        """
        with tf.Session() as sess:
            self.load(sess)
            print "Model restored."
            if x is None or y is None:
                x, y = self.x_test, self.y_test
            test_acc = self.evaluate(x, y)
            print "Test accuracy: %g" % test_acc

    def evaluate(self, x, y):
        # Evaluate accuracy, so no dropout
        acc = self.accuracy.eval(
            feed_dict={self.x: x, self.y: y, self.keep_prob: 1.0}
        )
        return acc

    def save(self, session, save_model=True, save_args=True, save_timings=True):
        prefix = "./%s/%s_%s" % (self.model_path, self.model_id, self.model_file)
        # save the tensorflow graph variables
        if save_model:
            saved_path = self.saver.save(session, "%s_model.ckpt" % prefix)
            print "Model saved in file: %s" % saved_path
        # save the arguments used to create that object
        if save_args:
            data = [
                (self.x_train, self.y_train),
                (self.x_valid, self.y_valid),
                (self.x_test, self.y_test),
                self.feature_list
            ]
            with open("%s_args.pkl" % prefix, 'wb') as handle:
                pkl.dump(
                    [data, self.hidden_dims, self.activation, self.optimizer, self.model_id],
                    handle,
                    pkl.HIGHEST_PROTOCOL
                )
            print "Args saved."
        # Save timings measured during training
        if save_timings:
            with open("%s_timings.pkl" % prefix, 'wb') as handle:
                pkl.dump(
                    [self.train_accuracies, self.valid_accuracies],
                    handle,
                    pkl.HIGHEST_PROTOCOL
                )
            print "Timings saved."

    def load(self, session):
        self.saver.restore(session, "./%s/%s_%s_model.ckpt" % (self.model_path, self.model_id, self.model_file))
        print "Model restored."



def parameter_search():
    hidd_sizes = [
        [1000, 100, 10],
        # TODO: continue...
        [1000, 500, 300, 100, 10]
    ]
    activations = ['swish', 'relu', 'sigmoid']
    optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rate = [0.1, 0.3, 0.5, 0.7, 0.9]


def main(args):
    # Load datasets
    data = get_data(args.data, 'r', voted_only=True)
    train, valid, test, feature_list = data

    print "\nBuilding the network..."
    estimator1 = ShortTermEstimator(
        data,
        args.hidden_sizes_1,
        args.activation_1,
        args.optimizer_1
    )

    print "\nTraining the network..."
    estimator1.train(
        args.patience,
        args.batch_size,
        args.dropout_rate_1
    )
    print "done."



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
    parser.add_argument("-op1","--optimizer_1", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="Optimizer to train the first network")
    parser.add_argument("-op2","--optimizer_2", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="Optimizer to train the second network")
    args = parser.parse_args()
    print "\n", args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    main(args)

