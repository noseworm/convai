import tensorflow as tf
import numpy as np
import cPickle as pkl
import argparse
import pyprind
import copy
import json
import time
import sys
import os

import features as _features
import inspect

ALL_FEATURES = []
for name, obj in inspect.getmembers(_features):
    if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
        ALL_FEATURES.append(name)

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
    Load data to train ranker. Build `k` fold cross validation train/val data
    :param files: list of data files to load
    :param target: field of each dictionary instance to estimate
        either 'R' for final dialog score, or 'r' for immediate reward
    :param feature_list: list of feature names (str) to load.
        if none will take the ones defined in TARGET_TO_FEATURES.
    :param voted_only: load messages which have been voted only
    :param val_prop: proportion of data to consider for validation set.
        Will also define the number of train/valid folds
    :param test_prop: proportion of data to consider for test set
    :return: collections of train x & y, valid x & y, and one test x & y
        x = numpy array of size (data, feature_length)
          ie: np.array( [[f1, f2, ..., fn], ..., [f1, f2, ..., fn]] )
        y = array of label values to predict
    """
    assert target in ['R', 'r'], "Unknown target: %s" % target

    print "\nLoading data..."
    raw_data = {}  # map file name to list of dictionaries
    ''' FORMAT:
    {
       file_name : [ {-}, {-}, ..., {-} ],
       ...
    }
    with {-} being the dictionary object containing context, article, etc...
    '''
    n = 0  # total number of examples
    for data_file in files:
        if voted_only and 'data/voted_data_' in data_file:
            with open(data_file, 'rb') as handle:
                raw_data[data_file] = json.load(handle)
                n += len(raw_data[data_file])
                # get the time id of the data
                # file_ids.append(data_file.split('_')[-1].replace('pkl', ''))
        elif (not voted_only) and 'data/full_data_' in data_file:
            with open(data_file, 'rb') as handle:
                raw_data[data_file] = json.load(handle)
                n += len(raw_data[data_file])
                # get the time id of the data
                # file_ids.append(data_file.split('_')[-1].replace('pkl', ''))
        else:
            print "Warning: will not consider file %s because voted_only=%s" % (data_file, voted_only)
    print "got %d examples" % n

    # Build map from article to filename to data_idx to avoid having overlap between train/valid/test sets
    article2file2id = {}
    ''' FORMAT: {
        article : { file_name: [idx, idx, ..., idx],
                    ...
                  },
        ...
    } '''
    for data_file, data in raw_data.iteritems():
        for idx, msg in enumerate(data):
            article = msg['article']
            if article not in article2file2id:
                article2file2id[article] = {}

            if data_file not in article2file2id[article]:
                article2file2id[article][data_file] = [idx]
            else:
                article2file2id[article][data_file].append(idx)
    print "got %d unique articles" % len(article2file2id)

    train_max_n = int(n * (1-val_prop-test_prop))  # size of training data
    valid_max_n = int(n * val_prop)  # size of valid data
    test_max_n  = int(n * test_prop) # size of test data

    test_data, remain_data = {}, {}  # store map from filename to list of indices
    test_n, remain_n = 0, 0          # number of examples
    for article, file2id in article2file2id.iteritems():
        # add to test set
        if test_n < test_max_n:
            for data_file, indices in file2id.iteritems():
                if data_file not in test_data:
                    test_data[data_file] = []
                test_data[data_file].extend(indices)
                test_n += len(indices)
        # keep the remaining for train & valid k-fold
        else:
            for data_file, indices in file2id.iteritems():
                if data_file not in remain_data:
                    remain_data[data_file] = []
                remain_data[data_file].extend(indices)
                remain_n += len(indices)

    # create list of Feature instances
    if feature_list is None:
        feature_list = TARGET_TO_FEATURES[target]
    feature_objects = _features.get(article=None, context=None, candidate=None, feature_list=feature_list)
    input_size = np.sum([f.dim for f in feature_objects])
    del feature_objects  # now that we have the input_size, don't need those anymore

    # construct data to save & return
    remain_x = []  # np.zeros((remain_n, input_size))
    remain_y = []
    test_x = []    # np.zeros((test_n, input_size))
    test_y = []

    print "building data..."
    for x, y, data in [(remain_x, remain_y, remain_data), (test_x, test_y, test_data)]:
        for data_file, indices in data.iteritems():
            # load the required features for that file
            features = {}
            feature_path = data_file.replace('.json', '.features')
            for feat in feature_list:
                with open("%s/%s.json" % (feature_path, feat), 'rb') as handle:
                    features[feat] = json.load(handle)
            for idx in indices:
                msg = raw_data[data_file][idx]
                # create input features for this msg:
                tmp = np.concatenate( [features[feat][idx] for feat in feature_list] )
                x.append(tmp.tolist())
                # set y labels
                if target == 'r':
                    if int(msg[target]) == -1: y.append(0)
                    elif int(msg[target]) == 1: y.append(1)
                    else: print "ERROR: unknown immediate reward value: %s" % msg[target]
                else:
                    y.append(msg[target])

    remain_x = np.array(remain_x)
    assert remain_x.shape == (remain_n, input_size), "%s != %s" % (remain_x.shape, (remain_n, input_size))
    remain_y = np.array(remain_y)
    assert len(remain_y) == remain_n, "%d != %d" % (len(remain_y), remain_n)
    test_x = np.array(test_x)
    assert test_x.shape == (test_n, input_size), "%s != %s" % (test_x.shape, (test_n, input_size))
    test_y = np.array(test_y)
    assert len(test_y) == test_n, "%d != %d" % (len(test_y), test_n)

    # reformat train & valid to build k-folds:
    trains = []
    valids = []
    n = len(remain_y)
    for k_fold in range(n / valid_max_n):
        start = valid_max_n * k_fold  # start index of validation set
        stop =  valid_max_n * (k_fold+1)  # stop index of validation set
        # define validation set for this fold
        tmp_valid_x = remain_x[start: stop]
        tmp_valid_y = remain_y[start: stop]
        valids.append((tmp_valid_x, tmp_valid_y))
        print "[fold %d] valid: %s" % (k_fold+1, tmp_valid_x.shape)
        # define training set for this fold
        # print "[fold %d] train indices: %s" % (k+fold+1, np.array(range(stop, n+start)) % n)
        tmp_train_x = remain_x[np.array(range(stop, n+start)) % n]
        tmp_train_y = remain_y[np.array(range(stop, n+start)) % n]
        trains.append((tmp_train_x, tmp_train_y))
        print "[fold %d] train: %s" % (k_fold+1, tmp_train_x.shape)
    print "test: %s" % (test_x.shape,)

    return trains, valids, (test_x, test_y), feature_list


class ShortTermEstimator(object):
    def __init__(self, data, hidden_dims, activation, optimizer, learning_rate, model_path='models', model_id=None, model_file='VoteEstimator'):
        """
        Build the estimator for short term reward: +1 / -1
        :param data: train, valid, test data to use
        :param hidden_dims: list of ints specifying the size of each hidden layer
        :param activation: tensor activation function to use at each layer
        :param optimizer: tensorflow optimizer object to train the network
        :param learning_rate: learning rate for the optimizer
        :param model_path: path of folders where to save the model
        :param model_id: if None, set to creation time
        :param model_file: name for saved model files
        """
        self.trains, self.valids, (self.x_test, self.y_test), self.feature_list = data
        self.n_folds = len(self.trains)
        _, self.input_dim = self.trains[0][0].shape

        self.hidden_dims = hidden_dims
        self.activation = activation
        self.optimizer = optimizer
        self.lr = learning_rate

        self.model_path = model_path
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = str(time.time())
        self.model_file = model_file

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
        optimizer = OPTIMIZERS[self.optimizer](learning_rate=self.lr)
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

    def train(self, session, patience, batch_size, dropout_rate, save=True):
        """
        :param session: tensorflow session
        :param patience: number of times to continue training when no improvement on validation
        :param batch_size: number of examples per batch
        :param dropout_rate: probability of drop out
        :param save: decide if we save the model & its parameters
        """
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.train_accuracies = []
        self.valid_accuracies = []

        # Perform k-fold cross validation: train/valid on k different part of the data
        fold = 0
        for (x_train, y_train), (x_valid, y_valid) in zip(self.trains, self.valids):
            fold += 1
            train_accs = []  # accuracies for this fold, to be added at the end of the fold
            valid_accs = []  # accuracies for this fold, to be added at the end of the fold
            n, _ = x_train.shape

            best_valid_acc = 0.0
            p = patience
            session.run(self.init_op)  # initialize model variables
            for epoch in range(20000):  # will probably stop before 20k epochs due to early stop
                # do 1 epoch: go through all training_batches
                for idx in range(0, n, batch_size):
                    _, loss = session.run(
                        [self.train_step, self.loss],
                        feed_dict={self.x: x_train[idx: idx + batch_size],
                                   self.y: y_train[idx: idx + batch_size],
                                   self.keep_prob: 1.0 - dropout_rate}
                    )
                    step = idx / batch_size
                    # if step % 10 == 0:
                    #     print "[fold %d] epoch %d - step %d - training loss: %g" % (fold, epoch+1, step, loss)
                # print "[fold %d] epoch %d - step %d - training loss: %g" % (fold, epoch+1, step, loss)
                # print "------------------------------"
                # Evaluate (so no dropout) on training set
                train_acc = self.evaluate(x_train, y_train)
                # print "[fold %d] epoch %d: train accuracy: %g" % (fold, epoch+1, train_acc)
                train_accs.append(train_acc)

                # Evaluate (so no dropout) on validation set
                valid_acc = self.evaluate(x_valid, y_valid)
                # print "[fold %d] epoch %d: valid accuracy: %g" % (fold, epoch+1, valid_acc)
                valid_accs.append(valid_acc)

                # early stop & save
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc  # set best acc
                    p = patience  # reset patience to initial value bcs score improved
                    if save:
                        self.save(session, save_args=False, save_timings=False)
                else:
                    p -= 1
                # print "[fold %d] epoch %d: patience: %d" % (fold, epoch+1, p)
                if p == 0:
                    break
            if save:
                # save the arguments and the timings when done
                self.save(session, save_model=False)
            # print "------------------------------"
            self.train_accuracies.append(train_accs)
            self.valid_accuracies.append(valid_accs)

    def test(self, session, x=None, y=None):
        """
        evaluate model on test set
        specify `x` and `y` if we want to evaluate on different set than self.test
        """
        self.load(session)
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
                self.trains,
                self.valids,
                (self.x_test, self.y_test),
                self.feature_list
            ]
            with open("%s_args.pkl" % prefix, 'wb') as handle:
                pkl.dump(
                    [data, self.hidden_dims, self.activation, self.optimizer, self.lr, self.model_id,
                     self.batch_size, self.dropout_rate],
                    handle,
                    pkl.HIGHEST_PROTOCOL
                )
            print "Args (and data) saved."
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



def sample_parameters(t):
    """
    randomly choose a set of parameters t times
    """
    default_features = ['AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_Article']
    features = filter(lambda f : f not in default_features, ALL_FEATURES)
    # [
    #     'AverageWordEmbedding_LastK', 'AverageWordEmbedding_kUser',
    #     'Similarity_CandidateUser',
    #     'Similarity_CandidateLastK', 'Similarity_CandidateLastK_noStop',
    #     'Similarity_CandidateKUser', 'Similarity_CandidateKUser_noStop',
    #     'Similarity_CandidateArticle', 'Similarity_CandidateArticle_noStop',
    #     'NonStopWordOverlap', 'BigramOverlap', 'TrigramOverlap', 'EntityOverlap',
    #     'GenericTurns',
    #     'WhWords', 'IntensifierWords', 'ConfusionWords', 'ProfanityWords', 'Negation',
    #     'DialogLength', 'LastUserLength', 'CandidateLength', 'ArticleLength',
    #     'DialogActCandidate', 'DialogActLastUser',
    #     'SentimentScoreCandidate', 'SentimentScoreLastUser'
    # ]
    # map from hidden sizes to used_before flag
    hidd_sizes = dict(
        [(k, False) for k in [
            (900, 300), (700, 300), (700, 100), (500, 50), (500, 100), (300, 50),
            (900, 500, 200), (900, 300, 50), (700, 500, 50), (700, 100, 50),
            (500, 700, 300), (500, 400, 100), (500, 100, 50), (300, 500, 100),
            (900, 500, 500, 100), (900, 600, 300, 100), (700, 300, 300, 100), (700, 500, 300, 100), (500, 200, 200, 50),
            (800, 500, 300, 100, 50), (900, 600, 300, 100, 50), (800, 300, 300, 300, 50)
        ]]
    )
    activations = ['swish', 'relu', 'sigmoid']
    optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta']
    learning_rates = [0.01, 0.001, 0.0001]
    dropout_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    batch_sizes = [32, 64, 128, 256, 512, 1024]

    feats, hidds, activs, optims, lrs, drs, bss = [], [], [], [], [], [], []
    # sample parameters
    for _ in range(t):
        # force 3*300 features for input of size > 900
        sampled_features = copy.deepcopy(default_features)
        n = np.random.randint(len(features)/2, len(features)+1)  # number of features to sample: between half and all
        sampled_features.extend(np.random.choice(features, n, replace=False))

        new_sizes = [h for h in hidd_sizes.keys() if not hidd_sizes[h]]
        if len(new_sizes) == 0:
            # reset all `used_before` flags to False
            for h in hidd_sizes.keys(): hidd_sizes[h] = False
            new_sizes = [h for h in hidd_sizes.keys() if not hidd_sizes[h]]
            assert len(new_sizes) == len(hidd_sizes)
        idx = np.random.choice(len(new_sizes))
        sampled_hidd = new_sizes[idx]
        hidd_sizes[sampled_hidd] = True  # flag this size to be sampled

        feats.append(sampled_features)
        hidds.append(sampled_hidd)
        activs.append(np.random.choice(activations))
        optims.append(np.random.choice(optimizers))
        lrs.append(np.random.choice(learning_rates))
        drs.append(np.random.choice(dropout_rates))
        bss.append(np.random.choice(batch_sizes))

    return feats, hidds, activs, optims, lrs, drs, bss


def main(args):
    if args.explore:
        # sample a bunch of parameters, and run those experiments
        feats, hidds, activs, optims, lrs, drs, bss = sample_parameters(args.explore)
        best_args = []  # store the best combination
        best_valid_acc = 0.0  # store the best validation accuracy
        best_model = None  # store the best model id
        valid_threshold = 0.63  # accuracy must be higher than 62% to be saved
        print "Will try %d different configurations..." % args.explore
        for idx in range(args.explore):
            with tf.Session() as sess:
                try:
                    # print sampled parameters
                    print "\n[%d] sampled features:\n%s" % (idx+1, feats[idx])
                    print "[%d] sampled hidden_sizes: %s" % (idx+1, hidds[idx])
                    print "[%d] sampled activation: %s" % (idx+1, activs[idx])
                    print "[%d] sampled optimizer: %s" % (idx+1, optims[idx])
                    print "[%d] sampled learning rate: %g" % (idx+1, lrs[idx])
                    print "[%d] sampled dropout rate: %g" % (idx+1, drs[idx])
                    print "[%d] sampled batch size: %d" % (idx+1, bss[idx])

                    # Load datasets
                    data = get_data(args.data, 'r', feature_list=feats[idx], voted_only=True)
                    trains = data[0]
                    print "[%d] Building the network..." % (idx+1,)
                    estimator1 = ShortTermEstimator(
                        data,
                        hidds[idx],
                        activs[idx],
                        optims[idx],
                        lrs[idx]
                    )
                    print "[%d] Training the network..." % (idx+1,)
                    estimator1.train(
                        sess,
                        args.patience,
                        bss[idx],
                        drs[idx],
                        save=False  # don't save for now
                    )
                    max_train = [max(estimator1.train_accuracies[i]) for i in range(len(trains))]
                    max_valid = [max(estimator1.valid_accuracies[i]) for i in range(len(trains))]
                    print "[%d] max train accuracies: %s" % (idx+1, max_train)
                    print "[%d] max valid accuracies: %s" % (idx+1, max_valid)
                    train_acc = np.mean(max_train)
                    valid_acc = np.mean(max_valid)
                    print "[%d] best avg. train accuracy: %g" % (idx+1, train_acc)
                    print "[%d] best avg. valid accuracy: %g" % (idx+1, valid_acc)

                    # save now if we got a good model
                    if valid_acc > valid_threshold:
                        estimator1.save(sess)

                    # update variables if we got better model
                    if valid_acc > best_valid_acc:
                        print "[%d] got better accuracy! new: %g > old: %g" % (idx+1, valid_acc, best_valid_acc)
                        best_valid_acc = valid_acc
                        best_model = estimator1.model_id
                        best_args = [feats[idx], hidds[idx], activs[idx], optims[idx], lrs[idx], drs[idx], bss[idx]]
                    else:
                        print "[%d] best validation accuracy is still %g" % (idx+1, best_valid_acc)

                # end of try block, catch CTRL+C errors to print current results
                except KeyboardInterrupt as e:
                    print e
                    print "best model: %s" % best_model
                    print "with parameters:"
                    print " - features:\n%s"     % (best_args[0],)
                    print " - hidden_sizes: %s"  % (best_args[1],)
                    print " - activation: %s"    % (best_args[2],)
                    print " - optimizer: %s"     % (best_args[3],)
                    print " - learning rate: %g" % (best_args[4],)
                    print " - dropout rate: %g"  % (best_args[5],)
                    print " - batch size: %d"    % (best_args[6],)
                    print "with average valid accuracy: %g" % best_valid_acc
                    sys.exit()

            # end of tensorflow session, reset for the next graph
            tf.reset_default_graph()

        # end of exploration, print best results:
        print "done!"
        print "best model: %s" % best_model
        print "with parameters:"
        print " - features:\n%s"     % best_args[0]
        print " - hidden_sizes: %s"  % best_args[1]
        print " - activation: %s"    % best_args[2]
        print " - optimizer: %s"     % best_args[3]
        print " - learning rate: %g" % best_args[4]
        print " - dropout rate: %g"  % best_args[5]
        print " - batch size: %d"    % best_args[6]
        print "with average valid accuracy: %g" % best_valid_acc

    else:
        # run one experiment with provided parameters
        # Load datasets
        data = get_data(args.data, 'r', feature_list=ALL_FEATURES, voted_only=True)
        trains = data[0]

        print "\nBuilding the network..."
        estimator1 = ShortTermEstimator(
            data,
            args.hidden_sizes_1,
            args.activation_1,
            args.optimizer_1,
            args.learning_rate_1
        )
        with tf.Session() as sess:
            print "\nTraining the network..."
            estimator1.train(
                sess,
                args.patience,
                args.batch_size,
                args.dropout_rate_1,
                save=True
            )
            max_train = [max(estimator1.train_accuracies[i]) for i in range(len(trains))]
            max_valid = [max(estimator1.valid_accuracies[i]) for i in range(len(trains))]
            print "max train accuracies: %s" % (max_train,)
            print "max valid accuracies: %s" % (max_valid,)
            train_acc = np.mean(max_train)
            valid_acc = np.mean(max_valid)
            print "best avg. train accuracy: %g" % train_acc
            print "best avg. valid accuracy: %g" % valid_acc
        print "done."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs='+', type=str, help="List of files to consider for training")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("-ex", "--explore", type=int, default=None, help="Number of times to sample parameters. If None, will use the one provided")
    # training parameters:
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during training")
    parser.add_argument("-p",  "--patience", type=int, default=20, help="Number of training steps to wait before stoping when validatiaon accuracy doesn't increase")
    # network architecture:
    parser.add_argument("-h1", "--hidden_sizes_1", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for first network")
    parser.add_argument("-h2", "--hidden_sizes_2", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for second network")
    parser.add_argument("-a1", "--activation_1", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for first network")
    parser.add_argument("-a2", "--activation_2", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for second network")
    parser.add_argument("-d1", "--dropout_rate_1", type=float, default=0.1, help="Probability of dropout layer in first network")
    parser.add_argument("-d2", "--dropout_rate_2", type=float, default=0.1, help="Probability of dropout layer in second network")
    parser.add_argument("-op1","--optimizer_1", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="Optimizer to train the first network")
    parser.add_argument("-op2","--optimizer_2", choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'], default='adam', help="Optimizer to train the second network")
    parser.add_argument("-lr1","--learning_rate_1", type=float, default=0.001, help="Learning rate for the first network")
    parser.add_argument("-lr2","--learning_rate_2", type=float, default=0.001, help="Learning rate for the second network")
    args = parser.parse_args()
    print "\n%s\n" % args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    main(args)

