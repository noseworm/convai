import tensorflow as tf
import time
import cPickle as pkl


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



class Estimator(object):
    SHORT_TERM = 0
    LONG_TERM = 1

    def __init__(self, mode, data, hidden_dims, hidden_dims_extra, activation, optimizer, learning_rate, model_path='models', model_id=None, model_file='Estimator'):
        """
        Build the estimator for either short term or long term reward based on mode
        :param mode: either SHORT_TERM or LONG_TERM
        :param data: train, valid, test data to use
        :param hidden_dims: list of ints specifying the size of each hidden layer
        :param activation: tensor activation function to use at each layer
        :param optimizer: tensorflow optimizer object to train the network
        :param learning_rate: learning rate for the optimizer
        :param model_path: path of folders where to save the model
        :param model_id: if None, set to creation time
        :param model_file: name for saved model files
        """
        self.mode = mode

        self.trains, self.valids, (self.x_test, self.y_test), self.feature_list = data
        self.n_folds = len(self.trains)
        _, self.input_dim = self.trains[0][0].shape

        self.hidden_dims = hidden_dims
        self.hidden_dims_extra = hidden_dims_extra
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

        self.condition = tf.placeholder(tf.int32, name="short|long_term_condition")

        # Fully connected dense layers
        h_fc = self.x  # (bs, in)
        for idx, hidd in enumerate(self.hidden_dims):
            h_fc = tf.layers.dense(inputs=h_fc,
                                   units=hidd,
                                   # kernel_initializer = Initializer function for the weight matrix.
                                   # bias_initializer: Initializer function for the bias.
                                   activation=ACTIVATIONS[self.activation],
                                   name='shortterm_dense_layer%d' % (idx + 1))  # (bs, hidd)

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # proba of keeping the neuron

        # TODO: use tf.cond() instead!!
        # https://www.tensorflow.org/api_docs/python/tf/cond

        if self.mode == self.SHORT_TERM:
            # Dropout layer
            h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
            # Output layer
            logits = tf.layers.dense(inputs=h_fc_drop,
                                     units=2,
                                     # no activation for the logits
                                     name='shortterm_logits_layer')  # (bs, 2)

            # Define prediction: class label (0,1) and the class probabilities:
            predictions = {
                "classes": tf.argmax(logits, axis=1, name="shortterm_pred_classes"),  # (bs,)
                "probabilities": tf.nn.softmax(logits, name="shortterm_pred_probas")  # (bs, 2)
            }

            # Loss tensor:
            # create one-hot labels
            onehot_labels = tf.one_hot(indices=tf.cast(self.y, tf.int32), depth=2)  # (bs, 2)
            # define the cross-entropy loss
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

            # Train operator:
            optimizer = OPTIMIZERS[self.optimizer](learning_rate=self.lr)
            self.train_step = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

            # Accuracy tensor:
            correct_predictions = tf.equal(
                predictions['classes'], self.y
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, tf.float32)
            )

        elif self.mode == self.LONG_TERM:
            for idx, hidd_extra in enumerate(self.hidden_dims_extra):
                h_fc = tf.layers.dense(inputs=h_fc,
                                       units=hidd_extra,
                                       # kernel_initializer = Initializer function for the weight matrix.
                                       # bias_initializer: Initializer function for the bias.
                                       activation=ACTIVATIONS[self.activation],
                                       name='longterm_dense_layer_%d' % (idx + 1))  # (bs, hidd)
            # Dropout layer
            h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
            # Output layer
            logits = tf.layers.dense(inputs=h_fc_drop,
                                     units=1,
                                     # no activation for the logits
                                     name='longterm_logits_layer')  # (bs, 1)

            # Loss tensor: mean squared error
            self.loss = tf.losses.mean_squared_error(self.y, logits)
            # Train operator:
            optimizer = OPTIMIZERS[self.optimizer](learning_rate=self.lr)
            self.train_step = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

            # Accuracy tensor:
            self.accuracy = 1 - self.loss

        else:
            print "ERROR: unknown estimator mode %s" % self.mode

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

