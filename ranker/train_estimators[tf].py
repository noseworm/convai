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


FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

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


def get_data(target, voted_only=False, val_prop=0.1, test_prop=0.1):
    """
    Load data to train ranker
    :param target: field of each dictionary instance to estimate
        either 'R' for final dialog score, or 'r' for immediate reward
    :param voted_only: load messages which have been voted only
    :param val_prop: proportion of data to consider for validation set
    :param test_prop: proportion of data to consider for test set
    :return: input size (int), train x & y, test x & y
        x = map from feature_name to numpy array of size (data, feature_length)
          ie: { <name of feature 1>: np.array( [[f1_dim],...,[f1_dim]] ),
                <name of feature 2>: np.array( [[f2_dim],...,[f2_dim]] ),
                ... }
        y = array of label values to predict
        input_size = feat_1.dim + ... + feat_n.dim
    """
    assert target in ['R', 'r'], "Unknown target: %s" % target

    print "\nLoading data..."
    data = []
    file_ids = []
    for data_file in FLAGS.data:
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
        train_x = dict(
            [(f.__class__.__name__, np.zeros((len(train_data), f.dim))) for f in feature_objects]
        )
        train_y = []
        valid_x = dict(
            [(f.__class__.__name__, np.zeros((len(valid_data), f.dim))) for f in feature_objects]
        )
        valid_y = []
        test_x = dict(
            [(f.__class__.__name__, np.zeros((len(test_data), f.dim))) for f in feature_objects]
        )
        test_y = []

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
    :param mode: indicating the context in which the model_fn was invoked:
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
    for p in ['input_size', 'hidden_sizes', 'activation', 'drop_rate', 'optimizer']:
        assert p in params

    # print "input size: %d" % params['input_size']
    input_layer = tf.concat(
        features.values(),  # (bs, f_dim)
        axis = 1,
        name = "input_layer"
    )
    # print "input_layer: %s" % input_layer

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
        "classes"       : tf.argmax(logits, axis=1, name="pred_classes_tensor"),  # (bs,)
        "probabilities" : tf.nn.softmax(logits, name="pred_probas_tensor")  # (bs, 2)
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
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name="accuracy_tensor")
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metric_ops)


def model2_fn():
    pass


def build_input_function(x, y, mode='train'):
    """
    Define input pipeline
    :param x: map from 'feature_name' to numpy array ~ (data, feat_dim)
    :param y: list of labels (0 for downvote / 1 for upvote)
    :param mode: `train` or `evaluate` to decide number of epochs & batch size
    :return: input function that would feed dict of numpy arrays into the model
    """
    assert mode in ['train', 'eval']
    if mode == 'train':
        return tf.estimator.inputs.numpy_input_fn(
            x          = x,
            y          = y,
            batch_size = FLAGS.batch_size,  # number of examples per training step
            num_epochs = None,              # train until specified number of train steps is reached
            shuffle    = False              # iterate through the data sequentially
        )
    elif mode == 'eval':
        return tf.estimator.inputs.numpy_input_fn(
            x          = x,
            y          = y,
            # ignore batch size, look at epoch for evaluation
            num_epochs = 1,     # evaluates the metrics over one epoch of data only
            shuffle    = False  # iterate through the data sequentially
        )
    else:
        print "Unknown input function mode: %s" % mode
        return None


def main(unused_argv):
    """
    tuto: https://www.tensorflow.org/tutorials/layers#evaluate_the_model
    """
    # Load datasets
    input_size, (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data('r', voted_only=True)

    # parse arguments
    if FLAGS.activation_1 == 'swish':
        activation_1 = lambda x: x*tf.sigmoid(x)
    elif FLAGS.activation_1 == 'relu':
        activation_1 = tf.nn.relu
    elif FLAGS.activation_1 == 'sigmoid':
        activation_1 = tf.sigmoid
    else:
        print "ERROR: unknown activation 1: %s" % FLAGS.activation_1
        return

    model1_dir = "./models/vote_estimator_model"
    steps = len(y_train) / FLAGS.batch_size
    steps *= 2

    # model1_config = tf.estimator.RunConfig()
    # model1_config = model1_config.replace(
    #     save_checkpoints_steps = steps,
    #     save_summary_steps     = steps
    # )  # save at every epoch

    # Build first RNN to predict immediate reward: +1 / -1
    nn1 = tf.estimator.Estimator(
        model_fn  = model1_fn,
        params    = {
            'input_size'  : input_size,
            'hidden_sizes': FLAGS.hidden_sizes_1,
            'activation'  : activation_1,  # try tf.nn.swish | tf.nn.relu | tf.sigmoid
            'drop_rate'   : FLAGS.dropout_rate_1,  # try 0.1 | 0.3 | 0.5 | 0.7 | 0.9
            'optimizer'   : tf.train.AdamOptimizer()
        },
        model_dir = model1_dir
        # config    = model1_config
    )

    # experiment = tf.contrib.learn.Experiment(
    #     estimator=nn1,
    #     train_input_fn=build_input_function(x_train, y_train, mode='train'),
    #     eval_input_fn=build_input_function(x_valid, y_valid, mode='eval'),
    #     train_steps=None,  # train until no more improvement on validation set
    #     eval_steps=1,  # evaluate one step only (one epoch)
    #     eval_delay_secs=60,  # start evaluating after k sec
    #     train_steps_per_iteration=steps  # do k train steps per iteration
    # )

    # Train - Validation loop
    model1_train_accuracies = []
    model1_valid_accuracies = []
    best_valid_acc = 0.0
    patience = FLAGS.patience
    while patience > 0:
        print "\nTrain for 1 epoch (% steps)" % steps

        # def continue_training(eval_results):
        #     # None argument for the first evaluation
        #     if not eval_results:
        #         return True
        #
        #     valid_acc = eval_results['accuracy']
        #     model1_valid_accuracies.append(valid_acc)
        #
        #     if len(model1_valid_accuracies) < FLAGS.patience:
        #         return True
        #
        #     # last k accuracies
        #     last_accs = model1_valid_accuracies[-(FLAGS.patience+1): -1]
        #     # make sure they are all above current acc to stop
        #     for acc in last_accs:
        #         if acc < valid_acc:
        #             return True
        #
        #     return False

        # evaluate_results = experiment.continuous_train_and_eval(
        #     continuous_eval_predicate_fn=continue_training
        # )
        # print "\nEvaluation results:\n%s" % evaluate_results
        # print "\nValidation accuracies:\n%s" % model1_valid_accuracies

        # Train on training set
        nn1.train(
            input_fn = build_input_function(x_train, y_train, mode='train'),
            steps    = steps  # train for 1 epoch
        )
        # Evaluate on train & validation set
        train_results = nn1.evaluate(input_fn=build_input_function(x_train, y_train, mode='eval'))
        valid_results = nn1.evaluate(input_fn=build_input_function(x_valid, y_valid, mode='eval'))
        train_acc = train_results['accuracy']
        valid_acc = valid_results['accuracy']
        print "\ntrain accuracy: %s" % train_acc
        print "valid accuracy: %s" % valid_acc

        model1_train_accuracies.append(train_acc)
        model1_valid_accuracies.append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
        else:
            patience -= 1
        print "patience: %d\n" % patience

    # Evaluate the model on test data
    test_results = nn1.evaluate(input_fn=build_input_function(x_test, y_test, mode='eval'))
    print "\ntest results:\n%s" % test_results

    # Confusion Matrix
    test_predictions = nn1.predict(input_fn=build_input_function(x_test, y_test, mode='eval'))
    print "\ntest predictions:\n%s" % test_predictions
    conf_mat = tf.confusion_matrix(
        labels      = y_test,
        predictions = test_predictions['prediction_classes'],
        num_classes = 2,
        name        = "confusion_matrix"
    )
    print "\nConfusion matrix:\n%s" % conf_mat

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
    parser.add_argument("data", nargs='+', type=str, help="List of files to consider for training")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    # training parameters:
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during training")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Number of training steps to wait before stoping when validatiaon accuracy doesn't increase")
    # parser.add_argument("-ts", "--train_steps", type=int, default=10, help="number of batches seen during 1 training step")
    # network architecture:
    parser.add_argument("-h1", "--hidden_sizes_1", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for first network")
    parser.add_argument("-h2", "--hidden_sizes_2", nargs='+', type=int, default=[500, 300, 100, 10], help="List of hidden sizes for second network")
    parser.add_argument("-a1", "--activation_1", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for first network")
    parser.add_argument("-a2", "--activation_2", choices=['sigmoid', 'relu', 'swish'], type=str, default='relu', help="Activation function for second network")
    parser.add_argument("-d1", "--dropout_rate_1", type=float, default=0.1, help="Probability of dropout layer in first network")
    parser.add_argument("-d2", "--dropout_rate_2", type=float, default=0.1, help="Probability of dropout layer in second network")
    FLAGS, unparsed = parser.parse_known_args()
    print '\n',FLAGS

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % FLAGS.gpu
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

