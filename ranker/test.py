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

from estimators import Estimator, ACTIVATIONS, OPTIMIZERS, SHORT_TERM_MODE, LONG_TERM_MODE


def main(args):
    # Load previously saved model arguments
    print "Loading previous model arguments..."
    with open("%sargs.pkl" % args.model, 'rb') as handle:
        model_args = pkl.load(handle)

    if len(model_args) == 12:
        data, \
        hidden_dims, hidden_dims_extra, activation, \
        optimizer, learning_rate, \
        model_path, model_id, model_name, \
        batch_size, dropout_rate, pretrained = model_args
    elif len(model_args) == 8:
        data, \
        hidden_dims, activation, \
        optimizer, learning_rate, \
        model_id, \
        batch_size, dropout_rate = model_args
        # reconstruct missing parameters
        hidden_dims_extra = [hidden_dims[-1]]
        model_name = args.model.split(model_id)[1].replace('_', '')
        pretrained = None
    else:
        print "WARNING: %d model arguments." % len(model_args)
        print "data + %s" % (model_args[1:],)
        return

    # reconstruct model_path just in case it has been moved:
    model_path = args.model.split(model_id)[0]
    if model_path.endswith('/'):
        model_path = model_path[:-1]  # ignore the last '/'

    # Load previously saved model timings
    with open("%stimings.pkl" % args.model, 'rb') as handle:
        train_accuracies, valid_accuracies = pkl.load(handle)

    n_folds = len(train_accuracies)
    max_train = [max(train_accuracies[i]) for i in range(n_folds)]
    max_valid = [max(valid_accuracies[i]) for i in range(n_folds)]
    print "prev. max train accuracies: %s" % (max_train,)
    print "prev. max valid accuracies: %s" % (max_valid,)
    train_acc = np.mean(max_train)
    valid_acc = np.mean(max_valid)
    print "prev. best avg. train accuracy: %g" % train_acc
    print "prev. best avg. valid accuracy: %g" % valid_acc

    print "Building the network..."
    estimator = Estimator(
        data,
        hidden_dims, hidden_dims_extra, activation,
        optimizer, learning_rate,
        model_path, model_id, model_name
    )

    with tf.Session() as sess:
        print "Reset network parameters..."
        estimator.load(sess, model_path, model_id, model_name)
        print "Testing the network..."
        test_acc = estimator.test(SHORT_TERM_MODE)
        print "test accuracy: %g" % test_acc

        # print "\nContinue training the network..."
        # estimator.train(
        #     sess,
        #     SHORT_TERM_MODE,
        #     args.patience,
        #     batch_size,
        #     dropout_rate,
        #     save=False,
        #     pretrained=(model_path, model_id, model_name),
        #     previous_accuracies=(train_accuracies, valid_accuracies)
        # )
        # n_folds = len(estimator.train_accuracies)  # number of folds x2 now that we retrained
        # max_train = [max(estimator.train_accuracies[i]) for i in range(n_folds)]
        # max_valid = [max(estimator.valid_accuracies[i]) for i in range(n_folds)]
        # print "max train accuracies: %s" % (max_train,)
        # print "max valid accuracies: %s" % (max_valid,)
        # train_acc = np.mean(max_train)
        # valid_acc = np.mean(max_valid)
        # print "best avg. train accuracy: %g" % train_acc
        # print "best avg. valid accuracy: %g" % valid_acc
        # print "Re-testing the network..."
        # test_acc = estimator.test(SHORT_TERM_MODE)
        # print "test accuracy: %g" % test_acc

        print "Get train, valid, test prediction on fold %d..."
        trains, valids, (x_test, y_test), feature_list = data
        train_acc, valid_acc = [], []
        for fold in range(len(trains)):
            train_predictions = estimator.predict(SHORT_TERM_MODE, trains[fold][0])
            same = float(np.sum(train_predictions == trains[fold][1]))
            acc  = same/len(train_predictions)
            train_acc.append(acc)
            print "[fold %d] train acc: %d/%d=%g" % (fold+1, same, len(train_predictions), acc)

            valid_predictions = estimator.predict(SHORT_TERM_MODE, valids[fold][0])
            same = float(np.sum(valid_predictions == valids[fold][1]))
            acc  = same/len(valid_predictions)
            valid_acc.append(acc)
            print "[fold %d] valid acc: %d/%d=%g" % (fold+1, same, len(valid_predictions), acc)

        print "avg. train acc. %g" % np.mean(train_acc)
        print "avg. valid acc. %g" % np.mean(valid_acc)
        test_predictions  = estimator.predict(SHORT_TERM_MODE, x_test)
        same = float(np.sum(test_predictions == y_test))
        acc = same/len(test_predictions)
        print "test acc: %d/%d=%g" % (same, len(test_predictions), acc)

    print "done."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model prefix to load")
    parser.add_argument("-g",  "--gpu", type=int, default=0, help="GPU number to use")
    parser.add_argument("-p",  "--patience", type=int, default=5, help="Number of training steps to wait before stoping when validatiaon accuracy doesn't increase")
    args = parser.parse_args()
    print "\n%s\n" % args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

    main(args)

