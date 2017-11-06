import numpy as np
import cPickle as pkl
import argparse

def main(args):
    for model_id in args.ids:
        with open('./%s_VoteEstimator_timings.pkl' % model_id, 'rb') as handle:
            train_accuracies, valid_accuracies = pkl.load(handle)
        max_trains = [max(train_accuracies[i]) for i in range(len(train_accuracies))]
        max_valids = [max(valid_accuracies[i]) for i in range(len(valid_accuracies))]

        with open("./%s_VoteEstimator_args.pkl" % model_id, 'rb') as handle:
            args = pkl.load(handle)

        print "%s \t avg. train: %g \t avg. valid: %g \t args: %s" % (model_id, np.mean(max_trains), np.mean(max_valids), args[1:])

        with open('./%s_valid%g.txt' % (model_id, np.mean(max_valids)), 'w') as handle:
            handle.write("avg. train: %g\n" % np.mean(max_trains))
            handle.write("avg. valid: %g\n" % np.mean(max_valids))
            handle.write("args: %s\n" % args[1:])
            handle.write("features:\n%s\n" % args[0][-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs='+', type=str, help="List of model ids to get validation score from timings")
    args = parser.parse_args()
    main(args)

