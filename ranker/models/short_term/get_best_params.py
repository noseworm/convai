import numpy as np
import cPickle as pkl
import argparse

def main(args):
    for root, dirs, files in os.walk('.'):
        model_timings = [f for f in files if f.endswith('_timings.pkl')]
        model_args = [f for f in files if f.endswith('_args.pkl')]
        
        for model_t, model_a in zip(model_timings, model_args):

            with open('%s/%s' % (root, model_t), 'rb') as handle:
                train_accuracies, valid_accuracies = pkl.load(handle)
            max_trains = [max(train_accuracies[i]) for i in range(len(train_accuracies))]
            max_valids = [max(valid_accuracies[i]) for i in range(len(valid_accuracies))]

            with open("%s/%s" % (root, model_a), 'rb') as handle:
                args = pkl.load(handle)

            print "%s \t avg. train: %g \t avg. valid: %g \t args: %s" % (model_id, np.mean(max_trains), np.mean(max_valids), args[1:])

            # TODO: continue...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("ids", nargs='+', type=str, help="List of model ids to get validation score from timings")
    args = parser.parse_args()
    main(args)

