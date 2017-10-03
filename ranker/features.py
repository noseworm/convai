import numpy as np

"""
This file defines different hand-enginered features based
on the conversation article, the conversation hisory, and
one candidate response.
These features will then be used as input signals to a
fully-connected feed-foward neural network to predict either
the full dialogue score or the utterance score.
"""

def get(article, context, candidate, feature_name):
    """
    The only method to call to get all features we want for a triple
    of (article, context, candidate response).
    :param article: the text of the conversation article to talk to
    :param context: the list of user & bot utterances so far
    :param candidate: the candidate response one model proposed
    :feature_name: string of features to return for the above triple.
        Note: all feature names should be between brackets ('<' and '>').
    :return: a numpy aray containing all features you requested for.
    """
    feature_name = feature_name.lower()

    features = np.array([])

    if '<candidate_length>' in feature_name or '<all>' in feature_name:
        features.append(_candidate_length(candidate))
    elif '<user_length>' in feature_name or '<all>' in feature_name:
        features.append(_user_length(context))
    # TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
    else:
        print "WARNING: no feature recognized in %s" % feature_name
    
    return features


def _candidate_length(candidate):
    """
    :return: float scalar (dim: 1) for the length of the candidate response
    """
    return float(len(candidate))

def _user_length(context):
    """
    :return: float scalar (dim: 1) for the length of the previous user utterance
    """
    return float(len(context[-1]))


# TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
# Make sure to explain new features very clearly and mention the number of dimensions it is (ie: the number of values it returns)


