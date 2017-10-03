
"""
This file defines different hand-enginered features based
on the conversation article, the conversation hisory, and
one candidate response.
These features will then be used as input signals to a
fully-connected feed-foward neural network to predict either
the full dialogue score or the utterance score.
"""


def get(article, context, candidate, feature_list):
    """
    The only method to call to get all features we want for a triple
    of (article, context, candidate response).
    :param article: the text of the conversation article to talk to
    :param context: the list of user & bot utterances so far
    :param candidate: the candidate response one model proposed
    :feature_list: list of features to return for the above triple.
    :return: an aray containing all feature objects you requested for.
    """
    feature_objects = []  # list of feature objects

    for f in feature_list:
        feature = eval(f)(article, context, candidate)
        feature_objects.append(feature)
        

    if len(feature_objects) == 0:
        print "WARNING: no feature recognized in %s" % (feature_list,)
    
    return feature_objects


class Feature(Object):

    self.dim = -1
    self.feat = None

    def __init__(self, dim, article=None, context=None, candidate=None):
        self.dim = dim
        
    def set(self, article, context, candidate):
        """
        To be implemented in each sub-class
        """
        pass


class CandidateLength(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(CandidateLength, self).__init__(1, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribure to float scalar (dim: 1) for the length of the candidate response
        """
        if candidate is None:
            self.feat = None
        else:
            self.feat = float(len(candidate))


class UserLength(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(UserLength, self).__init__(1, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to float scalar (dim: 1) for the length of the previous user utterance
        """
        if context = None:
            self.feat = None
        else:
            self.feat = float(len(context[-1]))


# TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
# Make sure to explain new features very clearly and mention the number of dimensions it is (ie: the number of values it returns)


