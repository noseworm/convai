from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import numpy as np

"""
This file defines different hand-enginered features based
on the conversation article, the conversation hisory, and
one candidate response.
These features will then be used as input signals to a
fully-connected feed-foward neural network to predict either
the full dialogue score or the utterance score.

continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit
"""

print "loading word2vec embeddings..."
w2v = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin", binary=True)

print "loading nltk english stop words..."
stop = set(stopwords.words('english'))


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
    # To get raw features call "np.array([f.feat for f in feature_objects]).flatten()"


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
        if context is None:
            self.feat = None
        else:
            self.feat = float(len(context[-1]))


class AverageWordEmbedding_Candidate(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(UserLength, self).__init__(w2v.vector_size, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the candidate response
        """
        if candidate is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            for tok in candidate.strip().split(' '):
                if tok in w2v:
                    X += w2v[tok]
            X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_User(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(UserLength, self).__init__(w2v.vector_size, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the last user turn
        """
        if context is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            for tok in context[-1].strip().split(' '):
                if tok in w2v:
                    X += w2v[tok]
            X = np.array(X)/np.linalg.norm(X)
            self.feat = X


# TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
# Make sure to explain new features very clearly and mention the number of dimensions it is (ie: the number of values it returns)
# continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit


