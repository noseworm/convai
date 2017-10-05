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


##################################################
### WORD EMBEDDING SIMILARITY METRIC FUNCTIONS ###
##################################################

def greedy_score(one, two):
    """Greedy matching between two texts"""
    dim = w2v.vector_size  # dimension of embeddings

    tokens1 = one.strip().split(" ")
    tokens2 = two.strip().split(" ")
    # X = np.zeros((dim,))  # array([ 0.,  0.,  0.,  0.])
    y_count = 0
    x_count = 0
    score = 0.0
    Y = np.zeros((dim,1))  # array([ [0.],  [0.],  [0.],  [0.] ])
    for tok in tokens2:    # for each token in the second text, add its column to Y
        if tok in w2v:
            y = np.array(w2v[tok])
            y /= np.linalg.norm(y)
            Y = np.hstack((Y, y.reshape((dim,1)) ))
            y_count += 1
    # Y ~ (dim, #of tokens in second text)
    # Y /= np.linalg.norm(Y)

    for tok in tokens1:  # for each token in the first text,
        if tok in w2v:
            x = np.array(w2v[tok])
            x /= np.linalg.norm(x)
            tmp = x.reshape((1,dim)).dot(Y)  # dot product with all other tokens from Y
            score += np.max(tmp)  # add the max value between this token and any token in Y
            x_count += 1

    # if none of the words in response or ground truth have embeddings, return zero
    if x_count < 1 or y_count < 1:
        return 0.0

    score /= float(x_count)
    return score


def extrema_score(one, two):
    """Extrema embedding score between two texts"""
    tokens1 = one.strip().split(" ")
    tokens2 = two.strip().split(" ")
    X = []
    for tok in tokens1:
        if tok in w2v:
            X.append(w2v[tok])
    Y = []
    for tok in tokens2:
        if tok in w2v:
            Y.append(w2v[tok])

    # if none of the words in text1 have embeddings, return 0
    if np.linalg.norm(X) < 0.00000000001:
        return 0.0

    # if none of the words in text2 have embeddings, return 0
    if np.linalg.norm(Y) < 0.00000000001:
        return 0.0

    xmax = np.max(X, 0)  # get positive max
    xmin = np.min(X, 0)  # get abs of min
    xtrema = []
    for i in range(len(xmax)):
        if np.abs(xmin[i]) > xmax[i]:
            xtrema.append(xmin[i])
        else:
            xtrema.append(xmax[i])
    X = np.array(xtrema)  # get extrema

    ymax = np.max(Y, 0)
    ymin = np.min(Y, 0)
    ytrema = []
    for i in range(len(ymax)):
        if np.abs(ymin[i]) > ymax[i]:
            ytrema.append(ymin[i])
        else:
            ytrema.append(ymax[i])
    Y = np.array(ytrema)  # get extrema

    score = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
    return score


def average_score(one, two):
    """Average embedding score between two texts"""
    dim = w2v.vector_size # dimension of embeddings
    tokens1 = one.strip().split(" ")
    tokens2 = two.strip().split(" ")
    X = np.zeros((dim,))
    for tok in tokens1:
        if tok in w2v:
            X += w2v[tok]
    Y = np.zeros((dim,))
    for tok in tokens2:
        if tok in w2v:
            Y += w2v[tok]

    # if none of the words in text1 have embeddings, return 0
    if np.linalg.norm(X) < 0.00000000001:
        return 0.0

    # if none of the words in text2 have embeddings, return 0
    if np.linalg.norm(Y) < 0.00000000001:
        return 0.0

    X = np.array(X)/np.linalg.norm(X)
    Y = np.array(Y)/np.linalg.norm(Y)
    score = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
    return score

######################################################
### WORD EMBEDDING SIMILARITY METRIC FUNCTIONS END ###
######################################################


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


class AverageWordEmbedding_LastK(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(UserLength, self).__init__(w2v.vector_size, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the last k turns
        """
        if context is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            content = ' '.join(context[-self.k:])
            for tok in content.strip().split(' '):
                if tok in w2v:
                    X += w2v[tok]
            X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_kUser(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(UserLength, self).__init__(w2v.vector_size, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the last k user turns
        """
        if context is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            for tok in content.strip().split(' '):
                if tok in w2v:
                    X += w2v[tok]
            X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class AverageWordEmbedding_Article(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(UserLength, self).__init__(w2v.vector_size, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average word embedding (dim: 300) of the article
        """
        if article is None:
            self.feat = None
        else:
            X = np.zeros((self.dim,), dtype='float32')
            for tok in article.strip().split(' '):
                if tok in w2v:
                    X += w2v[tok]
            X = np.array(X)/np.linalg.norm(X)
            self.feat = X


class GreedyMatch_CandidateUser(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(UserLength, self).__init__(1, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to greedy score (dim: 1) between candidate response & last user turn
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            res1 = greedy_score(candidate, context[-1])
            res2 = greedy_score(context[-1], candidate)
            self.feat = (res1 + res2) / 2.0

# TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
# Make sure to explain new features very clearly and mention the number of dimensions it is (ie: the number of values it returns)
# continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit


