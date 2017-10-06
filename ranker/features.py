from nltk.corpus import stopwords
import numpy as np

from embedding_metrics import w2v
from embedding_metrics import greedy_score, extrema_score, average_score

"""
This file defines different hand-enginered features based
on the conversation article, the conversation hisory, and
one candidate response.
These features will then be used as input signals to a
fully-connected feed-foward neural network to predict either
the full dialogue score or the utterance score.

continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit
"""

print "loading nltk english stop words..."
stop = set(stopwords.words('english'))
print stop
print ""


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
        feature = eval(f)(article=article, context=context, candidate=candidate)
        feature_objects.append(feature)

    if len(feature_objects) == 0:
        print "WARNING: no feature recognized in %s" % (feature_list,)
    
    return feature_objects
    # To get raw features call "np.array([f.feat for f in feature_objects]).flatten()"


#####################
### GENERIC CLASS ###
#####################

class Feature(object):

    def __init__(self, dim, article=None, context=None, candidate=None):
        self.dim = dim
        self.feat = None
        
    def set(self, article, context, candidate):
        """
        To be implemented in each sub-class
        """
        pass


############################
### SPECIFIC SUB-CLASSES ###
############################


### Length ###

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
            self.feat = float(len(candidate.strip().split()))


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
            self.feat = float(len(context[-1].strip().split()))


### Average embedding ###

class AverageWordEmbedding_Candidate(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=300
        super(AverageWordEmbedding_Candidate, self).__init__(w2v.vector_size, article, context, candidate)
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
        super(AverageWordEmbedding_User, self).__init__(w2v.vector_size, article, context, candidate)
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
        super(AverageWordEmbedding_LastK, self).__init__(w2v.vector_size, article, context, candidate)
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
        super(AverageWordEmbedding_kUser, self).__init__(w2v.vector_size, article, context, candidate)
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
        super(AverageWordEmbedding_Article, self).__init__(w2v.vector_size, article, context, candidate)
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


### Candidate -- user turn match ###

class GreedyScore_CandidateUser(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(GreedyScore_CandidateUser, self).__init__(1, article, context, candidate)
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


class AverageScore_CandidateUser(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(AverageScore_CandidateUser, self).__init__(1, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average embedding score (dim: 1) between candidate response & last user turn
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            self.feat = float(average_score(candidate, context[-1]))


class ExtremaScore_CandidateUser(Feature):

    def __init__(self, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(ExtremaScore_CandidateUser, self).__init__(1, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to extrema embedding score (dim: 1) between candidate response & last user turn
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            self.feat = float(extrema_score(candidate, context[-1]))


### Candidate -- last k turns match ###

class GreedyScore_CandidateLastK(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(GreedyScore_CandidateLastK, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to greedy score (dim: 1) between candidate response & last k turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = ' '.join(context[-self.k:])
            res1 = greedy_score(candidate, content)
            res2 = greedy_score(content, candidate)
            self.feat = (res1 + res2) / 2.0


class AverageScore_CandidateLastK(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(AverageScore_CandidateLastK, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average embedding score (dim: 1) between candidate response & last k turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = ' '.join(context[-self.k:])
            self.feat = float(average_score(candidate, content))


class ExtremaScore_CandidateLastK(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(ExtremaScore_CandidateLastK, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to extrema embedding score (dim: 1) between candidate response & last k turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = ' '.join(context[-self.k:])
            self.feat = float(extrema_score(candidate, content))


### Candidate -- last k turns without stop words match ###

class GreedyScore_CandidateLastK_noStop(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(GreedyScore_CandidateLastK_noStop, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to greedy score (dim: 1) between candidate response & last k turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = ' '.join(context[-self.k:])
            content = ' '.join(filter(lambda word: word not in stop, content.strip().split()))
            res1 = greedy_score(candidate, content)
            res2 = greedy_score(content, candidate)
            self.feat = (res1 + res2) / 2.0


class AverageScore_CandidateLastK_noStop(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(AverageScore_CandidateLastK_noStop, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average embedding score (dim: 1) between candidate response & last k turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = ' '.join(context[-self.k:])
            content = ' '.join(filter(lambda word: word not in stop, content.strip().split()))
            self.feat = float(average_score(candidate, content))


class ExtremaScore_CandidateLastK_noStop(Feature):

    def __init__(self, k=6, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(ExtremaScore_CandidateLastK_noStop, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to extrema embedding score (dim: 1) between candidate response & last k turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = ' '.join(context[-self.k:])
            content = ' '.join(filter(lambda word: word not in stop, content.strip().split()))
            self.feat = float(extrema_score(candidate, content))


# TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
# Make sure to explain new features very clearly and mention the number of dimensions it is (ie: the number of values it returns)
# continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit


if __name__ == '__main__':
    article = "russia asks facebook to comply with personal data policy friday, september 29, 2017 \
        on tuesday, russian government internet watchdog roskomnadzor 'insisted' us - based social \
        networking website facebook comply with law # 242 on personal data of users in order to \
        continue operating in the country . per law # 242 , user data of russian citizens should be \
        hosted on local servers - the rule which business - oriented networking site linkedin did \
        not agree to, for which linkedin was eventually blocked in the country."

    context = ["hello user ! this article is very interesting don 't you think ?",
        "hello chat bot ! yes indeed . looks like russia starts to apply the same rules as china",
        "yeah i don 't know about that .",
        "facebook should be available everywhere in the world",
        "yeah i don 't know about that .",
        "you don 't know much do you ? ",
        "i am not a fan of russian policies",
        "haha me neither !"
    ]
    candidate1 = "i am happy to make you laught"
    candidate2 = "ha ha ha"
    candidate3 = "i like facebook"

    feature_objects = get(article, context, candidate1,
        ['CandidateLength', 'UserLength',
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK', 'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
        'GreedyScore_CandidateUser', 'AverageScore_CandidateUser', 'ExtremaScore_CandidateUser',
        'GreedyScore_CandidateLastK', 'AverageScore_CandidateLastK', 'ExtremaScore_CandidateLastK',
        'GreedyScore_CandidateLastK_noStop', 'AverageScore_CandidateLastK_noStop', 'ExtremaScore_CandidateLastK_noStop']
    )

    for feature_obj in feature_objects:
        print feature_obj.__class__
        print "feature:", feature_obj.feat
        print "dim:", feature_obj.dim
        print ""


