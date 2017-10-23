from nltk.corpus import stopwords
from nltk import ngrams, word_tokenize, sent_tokenize
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from embedding_metrics import w2v
from embedding_metrics import greedy_score, extrema_score, average_score

import spacy
import logging
logger = logging.getLogger(__name__)

"""
This file defines different hand-enginered features based
on the conversation article, the conversation hisory, and
one candidate response.
These features will then be used as input signals to a
fully-connected feed-foward neural network to predict either
the full dialogue score or the utterance score.

continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit
"""

logger.info("loading nltk english stop words...")
stop = set(stopwords.words('english'))
logger.info(stop)
logger.info("")


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
            for tok in candidate.strip().split():
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
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
            for tok in context[-1].strip().split():
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
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
            logger.debug("last %d turns: %s" % (self.k, content))
            for tok in content.strip().split():
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
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
            logger.debug("last %d user turns: %s" % (self.k, content))
            for tok in content.strip().split():
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
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
            for tok in article.strip().split():
                if tok in w2v:
                    X += w2v[tok]
            if np.linalg.norm(X) > 0.00000000001:
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
            self.feat = [(res1 + res2) / 2.0]


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
            self.feat = [float(average_score(candidate, context[-1]))]


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
            self.feat = [float(extrema_score(candidate, context[-1]))]


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
            logger.debug("last %d turns: %s" % (self.k, content))
            res1 = greedy_score(candidate, content)
            res2 = greedy_score(content, candidate)
            self.feat = [(res1 + res2) / 2.0]


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
            logger.debug("last %d turns: %s" % (self.k, content))
            self.feat = [float(average_score(candidate, content))]


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
            logger.debug("last %d turns: %s" % (self.k, content))
            self.feat = [float(extrema_score(candidate, content))]


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
            logger.debug("last %d turns: %s" % (self.k, content))
            res1 = greedy_score(candidate, content)
            res2 = greedy_score(content, candidate)
            self.feat = [(res1 + res2) / 2.0]


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
            logger.debug("last %d turns: %s" % (self.k, content))
            self.feat = [float(average_score(candidate, content))]


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
            logger.debug("last %d turns: %s" % (self.k, content))
            self.feat = [float(extrema_score(candidate, content))]


### Candidate -- last k user turns match ###

class GreedyScore_CandidateKUser(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(GreedyScore_CandidateKUser, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to greedy score (dim: 1) between candidate response & last k user turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            logger.debug("last %d user turns: %s" % (self.k, content))
            res1 = greedy_score(candidate, content)
            res2 = greedy_score(content, candidate)
            self.feat = [(res1 + res2) / 2.0]


class AverageScore_CandidateKUser(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(AverageScore_CandidateKUser, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average embedding score (dim: 1) between candidate response & last k user turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            logger.debug("last %d user turns: %s" % (self.k, content))
            self.feat = [float(average_score(candidate, content))]


class ExtremaScore_CandidateKUser(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(ExtremaScore_CandidateKUser, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to extrema embedding score (dim: 1) between candidate response & last k user turns
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            logger.debug("last %d user turns: %s" % (self.k, content))
            self.feat = [float(extrema_score(candidate, content))]


### Candidate -- last k user turns without stop words match ###

class GreedyScore_CandidateKUser_noStop(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(GreedyScore_CandidateKUser_noStop, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to greedy score (dim: 1) between candidate response & last k user turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            content = ' '.join(filter(lambda word: word not in stop, content.strip().split()))
            logger.debug("last %d user turns: %s" % (self.k, content))
            res1 = greedy_score(candidate, content)
            res2 = greedy_score(content, candidate)
            self.feat = [(res1 + res2) / 2.0]


class AverageScore_CandidateKUser_noStop(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(AverageScore_CandidateKUser_noStop, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to average embedding score (dim: 1) between candidate response & last k user turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            content = ' '.join(filter(lambda word: word not in stop, content.strip().split()))
            logger.debug("last %d user turns: %s" % (self.k, content))
            self.feat = [float(average_score(candidate, content))]


class ExtremaScore_CandidateKUser_noStop(Feature):

    def __init__(self, k=3, article=None, context=None, candidate=None):
        # Constructor: call super class constructor with dim=1
        super(ExtremaScore_CandidateKUser_noStop, self).__init__(1, article, context, candidate)
        self.k = k
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        set feature attribute to extrema embedding score (dim: 1) between candidate response & last k user turns without stop words
        """
        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)[range(-2*self.k+1, 0, 2)]
            content = ' '.join(content)
            content = ' '.join(filter(lambda word: word not in stop, content.strip().split()))
            logger.debug("last %d user turns: %s" % (self.k, content))
            self.feat = [float(extrema_score(candidate, content))]


### Candidate -- article match ###



### Candidate -- article without stop words match ###



### n-gram & entity overlaps ###

class BigramOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(BigramOverlap, self).__init__(2, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         0 / 1 indicating if response has at least one bigram overlapping with:
        the previous user turn (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        any previous turn (binary feature size: 1) -- for f_pi(a, h, i)
        """

        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            content = ' '.join(content)
            response_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(candidate.lower()),2)])
            last_response_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(last_response.lower()), 2)])
            content_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(content.lower()), 2)])

            self.feat = np.zeros(2)
            if len(last_response_bigrams.intersection(response_bigrams)) > 0:
                self.feat[0] = 1
            if len(content_bigrams.intersection(response_bigrams)) > 0:
                self.feat[1] = 1

class TrigramOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(TrigramOverlap, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         0 / 1 indicating if response has at least one trigram overlapping with:
        the previous user turn (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        any previous turn (binary feature size: 1) -- for f_pi(a, h, i)
        the article (binary feature size: 1)
        """

        if candidate is None or context is None or article is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            content = ' '.join(content)
            response_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(candidate.lower()),3)])
            last_response_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(last_response.lower()), 3)])
            content_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(content.lower()), 3)])
            article_bigrams = set(['-'.join(grams) for grams in ngrams(word_tokenize(article.lower()), 3)])

            self.feat = np.zeros(3)
            if len(last_response_bigrams.intersection(response_bigrams)) > 0:
                self.feat[0] = 1
            if len(content_bigrams.intersection(response_bigrams)) > 0:
                self.feat[1] = 1
            if len(article_bigrams.intersection(response_bigrams)) > 0:
                self.feat[2] = 1

class EntityOverlap(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(EntityOverlap, self).__init__(3, article, context, candidate)
        self.nlp = spacy.load('en')
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         0 / 1 indicating if response has at least one entity overlapping with:
        the previous user turn (binary feature size: 1) -- for f_pi(a, h, i)
        any previous turn (binary feature size: 1) -- for f_pi(a, h, i)
        the article (binary feature size: 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        """

        if candidate is None or context is None or article is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            content = ' '.join(content)
            content = self.nlp(unicode(content))
            article = self.nlp(unicode(article))
            candidate = self.nlp(unicode(candidate))
            last_response = self.nlp(unicode(last_response))
            content_entities = set([ent.label_ for ent in content.ents])
            article_entities = set([ent.label_ for ent in article.ents])
            candidate_entities = set([ent.label_ for ent in candidate.ents])
            last_response_entities = set([ent.label_ for ent in last_response.ents])
            self.feat = np.zeros(3)
            if len(last_response_entities.intersection(candidate_entities)) > 0:
                self.feat[0] = 1
            if len(content_entities.intersection(candidate_entities)) > 0:
                self.feat[1] = 1
            if len(article_entities.intersection(candidate_entities)) > 0:
                self.feat[2] = 1


### word presence ###

class WhWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(WhWords, self).__init__(2, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         0 / 1 indicating if the turn has a word starting with "wh"
        on the candidate response (scalar of size 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        on the last user turn (scalar of size 1) -- for f_pi(a, h, i) and g_phi(a, h, i)
        """

        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            candidate = word_tokenize(candidate.lower())
            last_response = word_tokenize(last_response.lower())
            wh_candidate = len([word for word in candidate if word.startswith('wh')])
            wh_last = len([word for word in last_response if word.startswith('wh')])
            self.feat = np.zeros(2)
            if wh_candidate > 0:
                self.feat[0] = 1
            if wh_last > 0:
                self.feat[1] = 1


### length ###

class DialogLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(DialogLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         number of turns so far n, sqrt(n), log(n) (3 scalars: size 3) -- for g_phi(a, h, i)
        """

        if context is None:
            self.feat = None
        else:
            content = np.array(context)
            self.feat = np.zeros(3)
            self.feat[0] = len(content)
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of turns in context is zero: `%s`" % content
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class LastUserLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(LastUserLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         number of words n, sqrt(n), log(n) (3 scalars: size 3) -- for g_phi(a, h, i)
        """

        if context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_user_turn = content[-1]
            self.feat = np.zeros(3)
            self.feat[0] = len(word_tokenize(last_user_turn))
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of words in last user msg is zero: `%s`" % last_user_turn
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class CandidateLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(CandidateLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
        number of words n, sqrt(n), log(n) (3 scalars: dim 3)
        """
        if candidate is None:
            self.feat = None
        else:
            self.feat = np.zeros(3)
            self.feat[0] = len(word_tokenize(candidate))
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of words in candidate is zero: `%s`" % candidate
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class ArticleLength(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(LastUserLength, self).__init__(3, article, context, candidate)
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         number of sentences in the article n, sqrt(n), log(n) (3 scalars: size 3) -- for g_phi(a, h, i)
        """

        if article is None:
            self.feat = None
        else:
            article_sents = sent_tokenize(article)
            self.feat = np.zeros(3)
            self.feat[0] = len(article_sents)
            self.feat[1] = np.sqrt(self.feat[0])
            if self.feat[0] == 0:
                print "Warning: number of sentences in article is zero: `%s`" % article
                self.feat[2] = 0
            else:
                self.feat[2] = np.log(self.feat[0])


class IntensifierWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(IntensifierWords, self).__init__(4, article, context, candidate)
        self.intensifier_list = []
        with open('../data/intensifier_list.txt') as fp:
            for line in fp:
                self.intensifier_list.append(line.strip())
        self.set(article, context, candidate)

    def count(self, text):
        counter = 0
        for phrase in self.intensifier_list:
            if phrase in text:
                counter +=1
        return counter

    def set(self, article, context, candidate):
        """
         4 features:
         1: candidate - boolean value if the candidate response contains any words from the list
         2: candidate - float percentage of words which are flagged in the intensifier list
         3: last response - boolean value if the candidate response contains any words from the list
         4: last response - float percentage of words which are flagged in the intensifier list
        """

        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            last_response_intensifier_words_count = self.count(last_response)
            candidate_intensifier_words_count = self.count(candidate)
            self.feat = np.zeros(4)
            if candidate_intensifier_words_count > 0:
                self.feat[0] = 1
            if last_response_intensifier_words_count > 0:
                self.feat[2] = 1
            self.feat[1] = (1.0 * candidate_intensifier_words_count) / len(word_tokenize(candidate))
            self.feat[3] = (1.0 * last_response_intensifier_words_count) / len(word_tokenize(last_response))


class ConfusionWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(ConfusionWords, self).__init__(4, article, context, candidate)
        self.confusion_list = []
        with open('../data/confusion_list.txt') as fp:
            for line in fp:
                self.confusion_list.append(line.strip())
        self.set(article, context, candidate)

    def count(self, text):
        counter = 0
        for phrase in self.confusion_list:
            if phrase in text:
                counter +=1
        return counter

    def set(self, article, context, candidate):
        """
         4 features:
         1: candidate - boolean value if the candidate response contains any words from the list
         2: candidate - float percentage of words which are flagged in the confusion list
         3: last response - boolean value if the candidate response contains any words from the list
         4: last response - float percentage of words which are flagged in the confusion list
        """

        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            last_response_confusion_words_count = self.count(last_response)
            candidate_confusion_words_count = self.count(candidate)
            self.feat = np.zeros(4)
            if candidate_confusion_words_count > 0:
                self.feat[0] = 1
            if last_response_confusion_words_count > 0:
                self.feat[2] = 1
            self.feat[1] = (1.0 * candidate_confusion_words_count) / len(word_tokenize(candidate))
            self.feat[3] = (1.0 * last_response_confusion_words_count) / len(word_tokenize(last_response))


class ProfanityWords(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(ProfanityWords, self).__init__(4, article, context, candidate)
        self.profanity_list = []
        with open('../data/profanity_list.txt') as fp:
            for line in fp:
                self.profanity_list.append(line.strip())
        self.set(article, context, candidate)

    def count(self, text):
        counter = 0
        for phrase in self.profanity_list:
            if phrase in text:
                counter +=1
        return counter

    def set(self, article, context, candidate):
        """
         4 features:
         1: candidate - boolean value if the candidate response contains any words from the list
         2: candidate - float percentage of words which are flagged in the profanity list
         3: last response - boolean value if the candidate response contains any words from the list
         4: last response - float percentage of words which are flagged in the profanity list
        """

        if candidate is None or context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            last_response_hate_words_count = self.count(last_response)
            candidate_hate_words_count = self.count(candidate)
            self.feat = np.zeros(4)
            if candidate_hate_words_count > 0:
                self.feat[0] = 1
            if last_response_hate_words_count > 0:
                self.feat[2] = 1
            self.feat[1] = (1.0 * candidate_hate_words_count) / len(word_tokenize(candidate))
            self.feat[3] = (1.0 * last_response_hate_words_count) / len(word_tokenize(last_response))

class SentimentScoreCandidate(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(SentimentScoreCandidate, self).__init__(3, article, context, candidate)
        self.analyzer = SentimentIntensityAnalyzer()
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         3 features: one hot vector for candidate to be positive, negative or neutral
        """
        if candidate is None:
            self.feat = None
        else:
            candidate_vs = self.analyzer.polarity_scores(candidate)
            self.feat = np.zeros(3)
            if candidate_vs['compound'] >= 0.5: # positive
                self.feat[0] = 1
            elif candidate_vs['compound'] <= -0.5: # negative
                self.feat[1] = 1
            else:
                self.feat[2] = 1 # neutral

class SentimentScoreLastUser(Feature):
    def __init__(self, article=None, context=None, candidate=None):
        super(SentimentScoreLastUser, self).__init__(3, article, context, candidate)
        self.analyzer = SentimentIntensityAnalyzer()
        self.set(article, context, candidate)

    def set(self, article, context, candidate):
        """
         3 features: one hot vector for last user turn to be positive, negative or neutral
        """

        if context is None:
            self.feat = None
        else:
            content = np.array(context)
            last_response = content[-1]
            last_response_vs = self.analyzer.polarity_scores(last_response)
            self.feat = np.zeros(3)
            if last_response_vs['compound'] >= 0.5: # positive
                self.feat[0] = 1
            elif last_response_vs['compound'] <= -0.5: # negative
                self.feat[1] = 1
            else:
                self.feat[2] = 1 # neutral


# TODO: continue defining new features like embedding metrics, word overlap metrics, lookup for specific words, etc...
# Make sure to explain new features very clearly and mention the number of dimensions it is (ie: the number of values it returns)
# continue adding features from https://docs.google.com/document/d/1PAVoHP_I39L6Rk1e8pIvq_wFW-_oyVa1qy1hjKC9E5M/edit


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

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

    features = [
        'AverageWordEmbedding_Candidate', 'AverageWordEmbedding_User', 'AverageWordEmbedding_LastK', 'AverageWordEmbedding_kUser', 'AverageWordEmbedding_Article',
        'GreedyScore_CandidateUser', 'AverageScore_CandidateUser', 'ExtremaScore_CandidateUser',
        'GreedyScore_CandidateLastK', 'AverageScore_CandidateLastK', 'ExtremaScore_CandidateLastK',
        'GreedyScore_CandidateLastK_noStop', 'AverageScore_CandidateLastK_noStop', 'ExtremaScore_CandidateLastK_noStop',
        'GreedyScore_CandidateKUser', 'AverageScore_CandidateKUser', 'ExtremaScore_CandidateKUser',
        'GreedyScore_CandidateKUser_noStop', 'AverageScore_CandidateKUser_noStop', 'ExtremaScore_CandidateKUser_noStop',
        'EntityOverlap','BigramOverlap','TrigramOverlap','WhWords','DialogLength','LastUserLength','ArticleLength','CandidateLength'
    ]

    for feature in features:
        logger.info("class: %s" % feature)
        feature_obj = get(article, context, candidate1, [feature])[0]
        logger.info("feature: %s" % (feature_obj.feat,))
        logger.info("dim: %d" % feature_obj.dim)
        logger.info("")


