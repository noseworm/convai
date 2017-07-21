# Candidate response chooser model
import re
import operator
import random
import logging

logger = logging.getLogger(__name__)

# article should be a spacy converted article
# dataset file should have each candidate sentence in one line
class CandidateQuestions(object):
    def __init__(self,article,dataset_file,top_n=3):
        self.doc = article
        self.entities = [] # list of spacy tokens having entities
        self.entities_str = [] # string of the tokens
        self._get_entities()
        self.dataset = []
        self.entity2line = {} # dictionary containing a list of indices of the sentences
        ct = 0
        with open(dataset_file,'r') as fp:
            for line in fp:
                self.dataset.append(line)
                ents = re.findall('<(.*?)>',line,re.DOTALL)
                if len(ents) > 0:
                    for ent in ents:
                        if ent not in self.entity2line:
                            self.entity2line[ent] = []
                        self.entity2line[ent].append(ct)
                ct += 1

        self.token_distribution = {}  # distribution of words
        self.entity_distribution = {} # distribution of spacy entities
        for ent in set(self.entities):
            self.token_distribution[ent.text] = (self.entities_str.count(ent.text) * 1.0) / len(self.entities)
            self.entity_distribution[ent.ent_type_] = (len([p for p in self.entities
                if p.ent_type_ == ent.ent_type_])*1.0) / len(self.entities)

        # reverse sort the distribution dict
        self.token_distribution = sorted(self.token_distribution.items(), key=operator.itemgetter(1))
        self.token_distribution.reverse()
        self.top_n = top_n

    def _get_entities(self):
        self.line2token = {} # dictionary containing which tokens correspond to which line
        self.neighbors = {}
        for i,sent in enumerate(self.doc.sents):
            self.line2token[i] = []
            for token in sent:
                if token.ent_type_ != '':
                    self.entities.append(token)
                    self.entities_str.append(token.text)
                    self.line2token[i].append(token)

    # get the spacy token given string
    def _get_entity(self,token_str):
        # multiple tokens are present having same text but different ent_type_
        toks = [tok for tok in self.entities if tok.text == token_str]
        if len(toks) > 0:
            return random.choice(toks)
        return None

    # generate a random response based on the article
    def get_response(self):
        # select randomly among top_n entities as per distribution
        token = self._get_entity(random.choice(self.token_distribution[:self.top_n])[0])
        # get the ent_type_ and sample a line to use
        response = ''
        if token.ent_type_.lower() in self.entity2line:
            line = self.dataset[random.choice(self.entity2line[token.ent_type_.lower()])]
            response = re.sub('<(.*?)>',token.text,line)
        logger.info('Response: %s' % response)
        return response




