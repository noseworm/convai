# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# some logic-foo need to be done here.
import config
conf = config.get_config()
from models.wrapper import HRED_Wrapper, Dual_Encoder_Wrapper, HREDQA_Wrapper, CandidateQuestions_Wrapper, DumbQuestions_Wrapper, DRQA_Wrapper
import random
import copy
import spacy
import re
import time
import emoji

nlp = spacy.load('en')

BORED_COUNT = 2


class ModelSelection(object):

    def __init__(self):
        self.article_text = {}     # map from chat_id to article text
        self.candidate_model = {}  # map from chat_id to a simple model for the article
        self.article_nouns = {}    # map from chat_id to a list of nouns in the article
        self.boring_count = {}     # number of times the user responded with short answer

    def initialize_models(self):
        self.hred_model_twitter = HRED_Wrapper(conf.hred['twitter_model_prefix'], conf.hred['twitter_dict_file'], 'hred-twitter')
        self.hred_model_reddit = HRED_Wrapper(conf.hred['reddit_model_prefix'], conf.hred['reddit_dict_file'], 'hred-reddit')
        self.de_model_reddit = Dual_Encoder_Wrapper(conf.de['reddit_model_prefix'], conf.de['reddit_data_file'], conf.de['reddit_dict_file'], 'de-reddit')
        self.qa_hred = HREDQA_Wrapper(conf.followup['model_prefix'],conf.followup['dict_file'],'followup_qa')
        self.dumb_qa = DumbQuestions_Wrapper('',conf.dumb['dict_file'],'dumb_qa')
        self.drqa = DRQA_Wrapper('','','drqa')
        # warmup the models before serving
        r,_ = self.hred_model_twitter.get_response(1, 'test statement', [])
        r,_ = self.hred_model_reddit.get_response(1, 'test statement', [])
        r,_ = self.de_model_reddit.get_response(1, 'test statement', [])
        r,_ = self.qa_hred.get_response(1, 'test statement', [])
        r,_ = self.drqa.get_response(1, 'Where is Daniel?', [], 'Daniel went to the kitchen')

    def clean(self, chat_id):
        del self.article_text[chat_id]
        del self.candidate_model[chat_id]
        del self.article_nouns[chat_id]
        del self.boring_count[chat_id]

    # get all the nouns from the text
    def _get_nouns(self, chat_id):
        self.article_nouns[chat_id] = [p.text for p in self.article_text[chat_id] if p.pos_ == 'NOUN']
        print self.article_nouns[chat_id]

    def strip_emojis(self,str):
        tokens = set(str.split())
        emojis = list(tokens.intersection(set(emoji.UNICODE_EMOJI)))
        if len(emojis) > 0:
            text = ''.join(c for c in str if c not in emojis)
            emojis = ''.join(emojis)
            return text,emojis
        return str,None

    def get_response(self, chat_id, text, context):
        # if text containes /start, dont add it to the context
        if '/start' in text:
            # save the article for later use
            text = re.sub('\\start','',text)
            self.article_text[chat_id] = nlp(unicode(text))
            self._get_nouns(chat_id)
            self.candidate_model[chat_id] = CandidateQuestions_Wrapper('', self.article_text[chat_id],
                                                                       conf.candidate['dict_file'], 'candidate_question')
            self.boring_count[chat_id] = 0  # initialize bored count to 0 for this new chat
            # Always generate first response
            #resp = 'Nice article, what is it about?'
            # add a small delay
            time.sleep(2)
            resp, context = self.candidate_model[chat_id].get_response(chat_id, '', context)
            #context.append('<first_speaker>' + resp + '</s>')
            return resp,context
        # chat selection logic
        # if text contains a question, do not respond with a question (followup)
        # if query falls under dumb questions, respond appropriately
        if self.dumb_qa.isMatch(text):
            resp,context = self.dumb_qa.get_response(chat_id,text,context)
            return resp,context

        # if text contains emoji's, strip them
        text,emojis = self.strip_emojis(text)
        if emojis and len(text.strip()) < 1:
            # give back the emoji itself
            return emojis,context
        
        # if text does not contain anything else
        # if text contains question, run DRQA
        if '?' in text:
	    # if there is a common noun between text and article, run drqa
	    common = list(set(self.article_nouns[chat_id]).intersection(set(text.split(' '))))
            print 'common',common
	    if len(common) > 0:
            	resp,context = self.drqa.get_response(chat_id,text,context,article=self.article_text[chat_id].text)
	    	return resp,context
	    else:
		if random.choice([True,False,False,False]): # sampling with 0.25 probability
		   resp,context = self.drqa.get_response(chat_id,text,context,article=self.article_text[chat_id].text)
		   return resp,context

        # if text contains 2 words or less, add 1 to the bored count
        if len(text.strip().split()) <= 2:
            self.boring_count[chat_id] += 1
        # if user is bored, change the topic by asking a question (only if that question is not asked before)
        if self.boring_count[chat_id] >= BORED_COUNT:
            resp_c,context_c = self.candidate_model[chat_id].get_response(chat_id,'',copy.deepcopy(context))
            if resp_c != '':
                self.boring_count[chat_id] = 0  # reset bored count to 0
                return resp_c,context_c

        outputs = []
        # randomly decide a model to generate. now we change the selection so that we pre-select the model to generate before hand.
        models = ['hred-twitter','hred-reddit','de']
        if '?' not in text:
           models.append('qa')
        chosen_model = random.choice(models)
        origin_context = copy.deepcopy(context)
        if chosen_model == 'hred-twitter':
            resp,cont = self.hred_model_twitter.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
        if chosen_model == 'hred-reddit':
            resp,cont = self.hred_model_reddit.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
        if chosen_model == 'de':
            resp,cont = self.de_model_reddit.get_response(chat_id, text, origin_context, self.article_text[chat_id])
        if chosen_model == 'qa':
            resp,cont = self.qa_hred.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))

        # chat selection logic
        # for now, select in random
        # ch = random.choice(range(len(outputs)))

        return resp,cont

