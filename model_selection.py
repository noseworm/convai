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

nlp = spacy.load('en')

BORED_COUNT = 2


class ModelSelection(object):

    def __init__(self):
        self.hred_model_twitter = None
        self.hred_model_reddit = None
        self.article_text = {}
        self.candidate_model = {}
	self.article_nouns = {}
        self.boring_count = 0 # whenever user responds with single word text, we assume that they are bored. if counter hits = BORED_COUNT, ask a question

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
        r,_ = self.drqa.get_response(1,'Where is Daniel?',[],'Daniel went to the kitchen')

    # get all the nouns from the text
    def _get_nouns(self,chat_id):
        self.article_nouns[chat_id] = [p.text for p in self.article_text[chat_id] if p.pos_ == 'NOUN']
	print self.article_nouns[chat_id]

    def get_response(self,chat_id,text,context):
        # if text containes /start, dont add it to the context
        if '/start' in text:
            # save the article for later use
            text = re.sub('\\start','',text)
            self.article_text[chat_id] = nlp(unicode(text))
	    self._get_nouns(chat_id)
            self.candidate_model[chat_id] = CandidateQuestions_Wrapper('',self.article_text[chat_id],
                    conf.candidate['dict_file'],'candidate_question')
            # generate first response or not?
            # with some randomness generate the first response or leave blank
            if random.choice([True,False]):
                #resp = 'Nice article, what is it about?'
                resp,context = self.candidate_model[chat_id].get_response(chat_id,'',context)
                #context.append('<first_speaker>' + resp + '</s>')
            else:
                resp = ''
            return resp,context
        # chat selection logic
        # if text contains a question, do not respond with a question (followup)
        # if query falls under dumb questions, respond appropriately
        if self.dumb_qa.isMatch(text):
            resp,context = self.dumb_qa.get_response(chat_id,text,context)
            return resp,context

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

        # if text contains one word, and it has happened before (twice), change the topic, ask a question (only if that question is not asked before)
        if self.boring_count >= BORED_COUNT:
            resp_c,context_c = self.candidate_model[chat_id].get_response(chat_id,'',copy.deepcopy(context))
            if resp_c != '':
                return resp_c,context_c
        else:
            self.boring_count += 1

        outputs = []
        origin_context = copy.deepcopy(context)
        resp1,cont1 = self.hred_model_twitter.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
        outputs.append((resp1,cont1))
        origin_context = copy.deepcopy(context)
        resp2,cont2 = self.hred_model_reddit.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
        outputs.append((resp2,cont2))
        origin_context = copy.deepcopy(context)
        resp3,cont3 = self.de_model_reddit.get_response(chat_id, text, origin_context, self.article_text[chat_id])
        outputs.append((resp3,cont3))
        origin_context = copy.deepcopy(context)
        if '?' not in text:
            resp4,cont4 = self.qa_hred.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
            outputs.append((resp4,cont4))

        # chat selection logic
        # for now, select in random
        ch = random.choice(range(len(outputs)))

        return outputs[ch]
