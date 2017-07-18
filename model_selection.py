# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# some logic-foo need to be done here.
import config
conf = config.get_config()
from models.wrapper import HRED_Wrapper
import random

class ModelSelection(object):

    def __init__(self):
        self.hred_model_twitter = None
        self.hred_model_reddit = None

    def initialize_models(self):
        self.hred_model_twitter = HRED_Wrapper(conf.hred['twitter_model_prefix'], conf.hred['twitter_dict_file'], 'twitter',None)
        self.hred_model_reddit = HRED_Wrapper(conf.hred['reddit_model_prefix'], conf.hred['reddit_dict_file'], 'reddit',None)

    def get_response(self,chat_id,text,context):
        resp1,cont1 = self.hred_model_twitter.get_response(chat_id,text,context)
        resp2,cont2 = self.hred_model_reddit.get_response(chat_id,text,context)
        # doing a random toss for now
        if random.choice([True,False]):
            return resp1,cont1
        else:
            return resp2,cont2





