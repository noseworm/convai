# Configuration file

class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

config_data = {
        'test_bot_token':'355748420:AAEpaGukZEeC1jFwvVU2TVf3d92fgq6VrKU',
        'bot_token':'5319E57A-F165-4BEC-94E6-413C38B4ACF9',
        'bot_endpoint_backup':'https://ipavlov.mipt.ru/nipsrouter-alt/',
        'bot_endpoint':'https://koustuv.me/',
        # data endpoints
        # add your model endpoints here
        'data_base':'/root/convai/',
        'hred': {
            'twitter_model_prefix':'data/twitter_model/1489857182.98_TwitterModel',
            'reddit_model_prefix':'data/reddit_model/1485212785.88_RedditHRED',
            'twitter_dict_file':'data/twitter_model/Dataset.dict-5k.pkl',
            'reddit_dict_file':'data/reddit_model/Training.dict.pkl'
        },
        'de': {
            'reddit_model_prefix':'data/reddit-bpe5k_exp2/reddit_exp2',
            'reddit_data_file':'data/DE.dataset.pkl',
            'reddit_dict_file':'data/DE.dict.pkl'
        },
        'followup':{
            'model_prefix':'data/followup/',
            'dict_file':'data/followup/TrainingSmall.dict.pkl'
        },
        'candidate':{
            'dict_file':'data/candidate_dataset.txt'
        },
        'dumb':{
            'dict_file':'data/dumb_questions.json'
        },
        'topic':{
            'model_name':'/root/convai/data/yahoo_answers/fast.model.ep50.ng5.word2vec.bin',
            'dir_name':'data/yahoo_answers/',
            'top_k' : 2
        },
        "socket_port" : 8094,
        "ranker": {
            "model" : "/root/convai/ranker/models/short_term/0.643257/1510158946.66_VoteEstimator_args.pkl"
        }
}

def get_config():
    return dotdict(config_data)
