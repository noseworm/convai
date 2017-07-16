# Configuration file

class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

config_data = {
        'bot_token':'355748420:AAEpaGukZEeC1jFwvVU2TVf3d92fgq6VrKU',
        'bot_endpoint':'',
        # data endpoints
        # add your model endpoints here
        'hred': {
            'twitter_model_prefix':'/root/convai/data/twitter_model/1489857182.98_TwitterModel',
            'reddit_model_prefix':'/root/convai/data/reddit_model/1485212785.88_RedditHRED',
            'twitter_dict_file':'/root/convai/data/twitter_model/Dataset.dict-5k.pkl',
            'reddit_dict_file':'/root/convai/data/reddit_model/Training.dict.pkl'
        }
}

def get_config():
    return dotdict(config_data)
