import zmq
import sys
import config
import random
import copy
import spacy
import re
import time
import emoji
import json
import cPickle as pkl
from ranker import features
from Queue import Queue
from threading import Thread
from multiprocessing import Process, Pool
import uuid
from models.wrapper import HRED_Wrapper, Dual_Encoder_Wrapper, HREDQA_Wrapper, CandidateQuestions_Wrapper, DumbQuestions_Wrapper, DRQA_Wrapper, NQG_Wrapper, Echo_Wrapper, Topic_Wrapper
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)
conf = config.get_config()


# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# ZMQ version. Here, all models are initialized in a separate thread
# Then the main process sends out next response and context as a PUB msg,
# which every model_client listens in SUB
# Then all models calculate the response and send it to parent using PUSH-PULL
# NN model selection algorithm can therefore work in the parent queue
# to calculate the score of all incoming msgs
# Set a hard time limit to discard the messages which are slow
# N.B. `import zmq` **has to be** the first import.


nlp = spacy.load('en')

# Utils


def mogrify(topic, msg):
    """ json encode the message and prepend the topic """
    return topic + ' ' + json.dumps(msg)


def demogrify(topicmsg):
    """ Inverse of mogrify() """
    json0 = topicmsg.find('{')
    topic = topicmsg[0:json0].strip()
    msg = json.loads(topicmsg[json0:])
    return topic, msg


class Policy:
    NONE = -1        # between each chat
    OPTIMAL = 1      # current set of rules
    # exploratory policies:
    HREDx2 = 2       # hred-reddit:0.5 & hred-twitter:0.5
    HREDx3 = 3       # hred-reddit:0.33 & hred-twitter:0.33 & hred-qa:0.33
    HREDx2_DE = 4    # hred-reddit:0.25 & hred-twitter:0.25 & DualEncoder:0.5
    HREDx2_DRQA = 5  # hred-reddit:0.25 & hred-twitter:0.25 & DrQA:0.5
    DE_DRQA = 6      # DualEncoder:0.5 & DrQA:0.5
    START = 7
    FIXED = 8        # when allowed_model = True


class ModelID:
    DRQA = 'drqa'
    DUAL_ENCODER = 'de'
    HRED_REDDIT = 'hred-reddit'
    HRED_TWITTER = 'hred-twitter'
    DUMB_QA = 'dumb_qa'
    NQG = 'nqg'
    FOLLOWUP_QA = 'followup_qa'
    CAND_QA = 'candidate_question'
    TOPIC = 'topic_model'
    ECHO = 'echo_model'  # just for debugging purposes
    ALL = 'all'          # stub to represent all allowable models


ALL_POLICIES = [Policy.OPTIMAL, Policy.HREDx2, Policy.HREDx3,
                Policy.HREDx2_DE, Policy.HREDx2_DRQA, Policy.DE_DRQA]

BORED_COUNT = 2

# wait time is the amount of time we wait to let the models respond.
# Instead of the previous architecture, now we would like to respond faster.
# So the focus is that even if some models are taking a lot of time,
# do not wait for them!
WAIT_TIME = 10

# IPC pipes
# Parent to models
COMMAND_PIPE = 'ipc:///tmp/command.pipe'
# models to parent
BUS_PIPE = 'ipc:///tmp/bus.pipe'
# parent to bot caller
PARENT_PIPE = 'ipc:///tmp/parent_push.pipe'
# bot to parent caller
PARENT_PULL_PIPE = 'ipc:///tmp/parent_pull.pipe'


class ModelClient():
    """
    Client Process for individual models. Initialize the model
    and subscribe to channel to listen for updates
    """

    def __init__(self, model_name):
        #Process.__init__(self)
        self.model_name = model_name
        # select and initialize models
        if model_name == ModelID.HRED_REDDIT:
            logging.info("Initializing HRED Reddit")
            self.model = HRED_Wrapper(conf.hred['reddit_model_prefix'],
                                      conf.hred['reddit_dict_file'],
                                      ModelID.HRED_REDDIT)
        if model_name == ModelID.HRED_TWITTER:
            logging.info("Initializing HRED Twitter")
            self.model = HRED_Wrapper(conf.hred['twitter_model_prefix'],
                                      conf.hred['twitter_dict_file'],
                                      ModelID.HRED_TWITTER)
        if model_name == ModelID.FOLLOWUP_QA:
            logging.info("Initializing HRED Followup")
            self.model = HREDQA_Wrapper(conf.followup['model_prefix'],
                                        conf.followup['dict_file'],
                                        ModelID.FOLLOWUP_QA)
        if model_name == ModelID.DUAL_ENCODER:
            logging.info("Initializing Dual Encoder")
            self.model = Dual_Encoder_Wrapper(conf.de['reddit_model_prefix'],
                                              conf.de['reddit_data_file'],
                                              conf.de['reddit_dict_file'],
                                              ModelID.DUAL_ENCODER)
        if model_name == ModelID.DRQA:
            logging.info("Initializing DRQA")
            self.model = DRQA_Wrapper('', '', ModelID.DRQA)
        if model_name == ModelID.DUMB_QA:
            logging.info("Initializing DUMB QA")
            self.model = DumbQuestions_Wrapper(
                '', conf.dumb['dict_file'], ModelID.DUMB_QA)
        if model_name == ModelID.NQG:
            logging.info("Initializing NQG")
            self.model = NQG_Wrapper('', '', ModelID.NQG)
        if model_name == ModelID.ECHO:
            logging.info("Initializing Echo")
            self.model = Echo_Wrapper('', '', ModelID.ECHO)
        if model_name == ModelID.CAND_QA:
            logging.info("Initializing Candidate Questions")
            self.model = CandidateQuestions_Wrapper('',
                                                    conf.candidate['dict_file'],
                                                    ModelID.CAND_QA)
        if model_name == ModelID.TOPIC:
            logging.info("Initializing topic model")
            self.model = Topic_Wrapper('', '', '',conf.topic['dir_name'],
                    conf.topic['model_name'], conf.topic['top_k'])
        # message queue. This contains the responses generated by the model
        self.queue = Queue()
        self.is_running = True
        self.warmup()
        self.run()

    def warmup(self):
        """ Warm start the models before execution """
        if self.model_name != ModelID.DRQA:
            _, _ = self.model.get_response(1, 'test_statement', [])
        else:
            _, _ = self.model.get_response(1, 'Where is Daniel?', [], nlp(
                unicode('Daniel went to the kitchen')))

    def respond(self):
        """ Reply to the master on PUSH channel with the responses generated
        """
        socket = self.producer_context.socket(zmq.PUSH)
        socket.connect(BUS_PIPE)
        logging.info("Model {} push channel active".format(self.model_name))
        while self.is_running:
            msg = self.queue.get()
            socket.send_json(msg)
            self.queue.task_done()

    def act(self):
        """subscribe to master messages, and process them
        If msg contains key "control", process and exit
        """
        socket_b = self.consumer_context.socket(zmq.SUB)
        socket_b.connect(COMMAND_PIPE)
        socket_b.setsockopt(zmq.SUBSCRIBE, "user_response")
        # also subscribe to self topic
        socket_b.setsockopt(zmq.SUBSCRIBE, self.model_name)
        logging.info("Model {} subscribed to channels".format(self.model_name))
        while self.is_running:
            packet = socket_b.recv()
            topic, msg = demogrify(packet)
            if 'control' in msg:
                if msg['control'] == 'init':
                    logging.info(
                        "Model {} received init".format(self.model_name))
                if msg['control'] == 'preprocess':
                    if 'chat_id' in msg:
                        msg['user_id'] = msg['chat_id']
                    self.model.preprocess(**msg)
                if msg['control'] == 'exit':
                    logging.info("Received exit signal, model {}"
                                 .format(self.model_name))
                    self.is_running = False
            else:
                # assumes the msg will contain keyword parameters
                if 'chat_id' in msg:
                    msg['user_id'] = msg['chat_id']
                response, context = self.model.get_response(**msg)
                ## if blank response, do not push it in the channel
                if len(response) > 0:
                    resp_msg = {'text': response, 'context': context,
                                'model_name': self.model_name,
                                'chat_id': msg['chat_id'],
                                'chat_unique_id': msg['chat_unique_id']}
                    self.queue.put(resp_msg)

    def run(self):
        """Fire off the client"""
        self.producer_context = zmq.Context()
        self.consumer_context = zmq.Context()
        try:
            logging.info(
                "Starting {} response channel".format(self.model_name))
            resp_thread = Thread(target=self.respond)
            resp_thread.daemon = True
            resp_thread.start()
            logging.info("Starting {} act channel".format(self.model_name))
            act_thread = Thread(target=self.act)
            act_thread.daemon = True
            act_thread.start()
            while self.is_running:
                time.sleep(1)
            logging.info("Exiting {} client".format(self.model_name))
        except (KeyboardInterrupt, SystemExit):
            logging.info("Shutting down {} client".format(self.model_name))


# Initialize variables

article_text = {}     # map from chat_id to article text
chat_history = {}     # save the context / response pairs for a particular chat here
candidate_model = {}  # map from chat_id to a simple model for the article
article_nouns = {}    # map from chat_id to a list of nouns in the article
boring_count = {}     # number of times the user responded with short answer
policy_mode = Policy.NONE
job_queue = Queue()
response_queue = Queue()
model_responses = {}
# self.models = [self.hred_twitter, self.hred_reddit,
#               self.dual_enc, self.qa_hred, self.dumb_qa, self.drqa]
# modelIds = [ModelID.HRED_TWITTER, ModelID.HRED_REDDIT, ModelID.DUAL_ENCODER,
#                 ModelID.FOLLOWUP_QA, ModelID.DUMB_QA, ModelID.DRQA]
# Debugging
modelIds = [ModelID.ECHO, ModelID.CAND_QA, ModelID.HRED_TWITTER,
        ModelID.FOLLOWUP_QA, ModelID.DUMB_QA, ModelID.DRQA, ModelID.NQG, ModelID.TOPIC]
# modelIds = [ModelID.TOPIC]

# initialize only these models to catch the patterns
# TODO: or probably a pattern catcher?
dumb_qa_model = DumbQuestions_Wrapper(
    '', conf.dumb['dict_file'], ModelID.DUMB_QA)
# TODO: Load model to get the feature list to be used
# The model should be mentioned in the config file
model_args = []
with open(conf.ranker['model'],'rb') as fp:
    model_args = pkl.load(fp)
# Load the feature list
feature_list = model_args[0][-1]


def start_models():
    """ Warmup models in separate Process
    """
    # Warmup models
    topic = 'user_response'
    job = {'control': 'init', 'topic': topic}
    job_queue.put(job)


def stop_models():
    """ Send command to close the processes first
    """
    topic = 'user_response'
    job = {'control': 'exit', 'topic': topic}
    job_queue.put(job)


def submit_job(job_type='preprocess', to_model=ModelID.ALL,
               context=None, text='', chat_id='', chat_unique_id='',
               article=''):
    """ Submit Jobs to job queue, which will be consumed by the responder
    :job_type = preprocess / get_response / exit
    :to_model = all / specific model name
    """
    topic = 'user_response'
    if to_model != ModelID.ALL:
        topic = to_model
    # check if article is spacy instance
    if article and not isinstance(article, basestring):
        article = article.text
    if not context:
        context = []
    job = {'type': job_type, 'topic': topic, 'context': context,
            'text': text, 'chat_id': chat_id, 'chat_unique_id': chat_unique_id,
            'article_text': article}
    if job_type == 'preprocess' or job_type == 'exit':
        job['control'] = job_type

    print "Job : {}".format(json.dumps(job))
    job_queue.put(job)


def act():
    """ On getting a response from a model, add it to the
    chat_id model_responses list.
    Check if chat_id is present in model_responses, if not discard
    """
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(BUS_PIPE)
    logging.info("Child pull channel active")
    while True:
        packet = socket.recv_json()
        if packet['chat_unique_id'] in model_responses:
            # calculate the features here
            chat_id = packet['chat_id']
            context = packet['context']
            logging.info("Calculating features for model {}".format(packet['model_name']))
            # Context history is in chat_history[<chat_id>]
            context_till_now = chat_history[chat_id] + [context[-1]]
            packet['features'] = features.get(
                    article_text[chat_id], context_till_now,
                    packet['text'],
                    feature_list)
            logging.info("Done feature calculation for model {}".format(packet['model_name']))
            # Now store the packet in dict
            model_responses[packet['chat_unique_id']
                            ][packet['model_name']] = packet

            # TODO: Run approximator here and save the score in packet
        else:
            logging.info('Discarding message from model {} for chat id {}'.format(
                packet['model_name'], packet['chat_id']))


def responder():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(COMMAND_PIPE)
    logging.info("Child publish channel active")
    while True:
        job = job_queue.get()
        topic = job['topic']
        payload = mogrify(topic, job)
        socket.send(payload)


def consumer():
    """ ZMQ Consumer. Collect jobs from parent and run `get_response`
    """
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(PARENT_PULL_PIPE)
    logging.info("Parent pull channel active")
    is_running = True
    while is_running:
        msg = socket.recv_json()
        if 'control' in msg:
            if msg['control'] == 'exit':
                logging.info("Received exit command. Closing all models")
                stop_models()
                logging.info("Exiting")
                is_running = False
                sys.exit(0)
            if msg['control'] == 'clean':
                clean(msg['chat_id'])
        else:
            get_response(msg['chat_id'], msg['text'],
                         msg['context'], msg['allowed_model'])


def producer():
    """ ZMQ Response producer. Push response to bot callee.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(PARENT_PIPE)
    logging.info("Parent push channel active")
    while True:
        msg = response_queue.get()
        socket.send_json(msg)
        response_queue.task_done()


def clean(chat_id):
    article_text.pop(chat_id, None)
    candidate_model.pop(chat_id, None)
    article_nouns.pop(chat_id, None)
    boring_count.pop(chat_id, None)
    policy_mode = Policy.NONE


def strip_emojis(str):
    tokens = set(list(str))
    emojis = list(tokens.intersection(set(emoji.UNICODE_EMOJI.keys())))
    if len(emojis) > 0:
        text = ''.join(c for c in str if c not in emojis)
        emojis = ''.join(emojis)
        return text, emojis
    return str, None

# TODO: set allowed_model for model based debugging


def get_response(chat_id, text, context, allowed_model=None):
    # create a chat_id + unique ID candidate responses field
    # chat_unique_id is needed to uniquely determine the return
    # for each call
    chat_unique_id = str(chat_id) + '_' + str(uuid.uuid4())
    model_responses[chat_unique_id] = {}
    is_start = False

    # if text contains /start, don't add it to the context
    if '/start' in text:
        is_start = True
        # remove start token
        text = re.sub(r'\/start', '', text)
        # remove urls
        text = re.sub(r'https?:\/\/.*[\r\n]*',
                      '', text, flags=re.MULTILINE)
        # save the article for later use
        article_text[chat_id] = text
        article_nlp = nlp(unicode(text))
        # save all nouns from the article
        article_nouns[chat_id] = [
            p.text for p in article_nlp if p.pos_ == 'NOUN'
        ]

        # initialize bored count to 0 for this new chat
        boring_count[chat_id] = 0

        # initialize chat history
        chat_history[chat_id] = []

        # fire global preprocess call
        submit_job(job_type='preprocess',
                   article=article_text[chat_id],
                   chat_id=chat_id,
                   chat_unique_id=chat_unique_id)
        # fire candidate question and NQG
        submit_job(job_type='get_response',
                   to_model=ModelID.CAND_QA,
                   chat_id=chat_id,
                   chat_unique_id=chat_unique_id,
                   context=context,
                   text='')
        submit_job(job_type='get_response',
                   to_model=ModelID.NQG,
                   chat_id=chat_id,
                   chat_unique_id=chat_unique_id,
                   context=context,
                   text='')

    else:
        # fire global query
        if not allowed_model or allowed_model == ModelID.ALL:
            submit_job(job_type='get_response',
                       chat_id=chat_id,
                       chat_unique_id=chat_unique_id,
                       context=context,
                       text=text)
        else:
            submit_job(job_type='get_response',
                       to_model=allowed_model,
                       chat_id=chat_id,
                       chat_unique_id=chat_unique_id,
                       context=context,
                       text=text)
    # wait for responses to come in
    # if we have answer ready before the wait period, exit and return the answer
    done_processing = False
    wait_for = WAIT_TIME
    # response should be a dict of (response, context, model_name, policy_mode)
    response = {}
    # add feature list as another key of response
    done_features = set()
    while not done_processing and wait_for > 0:
        if is_start:
            if (ModelID.CAND_QA not in
                    model_responses[chat_unique_id]
                ) and (ModelID.NQG not in
                       model_responses[chat_unique_id]):
                continue
            else:
                # if found msg early, break
                done_processing = True
                break
        else:
            # if allowed model is not all, then wait for it to arrive
            # by elongating the wait_for time (useful for debugging certain
            # models)
            if allowed_model and allowed_model != ModelID.ALL:
                if allowed_model in model_responses[chat_unique_id]:
                    done_processing = True
                    break
                else:
                    wait_for += 1
            elif len(set(model_responses[chat_unique_id].keys())
                    .intersection(set(modelIds))) == len(modelIds):
                done_processing = True
                break
            # tick
        wait_for -= 1
        time.sleep(1)

    # got the responses, now choose which one to send.
    if is_start:
        # TODO: replace this with a proper choice / always NQG?
        choices = list(set([ModelID.CAND_QA, ModelID.NQG])
                .intersection(set(model_responses[chat_unique_id].keys())))
        selection = random.choice(choices)
        response = model_responses[chat_unique_id][selection]
        response['policyID'] = Policy.START
    else:
        # TODO: Replace this with ranking. Now using the optimal policy
        # check if allowed_model is set, then only reply from the allowed
        # model. This is done for debugging.
        # TODO: Probably remove this before final submission?
        if allowed_model and allowed_model != ModelID.ALL:
            response = model_responses[chat_unique_id][allowed_model]
            response['policyID'] = Policy.FIXED
        else:
            # if text contains emoji's, strip them
            text, emojis = strip_emojis(text)
            if emojis and len(text.strip()) < 1:
                # if text had only emoji, give back the emoji itself
                # NOTE: shouldn't we append the `resp` (in this case emoji)
                # to the context like everywhere else?
                response = {'response': emojis, 'context': context,
                            'model_name': 'emoji', 'policy': policy_mode}

            # if query falls under dumb questions, respond appropriately
            elif ModelID.DUMB_QA in model_responses[chat_unique_id]:
                logging.info("Matched dumb preset patterns")
                response = model_responses[
                    chat_unique_id][ModelID.DUMB_QA]
                response['policyID'] = Policy.FIXED
            elif ModelID.TOPIC in model_responses[chat_unique_id]:
                logging.info("Matched topic preset patterns")
                response = model_responses[
                    chat_unique_id][ModelID.TOPIC]
                response['policyID'] = Policy.FIXED
            elif '?' in text:
                # get list of common nouns between article and question
                common = list(set(article_nouns[chat_id]).intersection(
                    set(text.split(' '))))
                print 'common nouns between question and article:', common
                # if there is a common noun between question and article
                # select DrQA
                if len(common) > 0 and ModelID.DRQA in model_responses:
                    response = model_responses[
                        chat_unique_id][ModelID.DRQA]
                    response['policyID'] = Policy.FIXED

    if not response:
        # if text contains 2 words or less, add 1 to the bored count
        if len(text.strip().split()) <= 2:
            boring_count[chat_id] += 1
        # if user is bored, change the topic by asking a question
        # (only if that question is not asked before)
        if boring_count[chat_id] >= BORED_COUNT:
            response = model_responses[
                chat_unique_id][ModelID.CAND_QA]
            boring_count[chat_id] = 0  # reset bored count to 0

        # randomly decide a model to query to get a response:
        models = [ModelID.HRED_REDDIT, ModelID.HRED_TWITTER,
                  ModelID.DUAL_ENCODER, ModelID.NQG]
        if '?' in text:
            # if the user asked a question, also consider DrQA
            models.append(ModelID.DRQA)
        else:
            # if the user didn't ask a question, also consider hred-qa
            models.append(ModelID.FOLLOWUP_QA)

        available_models = list(set(model_responses[chat_unique_id]).intersection(models))
        if len(available_models) > 0:
            chosen_model = random.choice(available_models)
            response = model_responses[chat_unique_id][chosen_model]

        response['policyID'] = Policy.OPTIMAL

    # if still no response, then just send a random emoji
    if not response or 'text' not in response:
        logging.warn("Failure to obtain a response, using echo model")
        response = model_responses[chat_unique_id][ModelID.ECHO]
        response['policyID'] = Policy.FIXED


    # Now we have a response, so send it back to bot host
    # add user and response pair in chat_history
    chat_history[response['chat_id']].append(response['context'][-1])
    chat_history[response['chat_id']].append(response['text'])
    # remove the features because it is not JSON serializable
    del response['features']
    # Again use ZMQ, because lulz
    response_queue.put(response)
    # clean the unique chat ID
    del model_responses[chat_unique_id]


if __name__ == '__main__':
    """Run the main calling function:
            1. Initialize all the models
            2. Bot parent push channel `producer`
            3. Bot parent pull channel `consumer`
            4. Child models publish channel `responder`
            5. Child models pull channel `act`
    """
    # 1. Initializing the models
    mps = []
    mp_pool = Pool(processes=len(modelIds))
    for model in modelIds:
        #wx = ModelClient(model)
        #mps.append(wx)
        mp_pool.apply_async(ModelClient, args=(model,))
    #for mp in mps:
    #    mp.start()
    # 2. Parent -> Bot publish channel
    child_publish_thread = Thread(target=responder)
    child_publish_thread.daemon = True
    child_publish_thread.start()
    # 3. Bot -> Parent push channel
    child_pull_thread = Thread(target=act)
    child_pull_thread.daemon = True
    child_pull_thread.start()
    # 4. Parent -> Callee push channel
    parent_push_thread = Thread(target=producer)
    parent_push_thread.daemon = True
    parent_push_thread.start()
    # 5. Callee -> Parent pull channel
    parent_pull_thread = Thread(target=consumer)
    parent_pull_thread.daemon = True
    parent_pull_thread.start()

    # Model Init
    start_models()

    try:
        mp_pool.close()
        mp_pool.join()
        #for mp in mps:
        #    mp.join()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Sending shutdown signal to all models")
        stop_models()
        for mp in mps:
            mp.terminate()
        logging.info("Shutting down master")
