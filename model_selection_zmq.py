# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# ZMQ version. Here, all models are initialized in a separate thread
# Then the main process sends out next response and context as a PUB msg,
# which every model_client listens in SUB
# Then all models calculate the response and send it to parent using PUSH-PULL
# NN model selection algorithm can therefore work in the parent queue
# to calculate the score of all incoming msgs
# Set a hard time limit to discard the messages which are slow
# N.B. ZMQ Context has to be initialized from top process level, then boiled down
import sys
import config
conf = config.get_config()
from models.wrapper import HRED_Wrapper, Dual_Encoder_Wrapper, HREDQA_Wrapper,
    CandidateQuestions_Wrapper, DumbQuestions_Wrapper, DRQA_Wrapper, NQG_Wrapper, Echo_Wrapper
import random
import copy
import spacy
import re
import time
import emoji
import zmq
import json
from Queue import Queue
from threading import Thread
from multiprocessing import Process
import uuid
import logging
import traceback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)


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


class ModelID:
    DRQA = 'drqa'
    DUAL_ENCODER = 'de'
    HRED_REDDIT = 'hred-reddit'
    HRED_TWITTER = 'hred-twitter'
    DUMB_QA = 'dumb_qa'
    NQG = 'nqg'
    FOLLOWUP_QA = 'followup_qa'
    CAND_QA = 'candidate_question'
    ECHO = 'echo_model'  # just for debugging purposes


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


class ModelClient(Process):
    """
    Client Process for individual models. Initialize the model
    and subscribe to channel to listen for updates
    """

    def __init__(self, model_name):
        Process.__init__(self)
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
                                                    ModelID.CAND_QA),
        # message queue. This contains the responses generated by the model
        self.queue = Queue()
        self.warmup()

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
        while True:
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
        while True:
            packet = socket_b.recv()
            topic, msg = demogrify(packet)
            if 'control' in msg:
                if msg['control'] == 'init':
                    logging.info(
                        "Model {} received init".format(self.model_name))
                if msg['control'] == 'preprocess':
                    self.model.preprocess(**msg)
                if msg['control'] == 'exit':
                    logging.info("Received exit signal, model {}"
                                 .format(self.model_name))
                    sys.exit(0)
            else:
                # assumes the msg will contain keyword parameters
                response, context = self.model.get_response(**msg)
                resp_msg = {'response': response, 'context': context,
                            'model_name': self.model_name,
                            'chat_id': msg['chat_id']}
                self.queue.put(resp_msg)

    def run(self):
        """Fire off the client"""
        self.producer_context = zmq.Context()
        self.consumer_context = zmq.Context()
        try:
            logging.info("Starting {} response channel".format(self.model_name))
            resp_thread = Thread(target=self.respond)
            resp_thread.daemon = True
            resp_thread.start()
            logging.info("Starting {} act channel".format(self.model_name))
            act_thread = Thread(target=self.act)
            act_thread.daemon = True
            act_thread.start()
            while True:
                time.sleep(100)
        except (KeyboardInterrupt, SystemExit):
            print "Shutting down {} client".format(self.model_name)
            raise


class ModelSelection(object):

    def __init__(self):
        self.article_text = {}     # map from chat_id to article text
        self.candidate_model = {}  # map from chat_id to a simple model for the article
        self.article_nouns = {}    # map from chat_id to a list of nouns in the article
        self.boring_count = {}     # number of times the user responded with short answer
        self.policy_mode = Policy.NONE
        self.job_queue = Queue()
        self.response_queue = Queue()
        self.model_responses = {}
        # self.models = [self.hred_twitter, self.hred_reddit,
        #               self.dual_enc, self.qa_hred, self.dumb_qa, self.drqa]
        # self.modelIds = [ModelID.HRED_TWITTER, ModelID.HRED_REDDIT, ModelID.DUAL_ENCODER,
        #                 ModelID.FOLLOWUP_QA, ModelID.DUMB_QA, ModelID.DRQA]
        # Debugging
        self.modelIds = [ModelID.ECHO]
        # initialize only this model to catch the patterns
        self.dumb_qa_model = DumbQuestions_Wrapper(
            '', conf.dumb['dict_file'], ModelID.DUMB_QA)
        # run
        self.run()

    def init_model(self, model_name):
        return ModelClient(model_name)

    def start_models(self):
        """ Start models in separate Process
        """
        model_processes = [ModelClient(model) for model in self.modelIds]
        # daemonize and start
        for model_p in model_processes:
            model_p.start()
        # Warmup models
        topic = 'user_response'
        job = {'control': 'init', 'topic': topic}
        self.job_queue.put(job)
        return model_processes

    def stop_models(self):
        """ Send command to close the processes first
        """
        topic = 'user_response'
        job = {'control': 'exit', 'topic': topic}
        self.job_queue.put(job)

    def submit_job(self, job_type='preprocess', to_model='all',
                   context=None, text=None, chat_id=None, article=None):
        """ Submit Jobs to job queue, which will be consumed by the responder
        :job_type = preprocess / get_response / exit
        :to_model = all / specific model name
        """
        topic = 'user_response'
        if to_model != 'all':
            topic = to_model
        # check if article is spacy instance
        if article and not isinstance(article, basestring):
            article = article.text
        job = {'type': job_type, 'topic': topic, 'context': context,
               'text': text, 'chat_id': chat_id, 'article': article}
        self.job_queue.put(job)

    def act(self):
        """ On getting a response from a model, add it to the
        chat_id model_responses list.
        Check if chat_id is present in self.model_responses, if not discard
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(BUS_PIPE)
        logging.info("Child pull channel active")
        while True:
            packet = socket.recv_json()
            if packet['chat_id'] in self.model_responses:
                self.model_responses[packet['chat_id']
                                     ][packet['model_name']] = packet
            else:
                logging.info('Discarding message from model {} for chat id {}'.format(
                    packet['model_name'], packet['chat_id']))

    def responder(self):
        """ ZMQ Responder. Publish jobs to all clients
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind(COMMAND_PIPE)
        logging.info("Child publish channel active")
        while True:
            job = self.job_queue.get()
            topic = job['topic']
            payload = mogrify(topic, job)
            socket.send(payload)

    def consumer(self):
        """ ZMQ Consumer. Collect jobs from parent and run `get_response`
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.connect(PARENT_PULL_PIPE)
        logging.info("Parent pull channel active")
        while True:
            msg = socket.recv_json()
            if 'control' in msg:
                if msg['control'] == 'exit':
                    logging.info("Received exit command. Closing all models")
                    self.stop_models()
                    logging.info("Exiting")
                    sys.exit(0)
                if msg['control'] == 'clean':
                    self.clean(msg['chat_id'])
            else:
                self.get_response(msg['chat_id'], msg['text'],
                                  msg['context'], msg['allowed_model'])

    def producer(self):
        """ ZMQ Response producer. Push response to bot callee.
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect(PARENT_PIPE)
        logging.info("Parent push channel active")
        while True:
            msg = self.model_responses.get()
            socket.send_json(msg)
            self.model_responses.task_done()

    def clean(self, chat_id):
        del self.article_text[chat_id]
        del self.candidate_model[chat_id]
        del self.article_nouns[chat_id]
        del self.boring_count[chat_id]
        self.policy_mode = Policy.NONE

    def strip_emojis(self, str):
        tokens = set(list(str))
        emojis = list(tokens.intersection(set(emoji.UNICODE_EMOJI.keys())))
        if len(emojis) > 0:
            text = ''.join(c for c in str if c not in emojis)
            emojis = ''.join(emojis)
            return text, emojis
        return str, None

    # TODO: set allowed_model for model based debugging
    def get_response(self, chat_id, text, context, allowed_model=None):
        # create a chat_id + unique ID candidate responses field
        chat_unique_id = chat_id + '_' + uuid.uuid4()
        self.model_responses[chat_unique_id] = {}
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
            self.article_text[chat_id] = nlp(unicode(text))
            # save all nouns from the article
            self.article_nouns[chat_id] = [
                p.text for p in self.article_text[chat_id] if p.pos_ == 'NOUN'
            ]

            # initialize bored count to 0 for this new chat
            self.boring_count[chat_id] = 0

            # fire global preprocess call
            self.submit_job(job_type='preprocess',
                            article=self.article_text[chat_id],
                            chat_id=chat_id)
            # fire candidate question and NQG
            self.submit_job(job_type='get_response',
                            to_model=ModelID.CAND_QA,
                            chat_id=chat_id,
                            context=context,
                            text=text)
            self.submit_job(job_type='get_response',
                            to_model=ModelID.NQG,
                            chat_id=chat_id,
                            context=context,
                            text=text)

        else:
            # fire global query
            self.submit_job(job_type='get_response',
                            chat_id=chat_id,
                            context=context,
                            text=text)

        # wait for responses to come in
        # if we have answer ready before the wait period, exit and return the answer
        done_processing = False
        wait_for = WAIT_TIME
        # response should be a tuple of (response, context, model_name, policy_mode)
        response = {}
        while not done_processing and wait_for > 0:
            if is_start:
                if (ModelID.CAND_QA not in
                        self.model_responses[chat_unique_id]
                    ) and (ModelID.NQG not in
                           self.model_responses[chat_unique_id]):
                    continue
                else:
                    # if found msg early, break
                    done_processing = True
                    break
            else:
                # TODO: Run NN Model selection approximator here on whichever
                # responses are coming in
                if len(set(self.model_responses[chat_unique_id].keys())
                        .intersection(set(self.modelIds))) == len(self.modelIds):
                    done_processing = True
                    break
                # tick
            wait_for -= 1
            time.sleep(1)

        # got the responses, now choose which one to send.
        if is_start:
            # TODO: replace this with a proper choice / always NQG?
            choices = [ModelID.CAND_QA, ModelID.NQG]
            selection = random.choice(choices)
            response = self.model_responses[chat_unique_id][selection]
            response['policy'] = Policy.START
        else:
            # TODO: Replace this with ranking. Now using the optimal policy

            # if text contains emoji's, strip them
            text, emojis = self.strip_emojis(text)
            if emojis and len(text.strip()) < 1:
                # if text had only emoji, give back the emoji itself
                # NOTE: shouldn't we append the `resp` (in this case emoji)
                # to the context like everywhere else?
                response = {'response': emojis, 'context': context,
                            'model_name': 'emoji', 'policy': self.policy_mode}

            # if query falls under dumb questions, respond appropriately
            elif self.dumb_qa.isMatch(text) and Model.DUMB_QA in self.model_responses:
                response = self.model_responses[
                    chat_unique_id][ModelID.DUMB_QA]
            elif '?' in text:
                # get list of common nouns between article and question
                common = list(set(self.article_nouns[chat_id]).intersection(
                    set(text.split(' '))))
                print 'common nouns between question and article:', common
                # if there is a common noun between question and article
                # select DrQA
                if len(common) > 0 and Model.DRQA in self.model_responses:
                    response = self.model_responses[
                        chat_unique_id][ModelID.DRQA]

        if not response:
            # if text contains 2 words or less, add 1 to the bored count
            if len(text.strip().split()) <= 2:
                self.boring_count[chat_id] += 1
            # if user is bored, change the topic by asking a question
            # (only if that question is not asked before)
            if self.boring_count[chat_id] >= BORED_COUNT:
                response = self.model_responses[
                    chat_unique_id][ModelID.CAND_QA]
                self.boring_count[chat_id] = 0  # reset bored count to 0

            # randomly decide a model to query to get a response:
            models = [ModelID.HRED_REDDIT, ModelID.HRED_TWITTER,
                      ModelID.DUAL_ENCODER, ModelID.NQG]
            if '?' in text:
                # if the user asked a question, also consider DrQA
                models.append(ModelID.DRQA)
            else:
                # if the user didn't ask a question, also consider hred-qa
                models.append(ModelID.FOLLOWUP_QA)

            chosen_model = random.choice(models)
            while (chosen_model not in self.model_responses[chat_unique_id]) and len(models) > 0:
                models.remove(chosen_model)
                if len(models) > 0:
                    chosen_model = random.choice(models)

            if chosen_model in self.model_responses[chat_unique_id]:
                response = self.model_responses[chat_unique_id][chosen_model]

            response['policy'] = Policy.OPTIMAL

        # Now we have a response, so send it back to bot host
        # Again use ZMQ, because lulz
        self.response_queue.put(response)
        # clean the unique chat ID
        del self.model_responses[chat_unique_id]

    def run(self):
        """Function to run all the functions:
                1. Spin new process for each model
                2. Bot parent push channel `producer`
                3. Bot parent pull channel `consumer`
                4. Child models publish channel `responder`
                5. Child models pull channel `act`
        """
        model_processes = self.start_models()
        child_publish_thread = Thread(target=self.responder)
        child_publish_thread.daemon = True
        child_publish_thread.start()
        child_pull_thread = Thread(target=self.act)
        child_pull_thread.daemon = True
        child_pull_thread.start()
        parent_push_thread = Thread(target=self.producer)
        parent_push_thread.daemon = True
        parent_push_thread.start()
        parent_pull_thread = Thread(target=self.consumer)
        parent_pull_thread.daemon = True
        parent_pull_thread.start()

        try:
            while True:
                time.sleep(100)
        except (KeyboardInterrupt, SystemExit):
            logging.info("Shutting down all models")
            for model_p in model_processes:
                model_p.terminate()
            raise
            logging.info("Shutting down master")


if __name__ == '__main__':

    ModelSelection()
