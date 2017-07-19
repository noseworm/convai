"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import requests
import os
import json
import time
import random
import collections
import model_selection
import config
conf = config.get_config()
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)

MAX_CONTEXT = 3

mSelect = model_selection.ModelSelection()

class ConvAIRLLBot:

    def __init__(self):
        self.chat_id = None
        self.observation = None
        self.context = {} # keep contexts here

    def observe(self, m):
        if self.chat_id is None and m['message']['text'].startswith('/start '):
            self.chat_id = m['message']['chat']['id']
            self.context[self.chat_id] = collections.deque(maxlen=MAX_CONTEXT)
            logging.info("Start new chat #%s" % self.chat_id)
        elif self.chat_id is not None and m['message']['text'] == '/end':
            logging.info("End chat #%s" % self.chat_id)
            self.context[self.chat_id] = collections.deque(maxlen=MAX_CONTEXT)
            self.chat_id = None
            self.observation = None
            return

        if self.chat_id is None:
            logging.info("Dialog not started yet. Ignore message.")
        elif m['message']['chat']['id'] == self.chat_id:
            logging.info("Accept message as part of chat #%s" % self.chat_id)
            self.observation = m['message']['text']
            return self.observation
        else:
            logging.info("Multiple dialogues are not allowed. Ignore message.")

    def act(self):
        if self.chat_id is None:
            logging.info("Dialog not started yet. Do not act.")
            return

        message = {
            'chat_id': self.chat_id
        }

        # select from our models here
        text,context = mSelect.get_response(self.chat_id,self.observation,self.context[self.chat_id])
        self.context[self.chat_id] = context
        #texts = ['I love you!', 'Wow!', 'Really?', 'Nice!', 'Hi', 'Hello', '', '/end']
        #text = texts[random.randint(0, 7)]

        data = {}
        if text == '':
            logging.info("Decided to do not respond and wait for new message")
            return
        elif text == '/end':
            logging.info("Decided to finish chat %s" % self.chat_id)
            self.chat_id = None
            data['text'] = '/end'
            data['evaluation'] = {
                'quality': 0,
                'breadth': 0,
                'engagement': 0
            }
        else:
            logging.info("Decided to respond with text: %s" % text)
            data = {
                'text': text,
                'evaluation': 0
            }

        message['text'] = json.dumps(data)
        return message


def main():

    """
    !!!!!!! Put your bot id here !!!!!!!
    """
    BOT_ID = conf.test_bot_token

    if BOT_ID is None:
        raise Exception('You should enter your bot token/id!')

    BOT_URL = os.path.join('https://convaibot.herokuapp.com/', BOT_ID)

    bot = ConvAIRLLBot()
    print "loading models"
    mSelect.initialize_models()

    while True:
        logging.info("Get updates from server")
        res = requests.get(os.path.join(BOT_URL, 'getUpdates'))

        if res.status_code != 200:
            logging.info(res.text)
            res.raise_for_status()

        logging.info("Got %s new messages" % len(res.json()))
        for m in res.json():
            logging.info("Process message %s" % m)
            bot.observe(m)
            new_message = bot.act()
            if new_message is not None:
                logging.info("Send response to server.")
                res = requests.post(os.path.join(BOT_URL, 'sendMessage'),
                                    json=new_message,
                                    headers={'Content-Type': 'application/json'})
                if res.status_code != 200:
                    logging.info(res.text)
                    res.raise_for_status()
        logging.info("Sleep for 1 sec. before new try")
        time.sleep(1)


if __name__ == '__main__':
    main()
