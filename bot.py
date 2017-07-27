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
        self.ai = {}
        # self.context = {} # keep contexts here -> NEVER USED: contexts saved in self.ai[chat_id]['context']

    def observe(self, m):
	chat_id = m['message']['chat']['id']
        new_chat = False
        if chat_id not in self.ai:
            if m['message']['text'].startswith('/start '):
                self.ai[chat_id] = {}
                self.ai[chat_id]['chat_id'] = chat_id
                self.ai[chat_id]['observation'] = m['message']['text']
                self.ai[chat_id]['context'] = collections.deque(maxlen=MAX_CONTEXT)
                logging.info("Start new chat #%s" % self.chat_id)
                new_chat = True
            else:
                logging.info("chat not started yet. Ignore message")

        else:
	    if m['message']['text'] == '/end':
                logging.info("End chat #%s" % chat_id)
                del self.ai[chat_id]
            else:
                self.ai[chat_id]['observation'] = m['message']['text']
                logging.info("Accept message as part of chat #%s" % chat_id)
        return chat_id, new_chat

    def act(self, chat_id, new_chat, m):
        data = {}
        message = {
            'chat_id': chat_id
        }

        if chat_id not in self.ai:
	   if m['message']['chat']['id'] == chat_id and m['message']['text'] == '/end':
		logging.info("Decided to finish chat %s" % chat_id)
		data['text'] = '/end'
		data['evaluation'] = {  # let's have the best default value haha
			'quality': 5,
			'breadth': 5,
			'engagement': 5
	    	}
		message['text'] = json.dumps(data)
		return message
	   else:
		logging.info("Dialog not started yet. Do not act.")
                return

        if self.ai[chat_id]['observation'] is None:
            logging.info("No new messages for chat #%s. Do not act." % self.chat_id)
            return

        if new_chat:
            text = "Hello! I hope you're doing well. I am doing fantastic today! Let me go through the article real quick and we will start talking about it."
        else:
            # select from our models
            text, context = mSelect.get_response(chat_id,self.ai[chat_id]['observation'],self.ai[chat_id]['context'])
            self.ai[chat_id]['context'] = context
        #texts = ['I love you!', 'Wow!', 'Really?', 'Nice!', 'Hi', 'Hello', '', '/end']
        #text = texts[random.randint(0, 7)]

        if text == '':
            logging.info("Decided to do not respond and wait for new message")
            return
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
    BOT_ID = conf.bot_token

    if BOT_ID is None:
        raise Exception('You should enter your bot token/id!')

    BOT_URL = os.path.join(conf.bot_endpoint, BOT_ID)

    bot = ConvAIRLLBot()
    print "loading models"
    mSelect.initialize_models()  # should we start this in a new thread instead..?

    while True:
        try:
            time.sleep(1)
            logging.info("Get updates from server")
            res = requests.get(os.path.join(BOT_URL, 'getUpdates'))

            if res.status_code != 200:
                logging.info(res.text)
                res.raise_for_status()

            logging.info("Got %s new messages" % len(res.json()))
            for m in res.json():
                new_chat = True
                while new_chat:  # will become false once we call bot.observe(m) except the first time we talk to this person
                    logging.info("Process message %s" % m)
                    chat_id, new_chat = bot.observe(m) # return chat_id & if we started a new convo or not
                    new_message = bot.act(chat_id, new_chat, m) # pass chat_id, new_chat flag & message to act
                    if new_message is not None:
                        print new_message
                        logging.info("Send response to server.")
                        res = requests.post(os.path.join(BOT_URL, 'sendMessage'),
                                            json=new_message,
                                            headers={'Content-Type': 'application/json'})
                        if res.status_code != 200:
                            logging.info(res.text)
                            res.raise_for_status()

            logging.info("Sleep for 1 sec. before new try")
        except Exception as e:
            logging.error(e)


if __name__ == '__main__':
    main()
