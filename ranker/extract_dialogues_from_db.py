import json
import pymongo
import argparse
import cPickle as pkl
import time
from collections import defaultdict
import copy


score_map = {0: 0.,
             1: -1.,
             2: +1.}

def query_db():
    """
    """
    PORT = 8091
    CLIENT = '132.206.3.23'

    client = pymongo.MongoClient(CLIENT, PORT)
    db = client.convai

    chats = []

    for d_local in list(db.local.find({})):
        d_id = d_local['dialogId']
        d_servr = list(db.dialogs.find({'dialogId': d_id}))
        if len(d_servr) > 1:
            print "Error: two dialogs with same id (%s)!" % did
            continue
        elif len(d_servr) < 1:
            print "Warning: no dialog found in server for id %s." % d_id
            continue
        d_servr = d_servr[0]
        data = copy.deepcopy(d_servr)

        # map from local msg text to local msg object
        local_msgs = dict(
            [(msg['text'], msg) for msg in d_local['logs'] if msg['text'] is not None]
        )
        # list of messages on the server: the order we want to keep
        servr_msgs = [msg for msg in d_servr['thread'] if msg['text'] is not None]

        # n_msgs = min(len(local_msgs), len(servr_msgs))
        for msg in servr_msgs:
            text = msg['text']
            if text not in local_msgs:
                print "[%s] Warning: msg not in local server: %s" % (d_id, msg['text'])
                model = 'none'
                policy = -1
            else:
                model = local_msgs[text].get('model_name', 'human_from_db')
                policy = local_msgs[text].get('policyID', -1)
            msg['model'] = model
            msg['policy'] = policy

        data['thread'] = servr_msgs
        chats.append(data)

    return chats

def valid_chat(usr_turns, bot_turns, k=2):
    # Check that user sent at least k messages and bot replied with 2 more messages
    long_enough = len(usr_turns) >= k and len(bot_turns) >= k

    print "bot:%d %% usr:%d" % (len(bot_turns), len(usr_turns))
    valid_flow = len(bot_turns) == len(usr_turns) + 2  # normal flow: bot - bot - (usr - bot)*n
    early_stop = len(bot_turns) == len(usr_turns) + 1  # user sent /end before the bot reply or sent two msg in a row during the conversation
    usr_sent_more_than_1msg == len(bot_turns) <= len(usr_turns)  # usr sent more than 1 message before the bot had time to reply

    ## TODO: Check for bad language
    polite = True
    
    ## Check that user voted at least 95% of all bot messages
    # novote = filter(lambda turn: turn['evaluation']==0, bot_turns)
    # voted = float(len(novote)) / len(bot_turns) < 0.15  # voted at least 95% of all bot turns
    voted = True

    return long_enough and polite and voted


def reformat(json_data):
    """
    Create a list of dictionaries of the form {'article':<str>, 'context':<list of str>, 'candidate':<str>, 'r':<-1,0,1>, 'R':<0-5>}
    TODO: make sure the list is ordered by article!! ie: [article1, ..., article1, article2, ..., article2, article3, ..., ..., article_n]
    """
    formated_data = []

    for dialog in json_data:
        # get the bot id for this chat if there is one
        bid = None
        for usr in dialog['users']:
            if usr['userType'] == 'Bot':
               bid = usr['id']
        
        if bid is None:
            print "No bot in this chat (%s)! consider both users as potential bots!" % ([u['userType'] for u in dialog['users']], )
            both_human = True
        else:
            both_human = False

        # get user_turns and bot_turns
        usr_turns = [msg for msg in dialog['thread'] if msg['userId'] == uid]
        bot_turns = [msg for msg in dialog['thread'] if msg['userId'] != uid]

        if valid_chat(usr_turns, bot_turns):
            # get article text for that conversation
            article = dialog['context']
            # get full evaluation for that conversation
            full_evals = []
            for evl in dialog['evaluation']:
                if evl['userId'] != bid:
                    full_evals.append( (2.0*evl['quality'] + 1.0*evl['breadth'] + 1.0*evl['engagement']) / 4.0 )
            if len(full_eval) == 0:
                print "Warning: no full evaluation found for this conversation, skipping it"
                continue

            # Go through conversation to create a list of (article, context, candidate, score, reward) instances
            context = []
            last_msg_is_from_user = False
            for msg_idx, msg in enumerate(dialog['thread']):
                # if user started the conversation and not the bot there was a pb. skip this one
                # if msg_idx == 0 and msg['userId'] == uid:
                #     print "Warning: user started the conversation! Skipping this chat"
                #     break
                # if begining of converesation, just fill in the context
                if len(context) == 0:
                    context.append(msg['text'])
                    continue
                # if the user talked: add message to context
                if msg['userId'] == uid:
                    if last_msg_is_from_user:
                        context[-1] = context[-1]+' '+msg['text']
                    else:
                        context.append(msg['text'])
                    last_msg_is_from_user = True
                # if the bot talked:
                else:
                    # create (article, context, candidate, score, reward) instance
                    score = score_map[int(msg['evaluation'])]
                    formated_data.append({
                        'article':article, 'context': context, 'candidate': msg['text'], 'r': score, 'R': full_eval
                    })
                    # add bot response to context now
                    context.append(msg['text'])
                    last_msg_is_from_user = False

    return formated_data


def main():
    parser = argparse.ArgumentParser(description='Create pickle data for training, testing ranker neural net')
    parser.add_argument('--voted_only', action='store_true', help='consider only voted messages')
    args = parser.parse_args()
    print args

    print "\nGet json data from database..."
    json_data = query_db()
    print "Got %d dialogues" % len(json_data)

    print json_data[0]

    # extract array of dictionaries of the form {'article':<str>, 'context':<list of str>, 'candidate':<str>, 'r':<-1,0,1>, 'R':<0-5>}
    print "\nReformat dialogues into list of training examples..."
    full_data = reformat(json_data)
    print "Got %d examples" % len(full_data)
    
    print "\nSaving to pkl file..."
    with open('full_data_%s.pkl' % str(time.time()), 'wb') as handle:
        pkl.dump(full_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print "done."


if __name__ == '__main__':
    main()

