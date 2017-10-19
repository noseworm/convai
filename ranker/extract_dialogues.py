import json
import pymongo
import argparse
import cPickle as pkl
import time


def query_db():
    PORT = 8091
    CLIENT = '132.206.3.23'

    client = pymongo.MongoClient(CLIENT, PORT)
    db = client.convai
    log_db = db.dialogs
    chat_db = db.local

    local_chats = list(chat_db.find({}))

    return local_chats


def valid_chat(usr_turns, bot_turns, k=2):
    # Check that user sent at least k messages and bot replied with 2 more messages
    long_enough = len(usr_turns) >= k and len(bot_turns) == 2+len(usr_turns)

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
    NOTE: make sure the list is ordered by article!! ie: [article1, ..., article1, article2, ..., article2, article3, ..., ..., article_n]
    """
    formated_data = []

    for dialog in json_data:
        print dialog
        # get user id for this chat
        uid = None
        for usr in dialog['users']:
            if usr['userType'] == 'ai.ipavlov.communication.TelegramChat':
                uid = usr['id']
        if uid is None:
            print "Warning: no user of type ai.ipavlov.communication.TelegramChat found, skipping this chat!"
            continue

        # get user_turns and bot_turns
        usr_turns = [msg for msg in dialog['thread'] if msg['userId'] == uid]
        bot_turns = [msg for msg in dialog['thread'] if msg['userId'] != uid]

        if valid_chat(usr_turns, bot_turns):
            # get article text for that conversation
            article = dialog['context']
            # get full evaluation for that conversation
            full_eval = -1.0
            for evl in dialog['evaluation']:
                if evl['userId'] == uid:
                    full_eval = (evl['quality'] + evl['breadth'] + evl['engagement']) / 3.0
            if full_eval == -1.0:
                print "Warning: no full evaluation found for uid %s, skipping this chat" % uid
                continue

            # TODO: go through bot_turns and usr_turns to create a list of (context, candidate, score) triples


    return formated_data


def main():
    parser = argparse.ArgumentParser(description='Create pickle data for training, testing ranker neural net')
    parser.add_argument('--json_file', default=None, help='optional json file to reformat into training format. if None, will query databse.')
    args = parser.parse_args()
    print args

    print "\nGet json data:",
    if args.json_file:
        print "from file %s..." % args.json_file
        with open(args.json_file, 'r') as handle:
            json_data = json.loads(' '.join(handle.readlines()))

    else:
        print "from database..."
        json_data = query_db()
    print "Got %d dialogues" % len(json_data)

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

