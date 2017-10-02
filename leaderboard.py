# Simple Script to view the current chat leaderboard

import pymongo
import argparse
from texttable import Texttable

log_db = None  # connect to main logging database
chat_db = None  # connect to client side chat database
PORT = 8091

client = pymongo.MongoClient("localhost", PORT)
db = client.convai
log_db = db.dialogs
chat_db = db.local


def get_top_users():
    local_chats = list(chat_db.find({}))
    user_dict = {}
    if len(local_chats) > 0:
        for local_chat in local_chats:
            dialogId = local_chat['dialogId']
            log_chats = list(log_db.find({'dialogId': dialogId}))
            assert len(log_chats) == 1
            user = ''
            user_id = ''
            for users in log_chats[0]['users']:
                if users['userType'] == 'ai.ipavlov.communication.TelegramChat':
                    user = users['username']
                    user_id = users['id']
            if user not in user_dict:
                user_dict[user] = {'user_id': user_id, 'chats': 0, 'turns': 0, 'max_turns': 0, 'min_turns': 99999,
                                   'average_quality': 0, 'average_breadth': 0, 'average_engagement': 0, 'average_upvotes': 0, 'average_downvotes': 0}
            turns = [ch for ch in log_chats[0]
                     ['thread'] if ch['userId'] == user_id]
            user_dict[user]['chats'] += 1
            user_dict[user]['turns'] += len(turns)
            user_dict[user]['max_turns'] = max(
                len(turns), user_dict[user]['max_turns'])
            user_dict[user]['min_turns'] = min(
                len(turns), user_dict[user]['min_turns'])
            evaluation = {}
            for evals in log_chats[0]['evaluation']:
                if evals['userId'] == user_id:
                    evaluation = evals
            av_div = 1
            if user_dict[user]['chats'] > 1:
                av_div = 2
            user_dict[user]['average_quality'] = (
                1.0 * (user_dict[user]['average_quality'] + evaluation['quality']) / av_div)
            user_dict[user]['average_breadth'] = (
                1.0 * (user_dict[user]['average_breadth'] + evaluation['breadth']) / av_div)
            user_dict[user]['average_engagement'] = (
                1.0 * (user_dict[user]['average_engagement'] + evaluation['engagement']) / av_div)
            bot_turns = [ch for ch in log_chats[0]
                         ['thread'] if ch['userId'] != user_id]
            upvotes = len([ch for ch in bot_turns if ch['evaluation'] == 2])
            downvotes = len([ch for ch in bot_turns if ch['evaluation'] == 1])
            user_dict[user]['average_upvotes'] = (
                1.0 * (user_dict[user]['average_upvotes'] + upvotes) / av_div)
            user_dict[user]['average_down'] = (
                1.0 * (user_dict[user]['average_downvotes'] + downvotes) / av_div)
    return user_dict


if __name__ == '__main__':
    user_dict = get_top_users()
    t = Texttable()
    rows = ['username'] + user_dict[user_dict.keys()[0]].keys()
    indv_rows = [[user] + [user_dict[user][p] for p in rows[1:]] for user in user_dict]
    rows = [rows]
    rows.extend(indv_rows)
    print rows
    t.add_rows(rows)
    print t.draw()
