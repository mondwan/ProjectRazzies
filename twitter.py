"""
File: twitter.py
Author: Me
Email: 0
Github: 0
Description: Try to fetch tweets about razzi awards from twitter
"""

# imports
import sys
import os
import urllib
import calendar
import json

import configparser
import tweepy


# Twitter objects
class MyTweet(object):
    """
    An object for capturing interested contents from parsing twitter's tweet
    """
    def __init__(self, arg):
        """
        Constructor

        @param arg dict
        {
          text: string
          timestamp: unix timestamp
          hashtags: array of hash string
          retweet_count: int
          favorite_count: int
          author_followers: int
          author_friends: int
          author_num_of_status: int
        }
        """
        self.data = arg

    @classmethod
    def parse(cls, film_name, ctx):
        """
        Parse context from tweepy's api

        @param film_name string
        @param ctx dict
        @return Class::MyTweet
        """
        arg = {}
        arg['film_name'] = film_name
        arg['text'] = ctx.text
        arg['timestamp'] = calendar.timegm(ctx.created_at.utctimetuple())
        arg['hashtags'] = [tag['text'] for tag in ctx.entities['hashtags']]
        arg['retweet_count'] = ctx.retweet_count
        arg['favorite_count'] = ctx._json['favorite_count']
        arg['author_followers'] = ctx.author.followers_count
        arg['author_friends'] = ctx.author.friends_count
        arg['author_num_of_status'] = ctx.author.statuses_count

        return cls(arg)


def save_to_json(film_name, tweets):
    """
    Save given tweets to a json file inside twitter_data folder

    @param film_name string
    @param tweets Array of Class::MyTweet
    """
    data = [t.data for t in tweets]
    json_data = json.dumps(data)
    with open(
        os.path.join('.', 'twitter_data', '%s.json' % film_name),
        'w'
    ) as f:
        json.dump(json_data, f)

# Read configuration
cfg = configparser.ConfigParser()
cfg.read('config.ini')

# setting up constants
consumer_key = cfg['twitter']['consumer_key']
consumer_secret = cfg['twitter']['consumer_secret']
access_token = cfg['twitter']['access_token']
access_secret = cfg['twitter']['access_secret']
max_status_per_film = int(cfg['twitter']['max_status_per_film'])
film_names = cfg['film']['names'].split('\n')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
count = {}
my_tweets = []

for film_name in film_names:
    print 'film_name: %s' % film_name
    # Definition for search API
    # https://dev.twitter.com/rest/reference/get/search/tweets
    try:
        for status in tweepy.Cursor(
            api.search,
            lang='en',
            q=urllib.quote('"%s"' % film_name)
        ).items(limit=max_status_per_film):
            t = MyTweet.parse(film_name, status)
            my_tweets.append(t)
            print '%d Tweets of [%s] have been proccess' % (
                len(my_tweets), film_name
            )
    except tweepy.error.TweepError, e:
        print e
    finally:
        save_to_json(film_name, my_tweets)
        count[film_name] = len(my_tweets)
        del my_tweets[:]

# Report
with open(os.path.join('.', 'twitter_data', 'report.txt'), 'w') as f:
    for (fn, cnt) in count.items():
        f.write('Film name: %s\n' % fn)
        f.write('Tweet count: %d\n' % cnt)
