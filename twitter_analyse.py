"""
File: twitter_analyse.py
Author: Me
Email: 0
Github: 0
Description: Analyse tweets. For the detail, please refer to the document
```twitter_analyse.notes```
"""

# System lib
from __future__ import division
import json
import os
from math import log

# 3-rd party lib
# import nltk
from nltk.classify import NaiveBayesClassifier
from textblob import TextBlob

# Constants
TWEET_DIR = os.path.join('.', 'twitter_data')
OSCAR_DIR = os.path.join(TWEET_DIR, 'oscar')
RAZZIES_DIR = os.path.join(TWEET_DIR, 'razzies')
PREDICT_DIR = os.path.join(TWEET_DIR, 'proof', 'razzies')


def attribute_to_characteristic(tweet):
    """
    Extract attributes from a tweet and form a characteristic of a tweet

    @param tweet dict
    @return dict
      Charateristic of a tweet
    """
    ret = {}
    text = tweet['text']
    retweets = tweet['retweet_count']
    favorites = tweet['favorite_count']
    followers = tweet['author_followers']
    friends = tweet['author_friends']
    publishes = tweet['author_num_of_status']
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    ret['scaled_polarity'] = calculate_scaled_polarity(
        polarity,
        int(retweets),
        int(favorites),
        int(followers),
        int(friends),
        int(publishes)
    )

    # print 'p=%.2f re=%d fav=%d, fol=%d, fd=%d, pub=%d' % (
    #     polarity, retweets, favorites, followers, friends, publishes
    # )

    return ret


def calculate_scaled_polarity(
        polarity, retweets, favorites, followers, friends, publishes):
    """
    Return a scaled polarity for a tweet

    @param polarity float
    @param retweets int
    @param favorites int
    @param followers int
    @param friends int
    @param publishes int

    @return float
    """
    # Avoid zero case and negative value
    retweets = retweets if retweets > 0 else 1
    favorites = favorites if favorites > 0 else 1
    followers = followers if followers > 0 else 1
    friends = friends if friends > 0 else 1
    publishes = publishes if publishes > 0 else 1
    # Entropy
    ret = polarity * \
        (
            log(retweets, 2) +
            log(favorites, 2) +
            log(followers, 2) +
            log(friends, 2) +
            log(publishes, 2)
        )

    return round(ret, 2)


def tweets2film(tweet_characteristics):
    """
    Aggreate tweet's characteristics to form a film's characteristics

    @param tweet_characteristics list of dict

    @return dict
      characteristics of a film
    """

    ret = {
        'scaled_polarity': 0
    }

    summation = 0
    num_of_tweet = len(tweet_characteristics)
    for c in tweet_characteristics:
        summation += c['scaled_polarity']

    ret['scaled_polarity'] = round(summation / num_of_tweet, 2)

    return ret

features = []

for my_dir in [OSCAR_DIR, RAZZIES_DIR]:
    label = os.path.basename(my_dir)
    for fn in os.listdir(my_dir):
        path = os.path.join(my_dir, fn)
        film_name = os.path.splitext(fn)[0]
        # print 'dir=%s, film_name=%s, path=%s' % (my_dir, film_name, path)

        with open(path, 'r') as f:
            tweets = json.load(f)
            tweets = json.loads(tweets)

        tweet_characteristics = []
        for tweet in tweets:
            # Per tweet analyze
            characteristic = attribute_to_characteristic(tweet)
            tweet_characteristics.append(characteristic)

        film_characteristic = tweets2film(tweet_characteristics)
        print 'film: |%s|' % film_name
        print film_characteristic
        feature = (film_characteristic, label)
        features.append(feature)

# Train the classifier
classifier = NaiveBayesClassifier.train(features)

print 'Predicting...'

# Predict the film
for my_dir in [PREDICT_DIR]:
    for fn in os.listdir(my_dir):
        path = os.path.join(my_dir, fn)
        film_name = os.path.splitext(fn)[0]

        with open(path, 'r') as f:
            tweets = json.load(f)
            tweets = json.loads(tweets)

        tweet_characteristics = []
        for tweet in tweets:
            # Per tweet analyze
            characteristic = attribute_to_characteristic(tweet)
            tweet_characteristics.append(characteristic)

        film_characteristic = tweets2film(tweet_characteristics)
        print film_characteristic
        result = classifier.classify(film_characteristic)
        print 'film: |%s| predict: |%s|' % (film_name, result)

# classifier.show_most_informative_features()
