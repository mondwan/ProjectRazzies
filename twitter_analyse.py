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
import numpy
import random
import re

# 3-rd party lib
import nltk
from nltk.classify import NaiveBayesClassifier
from textblob import TextBlob
from nltk.corpus import movie_reviews
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer


# Constants
TWEET_DIR = os.path.join('.', 'twitter_data')
OSCAR_DIR = os.path.join(TWEET_DIR, 'oscar')
RAZZIES_DIR = os.path.join(TWEET_DIR, 'razzies')
PREDICT_DIR = os.path.join(TWEET_DIR, 'proof')
# PREDICT_OSCAR_DIR = os.path.join(PREDICT_DIR, 'oscar')
# PREDICT_RAZZIES_DIR = os.path.join(PREDICT_DIR, 'razzies')

#for stem and filter useless info in corpus
st = LancasterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
regex = re.compile(r'\w+\w+')

#getting nltk provided corpus for pos and neg movie reviews
#use train the classifier for classifing pos and neg tweets later
documents =[]
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		wordL = movie_reviews.words(fileid)
		wordF = filter(lambda i: regex.search(i), wordL) #filter single char and punctuation
		stWL = [st.stem(x) for x in wordF] #stem all words
		documents += [(stWL,category)]


random.shuffle(documents)


#find the most freq words from nltk provided movie reviews words pool
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:5000] #try to use the most freq 5000 words
word_features_filtered = filter(lambda j: regex.search(j), word_features) #filter single char and punctuation
word_stem_feat =  [st.stem(z) for z in word_features_filtered] 


#features extractor function
def document_features(document):
	document_words = set(document)
	features = {}
	for word in word_stem_feat:
		features['contains(%s)'% word] = (word in document_words)
		#if features['contains(%s)'% word]:
		#	print word + ' true'
	return features


featuresets = [(document_features(d), c) for (d,c) in documents]
classifier_pos_neg = nltk.NaiveBayesClassifier.train(featuresets[100:])


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
    textE = text.encode('utf8')
    textD = textE.decode('utf8','ignore')
    textF = filter(lambda i: regex.search(i), textD) #filter single char and punctuation
    textS = [st.stem(x) for x in textF] #stem words
    test_set = document_features(textS)

    rv = classifier_pos_neg.classify(test_set) #classify the tweet to pos or neg
 #   if rv != 'neg':
 #    print rv
  

    ret['scaled_polarity'] = calculate_scaled_polarity(
        polarity,
        int(retweets),
        int(favorites),
        int(followers),
        int(friends),
        int(publishes)
    )
    ret['retweets'] = retweets
    ret['favorites'] = favorites
    ret['followers'] = followers
    ret['friends'] = friends
    ret['publishes'] = publishes
    ret['polarity'] = polarity
    ret['PN_tag'] = rv

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

    ret = {}

    retweets_data = []
    favorites_data = []
    polarities_data = []
    friends_data = []
    followers_data = []
    tag_data = []
    
    count_tweets = len(tweet_characteristics)

    for t in tweet_characteristics:
        retweets_data.append(t['retweets'])
        favorites_data.append(t['favorites'])
        polarities_data.append(t['polarity'])
        friends_data.append(t['friends'])
        followers_data.append(t['followers'])
	tag_data.append(t['PN_tag'])

    retweets = numpy.array(retweets_data)
    favorites = numpy.array(favorites_data)
    polarities = numpy.array(polarities_data)
    friends = numpy.array(friends_data)
    followers = numpy.array(followers_data)
    PN_tag = 'pos'
    if tag_data.count('neg') > tag_data.count('pos'):
	PN_tag = 'neg'
	#print 'it is neg...'
     

    for data_set in [
        ('retweets', retweets),
        ('favorites', favorites),
        ('polarities', polarities),
        ('friends', friends),
        ('followers', followers)
    ]:
        data_name = data_set[0]
        data_list = data_set[1]
      #  print '|%s| sd: %f mean: %f min: %d max: %d' % (
            data_name,
            round(data_list.std(), 2),
            round(numpy.average(data_list), 2),
            data_list.min(),
            data_list.max(),
        )

    # ret['avg_followers'] = round(numpy.average(followers_data), 2)
    # ret['avg_friends'] = round(numpy.average(friends_data), 2)
    ret['avg_polarity'] = round(numpy.average(polarities_data), 2)
    ret['avg_retweet'] = round(numpy.average(retweets_data), 2)
    # ret['std_friends'] = round(friends.std(), 2)
    # ret['std_followers'] = round(followers.std(), 2)
    ret['std_polarity'] = round(polarities.std(), 2)
    ret['std_retweet'] = round(retweets.std(), 2)
    #ret['count'] = round((count_tweets/2000),2)
    #ret['PN_tag'] = PN_tag
    # ret['log_friends'] = round(log(sum(friends_data)) / log(2), 2)
    # ret['log_followers'] = round(log(sum(followers_data)) / log(2), 2)
 #   ret['log_retweets'] = round(log(sum(retweets_data)) / log(2), 2)
 #   ret['log_favorites'] = round(log(sum(favorites_data)) / log(2), 2)

    return ret


def construct_film_characteristic(film_name, tweet_characteristics):
    """
    Construct featuresets for given parameters

    @param film_name string
    @param tweet_characteristics list of dict

    @return featuresets
    """
    ret = {}

    # Analyze film's attributes
    ret['length_of_film'] = len(film_name)
    ret['number_of_words'] = len(film_name.split(' '))

    # Analyze tweet's characteristics
    aggreated_characteristic = tweets2film(tweet_characteristics)

    # Merge 2 characteristics
    ret = dict(ret.items() + aggreated_characteristic.items())

    return ret


features = []

for my_dir in [OSCAR_DIR, RAZZIES_DIR]:
    label = os.path.basename(my_dir)
    for fn in os.listdir(my_dir):
        path = os.path.join(my_dir, fn)
        film_name = os.path.splitext(fn)[0]
    #    print 'dir=%s, film_name=%s, path=%s' % (my_dir, film_name, path)

        with open(path, 'r') as f:
            tweets = json.load(f)
            tweets = json.loads(tweets)

        tweet_characteristics = []
        for tweet in tweets:
            # Per tweet analyze
            characteristic = attribute_to_characteristic(tweet)
            tweet_characteristics.append(characteristic)

        film_characteristic = construct_film_characteristic(
            film_name,
            tweet_characteristics
        )
        # print 'film: |%s|' % film_name
        # print film_characteristic
        feature = (film_characteristic, label)
        features.append(feature)

# Train the classifier
classifier = NaiveBayesClassifier.train(features)

# Predict the film
report = {}
predict_labels = ['oscar', 'razzies']
for predict_label in predict_labels:
    my_dir = os.path.join(PREDICT_DIR, predict_label)
    list_of_files = os.listdir(my_dir)

    report[predict_label] = {
        'number_of_match': 0,
        'number_of_films': len(list_of_files)
    }

    for fn in list_of_files:
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

        film_characteristic = construct_film_characteristic(
            film_name,
            tweet_characteristics
        )
        result = classifier.classify(film_characteristic)

        if result == predict_label:
            report[predict_label]['number_of_match'] += 1

        print film_characteristic
        print 'film: |%s| PREDICT: |%s|\n' % (film_name, result)

report['features'] = film_characteristic.keys()

# classifier.show_most_informative_features()
print "# Features in film's characteristic\n"

for f in report['features']:
    print '* %s' % f

print '\n# Prediction\n'
for predict_label in predict_labels:
    r = report[predict_label]
    print '## %s\n' % predict_label
    print 'match %d out of %d, accuracy=%d%%\n' % (
        r['number_of_match'],
        r['number_of_films'],
        round(r['number_of_match'] / r['number_of_films'] * 100)
    )

print '## overall\n'
print 'match %d out of %d, accuracy=%d%%\n' % (
    sum(
        [report[p]['number_of_match'] for p in predict_labels]
    ),
    sum(
        [report[p]['number_of_films'] for p in predict_labels]
    ),
    round(
        sum(
            [report[p]['number_of_match'] for p in predict_labels]
        ) /
        sum(
            [report[p]['number_of_films'] for p in predict_labels]
        ) * 100
    )
)
