import numpy as np
import pandas as pd
import tweepy as tw
import pickle
from nltk.tokenize import word_tokenize

from src import Cleaners

class Tweets:
    
    def __init__(self):
        model_hold = open('political_classifier.pickle', 'rb')
        self.Classifier = pickle.load(model_hold)
        model_hold.close()

    def init_input(self):
        bearer_token = input('Bearer Token: \n')
        username = input('Username: \n')
        self.get_tweets(username, bearer_token)
    
    def get_tweets(self, username, bearer_token):
        client = tw.Client(bearer_token = bearer_token)
        user = client.get_user(username = username)
        tweets = client.get_users_tweets(id = user.data.id, max_results = 100, exclude = 'retweets', tweet_fields = ['created_at'])
        tweets_text = [tweet.text for tweet in tweets.data]
        tweets_date = [tweet.created_at for tweet in tweets.data]
        self.clean_tweets(tweets_text, tweets_date)

    def clean_tweets(self, tweets_text, tweets_date):
        tweet_cleaned = []
        for tweet in enumerate(tweets_text):
            tweet_cleaned.append(remove_noise(word_tokenize(tweets_text[tweet[0]])))

        tweet_df = pd.DataFrame(
        {'tweets': tweet_cleaned,
        'dates': tweets_date,   
        })
        tweet_df = tweet_df[tweet_df.astype(str)['tweets'] != '[]']
        tweet_cleaned_token = tweet_df.tweets.tolist()

        tweet_sentiments = []
        for tweet in tweet_cleaned_token:
            tweet_sentiments.append(self.Classifier.classify(dict([token, True] for token in tweet)))

        tweet_prob_sentiments = []
        for tweet in tweet_cleaned_token:
            dist = self.Classifier.prob_classify(dict([token, True] for token in tweet))
            for label in dist.samples():
                tweet_prob_sentiments.append(dist.prob(label))  
        tweet_prob_sentiments = [x for x in tweet_prob_sentiments if x > 0.5]  

        pie_chart_data = np.array([tweet_sentiments.count('Liberal'), tweet_sentiments.count('Conservative')])

        time_series_data = pd.DataFrame(
        {'sentiments': tweet_sentiments,
        'sentiments_prob': tweet_prob_sentiments,
        'dates': tweet_df.dates,
        })
        time_series_data.loc[time_series_data['sentiments'].str.contains('Conservative'), 'sentiments_prob'] *= -1

       