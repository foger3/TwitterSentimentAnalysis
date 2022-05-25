import numpy as np
import pandas as pd
import tweepy as tw
import os
import pickle
from nltk.tokenize import word_tokenize

from src import Cleaners

class Tweets:
    
    def __init__(self):
        this_dir, this_filename = os.path.split(__file__) 
        data_path = os.path.join(this_dir, 'data/political_classifier.pickle')
        model_hold = open(data_path, 'rb')
        self.Classifier = pickle.load(model_hold)
        model_hold.close()
        self.clean = Cleaners.Cleaners()

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
        self.text = tweets_text

    def clean_tweets(self, tweets_text, tweets_date):
        tweet_cleaned = []
        for tweet in enumerate(tweets_text):
            tweet_cleaned.append(self.clean.remove_noise(word_tokenize(tweets_text[tweet[0]])))

        tweet_df = pd.DataFrame(
        {'tweets': tweet_cleaned,
        'dates': tweets_date,   
        })
        self.tweet_df = tweet_df[tweet_df.astype(str)['tweets'] != '[]']

    def get_sentiments(self):
        tweet_sentiments = []
        for tweet in self.tweet_df.tweets.tolist():
            tweet_sentiments.append(self.Classifier.classify(dict([token, True] for token in tweet))) 
        self.tweet_sentiments = tweet_sentiments
        
        tweet_prob_sentiments = []
        for tweet in self.tweet_df.tweets.tolist():
            dist = self.Classifier.prob_classify(dict([token, True] for token in tweet))
            for label in dist.samples():
                tweet_prob_sentiments.append(dist.prob(label))  
        self.tweet_prob_sentiments = [x for x in tweet_prob_sentiments if x > 0.5] 

    def pie_data(self):
        pie_chart_data = np.array([self.tweet_sentiments.count('Liberal'), 
                                self.tweet_sentiments.count('Conservative')])
        return(pie_chart_data)

    def time_series_data(self):
        time_series_data = pd.DataFrame(
        {'sentiments': self.tweet_sentiments,
        'sentiments_prob': self.tweet_prob_sentiments,
        'dates': self.tweet_df.dates,
        })
        time_series_data.loc[time_series_data['sentiments'].str.contains('Conservative'), 'sentiments_prob'] *= -1
        return(time_series_data)
    
    def density_data(self):
        return(self.tweet_df.tweets.tolist())

    def single(self):
        return(self.text)
       