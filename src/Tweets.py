import numpy as np
import pandas as pd
import tweepy as tw
import os, pickle
from nltk.tokenize import word_tokenize

from src import Cleaners, Visuals

class Tweets:
    
    def __init__(self):
        this_dir, this_filename = os.path.split(__file__) 
        data_path = os.path.join(this_dir, 'data/political_classifier.pickle')
        model_hold = open(data_path, 'rb')
        self.Classifier = pickle.load(model_hold)
        model_hold.close()
        self.clean = Cleaners.Cleaners()
        self.visual = Visuals.Visuals()

    def init_input(self):
        bearer_token = input('Bearer Token: \n')
        username = input('Username: \n')
        self.get_tweets(username, bearer_token)
    
    def get_tweets(self, username, bearer_token):
        client = tw.Client(bearer_token = bearer_token)
        user = client.get_user(username = username)
        tweets = client.get_users_tweets(id = user.data.id, max_results = 100, exclude = 'retweets', tweet_fields = ['created_at'])
        tweet_text = [tweet.text for tweet in tweets.data]
        tweet_date = [tweet.created_at for tweet in tweets.data]
        self.tweet_text = tweet_text
        self.clean_tweets(tweet_text, tweet_date)

    def clean_tweets(self, tweet_text, tweet_date):
        tweet_cleaned = []
        for tweet in enumerate(tweet_text):
            tweet_cleaned.append(self.clean.remove_noise(word_tokenize(tweet_text[tweet[0]])))

        tweet_df = pd.DataFrame({'tweets': tweet_cleaned, 'dates': tweet_date})
        self.tweet_df = tweet_df[tweet_df.astype(str)['tweets'] != '[]']

        tweet_sentiments = []
        for tweet in self.tweet_df.tweets.tolist():
            tweet_sentiments.append(self.Classifier.classify(dict([token, True] for token in tweet))) 
        self.tweet_sentiments = tweet_sentiments
        
        tweet_prob_sentiments = []
        for tweet in self.tweet_df.tweets.tolist():
            dist = self.Classifier.prob_classify(dict([token, True] for token in tweet))
            for label in dist.samples():
                tweet_prob_sentiments.append(dist.prob(label))  
        tweet_prob_sentiments = [x for x in tweet_prob_sentiments if x > 0.5]
        self.tweet_prob_sentiments = [x - 0.5 for x in tweet_prob_sentiments] 
        
        self.single_tweet(2)
        self.word_density()
        self.pie_chart()
        self.time_series_chart()
        
    def single_tweet(self, num):  
        self.visual.single_tweet(self.tweet_text, num)

    def word_density(self):
        self.visual.word_density(self.tweet_df.tweets.tolist())

    def pie_chart(self):
        pie_chart_data = np.array([self.tweet_sentiments.count('Liberal'), self.tweet_sentiments.count('Conservative')])
        self.visual.sentiment_plots_pie(pie_chart_data)

    def time_series_chart(self):
        time_series_data = pd.DataFrame({'sentiments': self.tweet_sentiments, 'sentiments_prob': self.tweet_prob_sentiments, 'dates': self.tweet_df.dates})
        time_series_data.loc[time_series_data['sentiments'].str.contains('Conservative'), 'sentiments_prob'] *= -1
        self.visual.sentiment_plots_time(time_series_data)
