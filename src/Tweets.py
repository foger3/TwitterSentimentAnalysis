import numpy as np
import tweepy as tw
from random import randrange

from src import Cleaners, Visuals

class Tweets():
    
    def __init__(self):
        self.clean = Cleaners.Cleaners()
        self.visual = Visuals.Visuals()
        self.text = None

    def init_input(self):
        bearer_token = input('Bearer Token: \n')
        username = input('Username: \n')
        self.get_tweets(username, bearer_token)
    
    def get_tweets(self, username, bearer_token):
        client = tw.Client(bearer_token = bearer_token)
        user = client.get_user(username = username)
        tweets = client.get_users_tweets(id = user.data.id, max_results = 100, exclude = 'retweets', tweet_fields = ['created_at'])
        tweet_text = [tweet.text for tweet in tweets.data]
        self.tweet_text = [tweet.text for tweet in tweets.data]
        tweet_date = [tweet.created_at for tweet in tweets.data]
        self.get_data(tweet_text, tweet_date)

    def get_data(self, tweet_text, tweet_date):
        tweet_df = self.clean.clean_tweets(tweet_text, tweet_date)
        tweet_sentiments_df = self.clean.clean_tweets_sentiment(tweet_df)
        self.get_plots(tweet_text, tweet_df, tweet_sentiments_df)
        
    def get_plots(self, tweet_text, tweet_df, tweet_sentiments_df):
        self.visual.single_tweet(tweet_text, randrange(0, len(tweet_text)))
        self.visual.word_density(tweet_df.tweets.tolist())
        self.visual.sentiment_plots_pie(
            np.array([tweet_sentiments_df.sentiments.value_counts().Liberal, 
                    tweet_sentiments_df.sentiments.value_counts().Conservative])
            )
        self.visual.sentiment_plots_time(tweet_sentiments_df)
