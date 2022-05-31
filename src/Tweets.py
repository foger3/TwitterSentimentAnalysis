import numpy as np
import tweepy as tw
import matplotlib.pyplot as plt
from random import randrange

from src import Cleaners, Visuals

class Tweets():
    
    def __init__(self):
        self.clean = Cleaners.Cleaners()
        self.visual = Visuals.Visuals()

    def init_input(self):
        bearer_token = input('Bearer Token: \n')
        username = input('Username: \n')
        self.get_tweets(username, bearer_token)
    
    def get_tweets(self, username, bearer_token, test = None):
        try:
            client = tw.Client(bearer_token = bearer_token) 
            user = client.get_user(username = username)
            tweets = client.get_users_tweets(
                id = user.data.id, 
                max_results = 100, 
                exclude = 'retweets', 
                tweet_fields = ['created_at']
                )
            tweet_text = [tweet.text for tweet in tweets.data]
            tweet_date = [tweet.created_at for tweet in tweets.data]
        except tw.errors.NotFound: # no username provided
            print('Please provide a Username!')
        except tw.errors.Unauthorized: # provided invalid bearer token
            print('Please provide a valid bearer token!')
        except tw.errors.BadRequest: # provided too long username
            print('Username too long or you used a special character!')             
        except AttributeError: # when user.data.id = 0
            print('Username does not exist!')
        except TypeError: # when tweets.data empty
            print('User has no tweets or replies!')
        else:
            if test == None:
                self.get_plots(
                    tweet_text, 
                    self.clean.clean_tweets(tweet_text, tweet_date),
                    self.clean.clean_sentiment(tweet_text, tweet_date)
                    )
            else:
                return(tweet_text, tweet_date)
        
    def get_plots(self, tweet_text, tweet_df, tweet_sentiments_df):
        self.visual.single_tweet(
            tweet_text, 
            randrange(0, len(tweet_text))
            )
        self.visual.word_density(tweet_df.tweets.tolist())
        self.visual.sentiment_plots_pie( 
            np.array([tweet_sentiments_df.sentiments.tolist().count('Liberal'), 
                    tweet_sentiments_df.sentiments.tolist().count('Conservative')])
            )
        self.visual.sentiment_plots_time(tweet_sentiments_df)
        plt.show()
