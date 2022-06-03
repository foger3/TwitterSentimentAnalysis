"""This module contains the class that executes the visualized analysis.

All methods and information required for the sentiment analysis are
combined in the class of this module. The module provides a visualized
overview of the political sentiments of the specfied user by accessing
the functionalities of the other modules/classes. Output includes a pie
chart, time series chart, wordcloud, and a single classified tweet.
Input for the package is handled here.

    Classes:
        Tweets: Handles input, access data and execute visualisation.

    Typical usage example:
        obj = Tweets()
        obj.get_tweets(username, bearer_tokens)
"""
import numpy as np
import tweepy as tw
import matplotlib.pyplot as plt
from random import randrange

from src import Cleaners, Visuals

class Tweets:
    """Handles input, access data and execute visualisation.

    Class that contains methods to handle user inputs, access
    twitter's API, get the cleaned tweet data of a specified twitter
    user, and call the according visualisation methods.

    Attributes:
        clean (class): Refers to cleaning methods from Cleaners.
        visual (class): Refers to visualization methods from Visuals.

    Methods:
        get_tweets: Gets the tweet data of a specified user.
        get_plots: Calls the visualisation methods and plots them.
    """
    def __init__(self):
        """Constructor for Tweets class."""
        self.clean = Cleaners.Cleaners()
        self.visual = Visuals.Visuals()

    def init_input(self):
        """Initialise the input for the program."""
        bearer_token = input('Bearer Token: \n')
        username = input('Username: \n')
        self.get_tweets(username, bearer_token)
    
    def get_tweets(self, username, bearer_token, test = False):
        """Gets the tweet data of the specified user.

        Method accesses the twitter API and retrieves the tweets of
        a certain user. The data is stored into two list, containing
        information on the tweets content and their dates. Includes
        error handling in case of misspecifications or errors.

        Args:
            username (str): String indicating the username of the
                twitter user.
            bearer_token (str): String indicating the bearer token.
            test (bool, optional): Used for testing get_tweets.
                Defaults to False.

        Returns (optional):
            tweet_text (list): List containing each tweets text (str),
                only returned for unittests.
            tweet_date (list): List containing each tweets date, only
                returned for unittests.
        """
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
        except tw.errors.NotFound: # No username provided.
            print('Please provide a Username!')
        except tw.errors.Unauthorized: # Provided invalid bearer token.
            print('Please provide a valid bearer token!')
        except tw.errors.BadRequest: # Username too long or invalid.
            print('Username too long or you used a special character!')
        except AttributeError: # When user.data.id = 0.
            print('Username does not exist!')
        except TypeError: # When tweets.data empty.
            print('User has no tweets or replies!')
        else:
            if test != True:
                self.get_plots(
                    tweet_text,
                    self.clean.clean_tweets(tweet_text, tweet_date),
                    self.clean.clean_sentiment(tweet_text, tweet_date)
                    )
            else:
                return(tweet_text, tweet_date)

    def get_plots(self, tweet_text, tweet_df, tweet_sentiments_df):
        """Calls the visualisation methods and plots them.

        Method calls visuallisation methods from the Visuals class
        and provides the cleaned data to those. A plotting function is
        called to display the create plots.

        Args:
            tweet_text (list): List containing each tweets text (str).
            tweet_df (dataframe): Dataframe containing columns for
                tokenized cleaned tweets (list(str)) and tweet dates,
                respectively.
            tweet_sentiments_df (dataframe): Dataframe containing
                columns for tweet sentiments (str), sentiment
                probabilities (float), tweet dates, and running
                averages (float), respectively.
        """
        self.visual.single_tweet(tweet_text, randrange(0, len(tweet_text)))
        self.visual.word_density(tweet_df.tweets.tolist())
        self.visual.sentiment_plots_pie(
            np.array(
                [tweet_sentiments_df.sentiments.tolist().count('Liberal'),
                tweet_sentiments_df.sentiments.tolist().count('Conservative')]
                )
            )
        self.visual.sentiment_plots_time(tweet_sentiments_df)
        plt.show()
