"""Sentiment Analysis of Twitter Users.

This package uses tools from natural language processing to perform a
sentiment analysis of twitters users. By inputting the username
(@handle) of a certain twitter user, the modules use a pre-trained
classifier to evaluate the most recent 100 tweets (or available ones)
and provide a visualisation of the overall political sentiment,
including more detailed output on sentiment development over time and
word frequency.

    Modules:
        Tweets: Contains class to access data and execute visualisation.
        Visuals: Contains class with visualisation methods.
        Cleaners: Contains class with data cleaning methods.
"""
from twitter_sentiment import Tweets
