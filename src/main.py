"""This module enables a political sentiment analysis of user tweets.

This module uses natural language processing to determine the sentiment
of twitters users. A pre-trained model evaluates recent tweets and 
provides a summary of measures of interest (e.g., political sentiment).
The model has been trained with tweets categorised as conservative vs. 
liberal. Tweets are accessed via the twitter API.

Example:
    $ python core.py

Functions:
    simple_nlp(): Counts nouns, adjectives, and verbs of tweets.

Todo:
    * incorporate tweet generator by using twitter API package
    * incorpoarte full sentiment analysis
"""
import src.Tweets

if __name__ == '__main__':
    tweet_sentiment = src.Tweets.Tweets()
    tweet_sentiment.init_input()
