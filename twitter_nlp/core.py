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
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents

class nlp:
    """A class that computes the counts for certain tweet lists.

    The list can be changed but the attribute has a reasonable default.
    
    Attributes:
        tweets (str): String indicating the tweet list to be used.
        counts (list): List including the counts of nouns, adjectives,
            and verbs from used tweets list.
    """
    def __init__(self, tweets = 'positive_tweets.json'):
        """Constructor for the nlp class.
        
        If the argument `tweets` isn't passed in, the default input
        is used.

        Args:
            tweets (str): String indicating the tweet list to be used.
                Strings to be choosen from are 'positive_tweets.json',
                'negative_tweets.json', 'tweets.20150430-223406.json'
                (default is 'positive_tweets.json').
        """
        self.tweets = tweets 

    def simple_nlp(self):
        """NLP function that counts three parts of speech. 
        
        This function counts the nouns, adjectives, and verbs of the
        example data provided by using the tag function from the nltk
        package. Takes no arguments.
        """
        noun_count = 0
        adj_count = 0
        verb_count = 0
        tweet_token = twitter_samples.tokenized(self.tweets)
        tweet_tag = pos_tag_sents(tweet_token) 

        for tweet in tweet_tag:
            for pair in tweet:
                tag = pair[1]
                if tag == "NN":
                    noun_count += 1
                elif tag == "JJ":
                    adj_count += 1
                elif tag == "VB":
                    verb_count += 1
        self.counts = [noun_count, adj_count, verb_count]
