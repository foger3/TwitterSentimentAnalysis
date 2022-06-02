"""This module contain the cleaner and classifier class.

This module serves soley the purpose of providing classes pointing to
the directory of the pre-trained classifier or methods to clean the 
accessed data and arrange it in accessible dataframes. The cleaning
process includes tokenization, normalizing, and removal of 
inconsequential parts of texts (e.g., stopwords, hyperlinks, hashtags).

    Classes:
        Classifier: Provides information on the classifier directory.
        CLeaners: Contains methods to clean tweet data and create 
            dataframes.
"""
import re
import os
import pickle

import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

class Classifier:
    """Provides information on the classifier directory.
    
    Points to the directory containing the pre-trained classifier. 
    The directory contains two versions, one trained on 10.000 and
    one trained on 30.000 tweets. The classifier is saved as a pickle.
    The default classifier is the one trained on 30.000 tweets.

    Note:
        If one wishes to change the classifier, the below path can be
        renamed to reflect the new classifier.
    """
    this_dir, this_filename = os.path.split(__file__) 
    data_path = os.path.join(this_dir, 'model/pol_classifier_30k.pickle') 

class Cleaners(Classifier):
    """Contains methods to clean tweet data and create data frames.

    Class that contains methods to clean twitter data, apply the pre-
    trained classifier and create dataframes with relevant information
    for processing in visualization methods. Inherits from Classifier 
    class.  

    Attributes:
        classifier (NaiveBayesClassifier): Pre-trained classifier for 
            sentiment analysis.
    
    Methods:
        clean_tweets: Tokenizes, cleans tweets, and returns dataframe.
        clean_sentiment: Applies classifier to cleaned tweets and 
            returns dataframe.
        remove_noise: Removes noise from tokenized tweets and 
            normalizes words.
        get_all_words: Generator that provides all words (tokens) in 
            a tweet.
    """    
    def __init__(self):
        """Constructor for Cleaners class."""
        model_hold = open(Cleaners.data_path, 'rb')
        self.classifier = pickle.load(model_hold)
        model_hold.close()

    def clean_tweets(self, tweet_text, tweet_date):
        """Tokenizes, cleans tweets, and returns dataframe.

        Method tokenizes tweets, applies the remove_noise method, and
        creates a dataframe containing the cleaned tweets and their 
        dates.

        Args:
            tweet_text (list): List containing each tweets text (str).
            tweet_date (list): List containing each tweets date.

        Returns:
            tweet_df (dataframe): Dataframe containing columns for 
                tokenized cleaned tweets (list(str)) and tweet dates, 
                respectively.
        """        
        tweet_cleaned = []
        for tweet in range(len(tweet_text)):
            tweet_cleaned.append(
                self.remove_noise(word_tokenize(tweet_text[tweet]))
                )

        tweet_df = pd.DataFrame(
            {'tweets': tweet_cleaned, 
             'dates': tweet_date}
            )
        # Next line: Removes rows with empty lists in 'tweets' column
        # (tweets with no content after noise removal).
        tweet_df = tweet_df[tweet_df.astype(str)['tweets'] != '[]']
        return(tweet_df)
    
    def clean_sentiment(self, tweet_text, tweet_date):   
        """Applies classifier to cleaned tweets and returns dataframe.

        Method applies the classifier to the cleaned tweets and 
        creates a dataframe containing the identified sentiments, 
        sentiment probabilities, and dates of each tweet. A running 
        average of the last 14 tweets is also calculated and added.

        Args:
            tweet_text (list): List containing each tweets text (str).
            tweet_date (list): List containing each tweets date.

        Returns:
            tweet_sentiments_df (dataframe): Dataframe containing 
                columns for tweet sentiments (str), sentiment 
                probabilities (float), tweet dates, and running 
                averages (float), respectively.
        """             
        tweet_df = self.clean_tweets(tweet_text, tweet_date)

        tweet_sentiments = []
        for tweet in tweet_df.tweets.tolist():
            tweet_sentiments.append(
                self.classifier.classify(
                    dict([token, True] for token in tweet)
                    )
                ) 
        
        tweet_prob_sentiments = []
        for tweet in tweet_df.tweets.tolist():
            dist = self.classifier.prob_classify(
                dict([token, True] for token in tweet)
                )
            for label in dist.samples():
                tweet_prob_sentiments.append(dist.prob(label))
        # Next line: Only keeping probabilities greater than 0.5. 
        # Ensures identified sentiments receive the matching value.
        tweet_prob_sentiments = [x for x in tweet_prob_sentiments if x > 0.5]
        # Next line: Transforms all probabilities to a value between 
        # 0 and 0.5 for an easier visuale investigation.
        tweet_prob_sentiments = [x - 0.5 for x in tweet_prob_sentiments]

        tweet_sentiments_df = pd.DataFrame(
            {'sentiments': tweet_sentiments, 
             'sentiments_prob': tweet_prob_sentiments, 
             'dates': tweet_df.dates}
            )
        # Next line: Converts sentiment probabilities of 'Conservative'
        # sentiments to negative values. Facilitates plotting.
        tweet_sentiments_df.loc[tweet_sentiments_df['sentiments'].str.contains('Conservative'), 'sentiments_prob'] *= -1
        tweet_sentiments_df = tweet_sentiments_df.sort_values('dates')
        # Next line: Calculates running average of last 14 tweets.
        tweet_sentiments_df['14_run_avg'] = tweet_sentiments_df.sentiments_prob.rolling(14).mean()
        return(tweet_sentiments_df)

    # Credit: (adapted from) 'Shaumik Daityari'.
    # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk#step-2-tokenizing-the-data
    def remove_noise(self, tweet_tokens):
        """Removes noise from tokenized tweets and normalizes words.

        Method removes usernames, hyperlinks, special characters,
        numbers, short words, and stopwords from tokenized tweets. A
        lemmatization function normalizes words to their root form.

        Args:
            tweet_tokens (list): List containing the tokens (str) of a 
                single tweet.

        Returns:
            cleaned_tokens (list): List containing the tokens (str) of 
                a single tweet with noise removed.
        """
        # Next line: Removes usernames that follow the '@' symbol.
        tweet_tokens = [
            a for a, b in zip(tweet_tokens, [''] + tweet_tokens) if b != '@'
            ]
        # Next few lines: Removes other cases described above.
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            token = re.sub(r'[0-9]', '', token) # remove numbers
            token = re.sub('[^A-Za-z0-9]+', '', token) # remove special
            token = re.sub(r'\b\w{1,2}\b', '', token) # remove shorts     
            token = re.sub(r'\bhtt\w+', '', token) # remove links
            token = re.sub(r'\btco\w+', '', token) # remove links

            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token.lower() not in (stopwords.words('english') + ['want', 'would', 'could']):
                cleaned_tokens.append(token.lower())
        return(cleaned_tokens)

    # Credit: (adapted from) 'Shaumik Daityari'.
    # https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk#step-2-tokenizing-the-data
    def get_all_words(self, tweet_cleaned):
        """Generator that provides all words (tokens) in a tweet.

        Args:
            tweet_cleaned (list): List containing the tokenized 
                cleaned tweets (list(str)).

        Yields:
            token (str): Yields the next token in the list.
        """        
        for tweet in tweet_cleaned:
            for token in tweet:
                yield token
