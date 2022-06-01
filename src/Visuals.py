"""This module contains the visualization class.

This module serves soley the purpose of providing a class containing
functions to visualise the data accessed by the specified twitter user.
The visualisations include a pie chart, time series chart, wordcloud, 
and a single classified tweet.

    Classes:
        Visuals: Class that contains the functions to visualise 
            cleaned data.
"""
import re
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from src import Cleaners

class Visuals(Cleaners.Classifier):
    """Contains the functions to visualise cleaned data.

    Class that contains functions to construct the different 
    visualizations. Graphics include a pie chart, time series chart and
    a word cloud. Additionally, an example tweet is printed to the 
    console. Inherits classifier from Classifier class. 

    Attributes:
        classifier (NaiveBayesClassifier): Pre-trained classifier for 
            sentiment analysis.
        clean (class): Class to refer to functions that clean data.
    
    Methods:
        word_density: Creates a wordcloud of the most common words in
            the data.
        sentiment_plots_pie: Creates a pie chart comparing both 
            sentiments in question.
        sentiment_plots_time: Creates a time series chart of the
            sentiment probabilities.
        single_tweet: Prints a random classified tweet to the console.
    """
    def __init__(self):
        """Constructor for Visuals class."""        
        model_hold = open(Visuals.data_path, 'rb')
        self.classifier = pickle.load(model_hold)
        model_hold.close()
        self.clean = Cleaners.Cleaners()

    def word_density(self, tweet_cleaned, random = None):
        # Credit: https://www.pythonprogramming.in/how-to-create-a-word-cloud-from-a-corpus.html
        """Creates a wordcloud of the most common words in the data.

        Args:
            tweet_cleaned (list): List containing the tokenized 
                cleaned tweets (list(str)).
            random (int, optional): Integer used for testing 
                word_density by setting a seed. Defaults to None.
        """        
        fig, ax = plt.subplots(figsize = (10, 6))
        # Next line: Uses FreqDist to get frequency of each word.
        freq_words = FreqDist(self.clean.get_all_words(tweet_cleaned))
        # Next line: Creates dictionary and filters out short words.
        filter_words = dict(
            [(m, n) for m, n in freq_words.items() if len(m) > 3]
            )
        cloud = WordCloud(
            random_state = random
            ).generate_from_frequencies(filter_words)
        ax.imshow(cloud, interpolation = 'bilinear')
        ax.axis('off')

    def sentiment_plots_pie(self, pie_chart_data):
        """Creates a pie chart comparing both sentiments in question.

        Args:
            pie_chart_data (int): Array containing the absolute number 
                of each sentiment identified in the cleaned tweets.
        """        
        fig, ax = plt.subplots(figsize = (8, 6))
        fig.patch.set_facecolor('white')
        patches, texts, pcts = ax.pie(
            pie_chart_data, 
            # Next line: Adjusts for the fact that sentiment might be 0.
            autopct = lambda p:'{:.1f}%'.format(p) if p > 0 else '', 
            startangle = 90, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'black'},
            textprops={'size': 'x-large'}, 
            explode = [0.2 if (pie_chart_data[0] > 0 and \
                       pie_chart_data[1] > 0) else 0, 0]
            )
        ax.legend(labels = ['Liberal', 'Conservative'], loc = 'lower right')
        ax.set_title('Pie Chart of Sentiment Ratio', fontsize = 18)
        plt.setp(pcts, color = 'white', fontweight = 'bold')
        
    def sentiment_plots_time(self, time_series_data):
        """Creates a time series chart of the sentiment probabilities.

        Function uses the sentiment probabilities and dates to create
        a time series chart. Due to difficulties in interpreting the
        fluctuations within sentiment probability, a running average
        including the last 14 values is created.

        Args:
            time_series_data (DataFrame): Dataframe containing columns 
                for tweet sentiments (str), sentiment probabilities 
                (float), tweet dates, and running averages (float), 
                respectively.
        """        
        fig, ax = plt.subplots(figsize = (12, 6))
        ax.plot(
            time_series_data.dates, 
            time_series_data.sentiments_prob, 
            color = 'grey'
            )
        sns.lineplot(
            x = 'dates', 
            y = '14_run_avg', 
            data = time_series_data, 
            color = 'darkorange', 
            linewidth = 2, 
            label = '14 Tweet Running Average'
            )
        # Next two lines: Enable automatic and uniform x ticks/labels.
        xtick_locator = mdates.AutoDateLocator(interval_multiples = False)
        xtick_formatter = mdates.AutoDateFormatter(xtick_locator)        
        ax.set_xlabel('Dates', fontsize = 14)
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)
        ax.set_ylabel('Sentiment', fontsize = 14)
        ax.set_yticks(np.arange(-0.5, 0.6, 0.25))
        ax.set_yticklabels(
            labels = ['', 'Conservative', '', 'Liberal', ''], 
            fontsize = 12, 
            rotation = 90, 
            va = 'center'
            )
        ax.set_ylim(-0.65, 0.65)
        ax.set_title('Tweet Sentiment Over Time', fontsize = 18)
        # Next line: Rotates x axis labels and centers them on ticks.
        fig.autofmt_xdate(rotation = 20, ha = 'center')  

    def single_tweet(self, tweet_text, num):
        """Prints a random classified tweet to the console.

        Args:
            tweet_text (list): List containing each tweets text (str).
            num (int): Random integer to select a tweet from the tweet
                text list.
        """   
        single_tweet_token = self.clean.remove_noise(
            word_tokenize(tweet_text[num - 1])
            )
        # Next line: removes hyperlinks from text
        single_tweet = re.sub(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet_text[num - 1]
            )
        print(
            f'The following Tweet: \n\n"{single_tweet.strip()}"',
            f'\n\nHas been classified as: "{self.classifier.classify(dict([token, True] for token in single_tweet_token))}"'
            )
