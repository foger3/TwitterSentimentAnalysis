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

    def __init__(self):
        model_hold = open(Visuals.data_path, 'rb')
        self.classifier = pickle.load(model_hold)
        model_hold.close()
        self.clean = Cleaners.Cleaners()

    def word_density(self, tweet_cleaned, random = None):
        fig, ax = plt.subplots(figsize=(10, 6))
        freq_words = FreqDist(self.clean.get_all_words(tweet_cleaned))
        filter_words = dict(
            [(m, n) for m, n in freq_words.items() if len(m) > 3]
            )
        cloud = WordCloud(random_state = random).generate_from_frequencies(filter_words)
        ax.imshow(cloud, interpolation = 'bilinear')
        ax.axis("off")

    def sentiment_plots_pie(self, pie_chart_data):
        fig, ax = plt.subplots(figsize = (8, 6))
        fig.patch.set_facecolor('white')
        patches, texts, pcts = ax.pie(
            pie_chart_data, 
            autopct = lambda p:'{:.1f}%'.format(p) if p > 0 else '', 
            startangle = 90, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'black'},
            textprops={'size': 'x-large'}, 
            explode = [0.2 if (pie_chart_data[0] > 0 and \
                    pie_chart_data[1] > 0) else 0, 0]
            )
        ax.legend(
            labels = ['Liberal', 'Conservative'], 
            loc = 'lower right'
            )
        ax.set_title('Pie Chart of Sentiment Ratio', fontsize = 18)
        plt.setp(pcts, color = 'white', fontweight = 'bold')
        
    def sentiment_plots_time(self, time_series_data):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            time_series_data.dates, 
            time_series_data.sentiments_prob, 
            color = "grey"
            )
        sns.lineplot(
            x = 'dates', 
            y = '14_run_avg', 
            data = time_series_data, 
            color = "darkorange", 
            linewidth = 2, 
            label = "14 Tweet Running Average"
            )
        xtick_locator = mdates.AutoDateLocator(
            interval_multiples = False
            )
        xtick_formatter = mdates.AutoDateFormatter(
            xtick_locator
            )        
        ax.set_xlabel('Dates', fontsize = 14)
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)
        ax.set_ylabel("Sentiment", fontsize = 14)
        ax.set_yticks(np.arange(-0.5, 0.6, 0.25))
        ax.set_yticklabels(
            labels = ['', 'Conservative', '', 'Liberal', ''], 
            fontsize = 12, 
            rotation = 90, 
            va = 'center'
            )
        ax.set_ylim(-0.65, 0.65)
        ax.set_title('Tweet Sentiment Over Time', fontsize = 18)
        fig.autofmt_xdate(rotation = 20, ha = 'center')  

    def single_tweet(self, tweet_text, num):
        single_tweet_token = self.clean.remove_noise(
            word_tokenize(tweet_text[num - 1])
            )
        single_tweet = re.sub(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet_text[num - 1]
            )
        print(f'The following Tweet: \n\n"{single_tweet.strip()}"',
            f'\n\nHas been classified as: "{self.classifier.classify(dict([token, True] for token in single_tweet_token))}"')
