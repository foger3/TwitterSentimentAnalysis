import re, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from src import Cleaners, Tweets

class Visuals:

    def __init__(self):
        model_hold = open('political_classifier.pickle', 'rb')
        self.Classifier = pickle.load(model_hold)
        model_hold.close()
        self.tweet = Tweets.Tweets()
        self.clean = Cleaners.Cleaners()
    
    def sentiment_plots_pie(self):
        fig, ax = plt.subplots(figsize = (6, 6))
        fig.patch.set_facecolor('white')
        patches, texts, pcts = ax.pie(
            self.tweet.pie_data, labels= ['Liberal', 'Conservative'], 
            autopct = '%.1f%%', startangle = 90, 
            wedgeprops={'linewidth': 1, 'edgecolor': 'black'},
            textprops={'size': 'x-large'}, explode = [0.2, 0])
        plt.setp(pcts, color = 'white', fontweight = 'bold')
        ax.set_title('Pie Chart of Sentiment Ratio', fontsize = 18)
        plt.show()
        
    def sentiment_plots_time(self):
        xtick_locator = mdates.AutoDateLocator(interval_multiples = False)
        xtick_formatter = mdates.AutoDateFormatter(xtick_locator)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.tweet.time_series_data.dates, self.tweet.time_series_data.sentiments_prob, color = "grey")
        ax.set_xlabel('Date (in months)', fontsize = 14)
        ax.xaxis.set_major_locator(xtick_locator)
        ax.xaxis.set_major_formatter(xtick_formatter)
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.set_yticklabels(labels = ['', 'Conservative', '', 'Liberal', ''], fontsize = 12, rotation = 90, va = 'center')
        ax.set_title('Tweet Sentiment Over Time', fontsize = 18)
        fig.autofmt_xdate(rotation = 20, ha = 'center')
        plt.show()

    def word_density(self):
        plt.subplots(figsize=(10, 6))
        freq_words = FreqDist(self.clean.get_all_words(self.tweet.density_data))
        filter_words = dict([(m, n) for m, n in freq_words.items() if len(m) > 3])
        cloud = WordCloud().generate_from_frequencies(filter_words)
        plt.imshow(cloud, interpolation = 'bilinear')
        plt.axis("off")
        plt.show()

    def single_tweet(self, num):  
        single_tweet_token = self.clean.remove_noise(word_tokenize(self.tweet.single[num - 1]))
        single_tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                            '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', self.tweet.single[num - 1])
        print(f'The following Tweet: \n\n"{single_tweet.strip()}"')
        print(f'\nHas been classified as: "{self.Classifier.classify(dict([token, True] for token in single_tweet_token))}"')
