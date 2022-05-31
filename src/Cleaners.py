import re
import os
import pickle

import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

class Classifier:
    this_dir, this_filename = os.path.split(__file__) 
    data_path = os.path.join(this_dir, 'model/pol_classifier.pickle') 

class Cleaners(Classifier):
    
    def __init__(self):
        model_hold = open(Cleaners.data_path, 'rb')
        self.classifier = pickle.load(model_hold)
        model_hold.close()

    def clean_tweets(self, tweet_text, tweet_date):
        tweet_cleaned = []
        for tweet in range(len(tweet_text)):
            tweet_cleaned.append(
                self.remove_noise(word_tokenize(tweet_text[tweet]))
                )

        tweet_df = pd.DataFrame(
            {'tweets': tweet_cleaned, 
             'dates': tweet_date}
            )
        tweet_df = tweet_df[tweet_df.astype(str)['tweets'] != '[]']
        return tweet_df
    
    def clean_sentiment(self, tweet_text, tweet_date):        
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
        tweet_prob_sentiments = [x for x in tweet_prob_sentiments if x > 0.5]
        tweet_prob_sentiments = [x - 0.5 for x in tweet_prob_sentiments]

        tweet_sentiments_df = pd.DataFrame(
            {'sentiments': tweet_sentiments, 
             'sentiments_prob': tweet_prob_sentiments, 
             'dates': tweet_df.dates}
            )
        tweet_sentiments_df.loc[tweet_sentiments_df['sentiments'].str.contains('Conservative'), 'sentiments_prob'] *= -1
        tweet_sentiments_df = tweet_sentiments_df.sort_values('dates')
        tweet_sentiments_df['14_run_avg'] = tweet_sentiments_df.sentiments_prob.rolling(14).mean()
        return tweet_sentiments_df

    def remove_noise(self, tweet_tokens):
        # CREDIT TO AUTHOR 
        tweet_tokens = [
            a for a, b in zip(tweet_tokens, [''] + tweet_tokens) if b != '@'
            ]
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
        return cleaned_tokens

    def get_all_words(self, cleaned_tokens_list):
        # CREDIT TO AUTHOR 
        for tweet in cleaned_tokens_list:
            for token in tweet:
                yield token
