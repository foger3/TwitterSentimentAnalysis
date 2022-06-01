"""Building classifier for sentiment analysis

This scripts allows users to build their own classifier for sentiment
analysis or other purposes. The classifier is built using the Naive 
Bayes classifier from the nltk package. Steps to build the classifier
follow a top-to-bottom execution of the present script. All steps are
commented out and can be executed individually.


    Note:
        All instances in the building process assigning the words 
        'Liberal' or 'Conservative' and respective references within 
        the package modules have to be replaced in order to customize
        analysis to one's desired needs.
    
    Credit:
        The process and functions were adopted from the code provided by:
        'Shaumik Daityari' at:
        https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk#step-2-tokenizing-the-data
"""

##### Required Imports #####
import os
import csv
import re
import random
import pickle

import tweepy as tw
import pandas as pd
from nltk import classify, NaiveBayesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


##### Set working directory to store/access generated CSV files #####
os.chdir('c:\\Users\\...\\..\\...\\Data Files')


##### Required functions to get data in shape for model build #####
# remove_noise: Removes noise within tweets, inlcuding usernames,
# special characters, links, specific words (e.g., 'tune').
# Removes noise for each token after tagging them with their
# context (e.g., 'Noun' = 'NN') with pos_tag and afterwards
# normalize words to their stem version ('running' = 'run') with
# WordNetLemmatizer().lemmatize. Lastly, stopwords are removed.
# Proper documentation of this function can be found in the
# module "Cleaners" in the "src" folder.
def remove_noise(tweet_tokens):

    tweet_tokens = [
        a for a, b in zip(tweet_tokens, [''] + tweet_tokens) if b != '@'
        ]
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub(r'[0-9]', "", token) # remove numbers
        token = re.sub("[^A-Za-z0-9]+", "", token) # remove special chr
        token = re.sub(r'\b\w{1,2}\b', "", token) # remove words < 3chr     
        token = re.sub(r'\bhtt\w+', "", token) # remove links
        token = re.sub(r'\btco\w+', "", token) # remove links
        token = re.sub(r'\b[Tt]*une\b', "", token) # special for shows

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith("VB"):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token.lower() not in (stopwords.words('english') + ['want', 'would', 'could']):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_tweets_for_model(cleaned_tokens_list):
    """Generator to convert cleaned tokens to dictionary for model build.

    Args:
        cleaned_tokens_list (list): List of tokenized tweets after
            noise removal.

    Yields:
        dictionary: Tokens as keys and True as the value
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


##### Getting Twitter data and according lists to build model #####
# Connecting to and authenticating 'Twitter Developer API':
client = tw.Client(bearer_token = 'your_bearer_token')

# Generation of CSV for each user to be included in training data set:
# CSV files will be saved in set up working directory.
username = 'username'
user = client.get_user(username = username)
with open('%s.csv' % (username), 'w', encoding = 'utf-8') as file:
    w = csv.writer(file)
    w.writerow(['tweet_text', 'timestampe', 'username'])
    for tweet in tw.Paginator(
        client.get_users_tweets, 
        id = user.data.id, exclude = 'retweets', 
        tweet_fields = ['author_id', 'created_at'], 
        max_results = 100
        ).flatten(limit = 1000):
        w.writerow([tweet.text, 
                    tweet.created_at, 
                    tweet.author_id])

# Read in CSV files into separate lists: 
# Setup: Create multiple lists for the two sentiments to be compared
# and simply add the respective lists together with '+'.
# Example: tweets_lib_list = tweets_lib_list1 + tweets_lib_list2 +...
# e.g., liberal list1
df_lib = pd.read_csv("username.csv")
tweets_lib_list = df_lib.tweet_text.to_list() 

# e.g., conservative list1
df_con = pd.read_csv("username.csv")
tweets_con_list = df_con.tweet_text.to_list() 

 
##### Building the model/classifier #####
# Tokenize tweets from both lists and append to new list:
conservative_tweet_token = []
for tweet in range(len(tweets_con_list)):
    conservative_tweet_token.append(word_tokenize(tweets_con_list[tweet]))

liberal_tweet_token = []
for tweet in range(len(tweets_lib_list)):
    liberal_tweet_token.append(word_tokenize(tweets_lib_list[tweet]))

# Removing noise from tokenized tweets and remove empty tweets:
conservative_cleaned = []
for tweet in conservative_tweet_token:
    conservative_cleaned.append(remove_noise(tweet))
conservative_cleaned_tok = [x for x in conservative_cleaned if x != []]

liberal_cleaned = []
for tweet in liberal_tweet_token:
    liberal_cleaned.append(remove_noise(tweet))
liberal_cleaned_tok = [x for x in liberal_cleaned if x != []]

# Convert tweet lists into python dictionary (required):
conservative_tokens_model = get_tweets_for_model(conservative_cleaned_tok)
liberal_tokens_model = get_tweets_for_model(liberal_cleaned_tok)

# Create data sets for both sentiments:
conservative_dataset = [
    (tweet_dict, "Conservative") for tweet_dict in conservative_tokens_model
    ]

liberal_dataset = [
    (tweet_dict, "Liberal") for tweet_dict in liberal_tokens_model
    ]

# Merge data sets and randomly shuffle them:
dataset = conservative_dataset + liberal_dataset
random.shuffle(dataset)

# Split overall data set into train & test (ratio 70:30):
lim = round(len(dataset)*0.7)
train_data = dataset[:lim]
test_data = dataset[lim:]

# Train model, check accuracy, and show most informative features:
classifier = NaiveBayesClassifier.train(train_data)  
print("Accuracy is:", classify.accuracy(classifier, test_data))
classifier.show_most_informative_features(20)


##### Save and load classifier #####
# Saving classifier (will be stored in current woring directory):
model_hold = open('classifier.pickle', 'wb')
pickle.dump(classifier, model_hold)
model_hold.close()

# Loading classifier (will be stored in current woring directory):
model_hold = open('classifier.pickle', 'rb')
classifier = pickle.load(model_hold)
model_hold.close()
# The classifier can be stored in the current directory (...src/model)
# and replace the one provided by default 'pol_classifier.pickle' to
# run your own sentiment analysis with the customized classifier.
