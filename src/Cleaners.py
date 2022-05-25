import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag


class Cleaners:
    
    def __init__(self):
        pass

    def remove_noise(tweet_tokens):
        tweet_tokens = [a for a, b in zip(tweet_tokens, [''] + tweet_tokens) if b != '@']
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            token = re.sub(r'[0-9]', "", token) # remove numbers
            token = re.sub('[^A-Za-z0-9]+', "", token) # remove special chr
            token = re.sub(r'\b\w{1,2}\b', "", token) # remove words < 3chr     
            token = re.sub(r'\bhtt\w+', "", token) # remove links
            token = re.sub(r'\btco\w+', "", token) # remove links

            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token.lower() not in stopwords.words('english'):
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def get_all_words(cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token


