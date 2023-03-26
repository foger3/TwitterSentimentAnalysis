# Sentiment Analysis of Twitter Users

[![Python 3+7 ready](https://img.shields.io/badge/python-3.8%2B-yellowgreen.svg)](https://www.python.org/)
[![Licence](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

## General Idea
After the input of a valid bearer token (Twitter API access) and an 
existing twitter username (@handle), this package provides information
on the twitter user's political sentiment in the form of data 
visualisations (pie chart, time series chart, wordcloud). Additonally,
a single tweet will be classified to illustrate the classification 
process.

## Package Content
The package includes modules executing the described functionality and
a script detailing how the political classifier was build 
(.../twitter_sentiment/model) as well as the used data to train it. 
Users of this package are invited to do their own sentiment analysis 
using the provided script or replicate the provided classifier.

## Running the package
```bash
git clone https://github.com/Programming-The-Next-Step-2022/TwitterSentimentAnalysis.git
cd TwitterSentimentAnalysis
python -m twitter_sentiment.main
```
