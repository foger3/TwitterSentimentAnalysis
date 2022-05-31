# Sentiment Analysis of Twitter Users

[![Repo Size](https://github-size-badge.herokuapp.com/Programming-The-Next-Step-2022/twitter_nlp.svg)](https://github.com/Programming-The-Next-Step-2022/twitter_nlp)
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
scripts detailing how the political classifier was build 
(.../src/model). Users of this package are invited to do their own 
sentiment analysis using the provided scripts or replicate the procided
classifier.

## Running the package
```bash
git clone https://github.com/Programming-The-Next-Step-2022/twitter_nlp.git
cd PasswordGenerator
python -m src.main
```
