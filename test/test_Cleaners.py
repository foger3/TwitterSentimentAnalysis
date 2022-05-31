import unittest

from src.Cleaners import Cleaners
from test.test_Data import TestData

class TestCleaners(unittest.TestCase, TestData):
    
    def test_clean_tweets(self):
        obj = Cleaners()
        test_obj = obj.clean_tweets(TestCleaners.tweet_text, 
                                    TestCleaners.tweet_date)
        self.assertEqual(True, test_obj.equals(
                TestCleaners.tweet_df
                )
            )

    def test_clean_tweets_not_none(self):
        obj = Cleaners()
        self.assertIsNotNone(
            obj.clean_tweets(TestCleaners.tweet_text, 
                            TestCleaners.tweet_date)
            )

    def test_clean_sentiment(self):
        obj = Cleaners()
        test_obj = obj.clean_sentiment(TestCleaners.tweet_text,
                                    TestCleaners.tweet_date)  
        self.assertEqual(True, test_obj.equals( 
                TestCleaners.tweet_sentiments_df
                )
            )
    
    def test_clean_sentiment_not_none(self):
        obj = Cleaners() 
        self.assertIsNotNone(
            obj.clean_sentiment(TestCleaners.tweet_text, 
                                TestCleaners.tweet_date) 
            )

    def test_remove_noise(self):
        obj = Cleaners()
        self.assertEqual(
            obj.remove_noise(TestCleaners.tweet_token[0]),
            ['test', 'function']
            )

    def test_remove_noise_not_none(self):
        obj = Cleaners()
        self.assertIsNotNone(
            obj.remove_noise(TestCleaners.tweet_token[0])
            )
    
    def test_get_all_words(self):
        obj = Cleaners()
        test_obj = obj.get_all_words(TestCleaners.tweet_token)        
        self.assertEqual(
            (next(test_obj), next(test_obj), next(test_obj), 
            next(test_obj), next(test_obj), next(test_obj), 
            next(test_obj)),
            ('Test', 'of', 'this', 'function', 'Other', 'test', 'now')
            )

if __name__ == '__main__':
    unittest.main()
