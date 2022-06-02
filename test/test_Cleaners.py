import unittest

from src.Cleaners import Cleaners
from test.test_Data import TestData

class TestCleaners(unittest.TestCase, TestData):
    """Contains the unit tests for the Cleaners class.

    Methods of the Cleaners class are tested by comparing the output
    of the methods to an expected test output. Certain methods are
    tested on wether any output is produced. Inherits test data from 
    TestData to provide input for methods and compare outputs.
    """
    def test_clean_tweets(self):
        """Tests if output of clean_tweets is equal to test output."""       
        obj = Cleaners()
        test_obj = obj.clean_tweets(
            TestCleaners.tweet_text, 
            TestCleaners.tweet_date
            )
        self.assertEqual(
            True, 
            test_obj.equals(TestCleaners.tweet_df)
            )

    def test_clean_tweets_not_none(self):
        """Tests if clean_tweets returns output different from none."""        
        obj = Cleaners()
        self.assertIsNotNone(
            obj.clean_tweets(
                TestCleaners.tweet_text, 
                TestCleaners.tweet_date
                )
            )

    def test_clean_sentiment(self):
        """Tests if output of clean_sentiment is equal to test output."""        
        obj = Cleaners()
        test_obj = obj.clean_sentiment(
            TestCleaners.tweet_text,
            TestCleaners.tweet_date
            )  
        self.assertEqual(
            True, 
            test_obj.equals(TestCleaners.tweet_sentiments_df)
            )
    
    def test_clean_sentiment_not_none(self):
        """Tests if clean_sentiment returns output different from none."""        
        obj = Cleaners() 
        self.assertIsNotNone(
            obj.clean_sentiment(
                TestCleaners.tweet_text, 
                TestCleaners.tweet_date
                ) 
            )

    def test_remove_noise(self):
        """Tests if output of remove_noise is equal to test output."""        
        obj = Cleaners()
        self.assertEqual(
            obj.remove_noise(TestCleaners.tweet_token[0]),
            ['test', 'function']
            )

    def test_remove_noise_not_none(self):
        """Tests if remove_noise returns output different from none."""        
        obj = Cleaners()
        self.assertIsNotNone(
            obj.remove_noise(TestCleaners.tweet_token[0])
            )
    
    def test_get_all_words(self):
        """Tests if get_all_words generates test output."""       
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
