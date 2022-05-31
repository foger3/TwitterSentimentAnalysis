import unittest

from src.Tweets import Tweets
from test.test_Data import TestData

class TestTweets(unittest.TestCase, TestData):
    
    def test_get_tweets(self):
        obj = Tweets()
        self.assertEqual(
            obj.get_tweets(
                'luca_tom_third', 
                'Bearer Token', 
                test = "test"),
            (TestTweets.tweet_text, TestTweets.tweet_date)
            )

if __name__ == '__main__':
    unittest.main()
