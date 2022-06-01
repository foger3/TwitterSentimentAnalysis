import unittest

from src.Tweets import Tweets
from test.test_Data import TestData

class TestTweets(unittest.TestCase, TestData):
    """Contains the unit tests for the Tweets class.

    The main function of the Tweets class is tested by comparing the 
    output of the function to an expected test output. Inherits test 
    data from TestData to provide input for functions.
    """
    def test_get_tweets(self):
        """Tests if output of get_tweets is equal to test output."""        
        obj = Tweets()
        self.assertEqual(
            obj.get_tweets(
                'luca_tom_third', 
                'bearer_token',
                test = True
                ),
            (TestTweets.tweet_text, TestTweets.tweet_date)
            )

if __name__ == '__main__':
    unittest.main()
