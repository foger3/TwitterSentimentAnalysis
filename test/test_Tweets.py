import unittest

from src.Tweets import Tweets
from test.test_Data import TestData

class TestTweets(unittest.TestCase, TestData):
    """Contains the unit tests for the Tweets class.

    The main method of the Tweets class is tested by comparing the
    output of the method to an expected test output. Inherits test
    data from TestData to provide input for method and compare output.
    """
    def test_get_tweets(self):
        """Tests if output of get_tweets is equal to test output."""
        obj = Tweets()
        self.assertEqual(
            obj.get_tweets(
                'luca_tom_third',
                'AAAAAAAAAAAAAAAAAAAAADmacgEAAAAApcsL8QoWghaKhVP1mFWJisEHbF4%'
                '3D3jruDJIoCcyZgpOKieoweAsTtp9ZAmEc9MXBgyusHUHfzjkfQ6',
                test = True
                ),
            (TestTweets.tweet_text, TestTweets.tweet_date)
            )

if __name__ == '__main__':
    unittest.main()
