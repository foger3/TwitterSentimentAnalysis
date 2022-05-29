import unittest

import pandas as pd

from src.Cleaners import Cleaners
from test.test_Data import TestData

class TestCleaners(unittest.TestCase, TestData):
    def test_clean_tweets(self):
        obj = Cleaners()
        test_obj = obj.clean_tweets(TestCleaners.tweet_text, 
                                    TestCleaners.tweet_date)
        self.assertEqual(True, test_obj.equals( 
            pd.DataFrame({'tweets': [['test', 'function'], ['test']],
                        'dates': [TestCleaners.tweet_date[0], 
                                TestCleaners.tweet_date[1]]})
            )
        )

    def test_clean_sentiment(self):
        obj = Cleaners()
        test_obj = obj.clean_tweets(TestCleaners.tweet_text, 
                                    TestCleaners.tweet_date)
        test_obj = obj.clean_sentiment(test_obj)  
        self.assertEqual(True, test_obj.equals( 
            pd.DataFrame({'sentiments': ['Conservative', 
                                        'Conservative'],
                        'sentiments_prob': [-0.34753563238125573,
                                            -0.04166585465758832],
                        'dates': [TestCleaners.tweet_date[0], 
                                TestCleaners.tweet_date[1]],
                        '14_run_avg': [float('nan'), float('nan')]
                        }).sort_values('dates')
            )
        )

    def test_remove_noise(self):
        obj = Cleaners()
        self.assertEqual(
            obj.remove_noise(TestCleaners.tweet_token[0]),
            ['test', 'function']
        )
    
    def test_get_all_words(self):
        obj = Cleaners()
        test_obj = obj.get_all_words(TestCleaners.tweet_token)        
        self.assertEqual(
            (next(test_obj), next(test_obj), 
            next(test_obj), next(test_obj)),
            ('Test', 'of', 'this', 'function')
        )

if __name__ == '__main__':
    unittest.main()
