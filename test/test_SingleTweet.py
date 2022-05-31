import unittest
import sys
import io

from src.Visuals import Visuals
from test.test_Data import TestData

class TestSingleTweet(unittest.TestCase, TestData):
    
    def test_single_tweet(self):
        obj = Visuals()
        capturedOutput = io.StringIO()            
        sys.stdout = capturedOutput                    
        obj.single_tweet(TestSingleTweet.tweet_text, num = 1)                                    
        sys.stdout = sys.__stdout__  
        self.assertEqual(
            capturedOutput.getvalue(),
            ('The following Tweet: \n\n"Test of this function" ' 
            '\n\nHas been classified as: "Conservative"\n')
            )

    def test_single_tweet_print(self):
        obj = Visuals()
        capturedOutput = io.StringIO()            
        sys.stdout = capturedOutput                    
        obj.single_tweet(TestSingleTweet.tweet_text, num = 1)                                    
        sys.stdout = sys.__stdout__                     
        print ('Captured', capturedOutput.getvalue()) 

if __name__ == '__main__':
    unittest.main()
