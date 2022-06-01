import unittest
import sys
import os
import io

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageChops 

from src.Visuals import Visuals
from test.test_Data import TestData

class TestVisuals(unittest.TestCase, TestData):
    """Contains the unit tests for the Visuals class.

    Functions of the Visuals class are tested by comparing the output
    of the function to an expected test output. Expected images for
    the test output are stored in the test/test_visuals directory. 
    Inherits test data from TestData to provide input for functions.
    """    
    this_dir, this_filename = os.path.split(__file__) 

    def test_word_density(self):
        """Tests if output of function is not different to expected image.
        
        The output of word_density is saved to a file and compared to 
        the expected image by examining the pixel-by-pixel differences.
        """        
        obj = Visuals()
        obj.word_density(TestVisuals.tweet_df.tweets.tolist(), 1)
        plt.savefig(
            TestVisuals.this_dir 
            + '/test_visuals/test_word_density.png'
            )
        diff = ImageChops.difference(
            Image.open(
                TestVisuals.this_dir 
                + '/test_visuals/test_expected_word_density.png'
                ).convert('RGB'), 
            Image.open(
                TestVisuals.this_dir 
                + '/test_visuals/test_word_density.png'
                ).convert('RGB')
            )
        self.assertIsNone(diff.getbbox())
        os.remove(TestVisuals.this_dir +'/test_visuals/test_word_density.png')

    def test_sentiment_plots_pie(self):
        """Tests if output of function is not different to expected image.
        
        The output of sentiment_plots_pie is saved to a file and 
        compared to the expected image by examining the pixel-by-pixel 
        differences.
        """        
        obj = Visuals()
        obj.sentiment_plots_pie(
            np.array(
                [TestVisuals.tweet_sentiments_df.sentiments.tolist().count('Liberal'), 
                 TestVisuals.tweet_sentiments_df.sentiments.tolist().count('Conservative')]
                )
            )
        plt.savefig(
            TestVisuals.this_dir 
            + '/test_visuals/test_pie_chart.png'
            )
        diff = ImageChops.difference(
            Image.open(
                TestVisuals.this_dir 
                + '/test_visuals/test_expected_pie_chart.png'
                ).convert('RGB'), 
            Image.open(
                TestVisuals.this_dir 
                + '/test_visuals/test_pie_chart.png'
                ).convert('RGB')
            )
        self.assertIsNone(diff.getbbox())
        os.remove(TestVisuals.this_dir + '/test_visuals/test_pie_chart.png')        

    def test_sentiment_plots_time(self):
        """Tests if output of function is not different to expected image.
        
        The output of sentiment_plots_time is saved to a file and
        compared to the expected image by examining the pixel-by-pixel
        differences.
        """        
        obj = Visuals()
        obj.sentiment_plots_time(TestVisuals.tweet_sentiments_df)
        plt.savefig(
            TestVisuals.this_dir 
            + '/test_visuals/test_time_series.png'
            )
        diff = ImageChops.difference(
            Image.open(
                TestVisuals.this_dir 
                + '/test_visuals/test_expected_time_series.png'
                ).convert('RGB'), 
            Image.open(
                TestVisuals.this_dir 
                + '/test_visuals/test_time_series.png'
                ).convert('RGB')
            )
        self.assertIsNone(diff.getbbox())
        os.remove(TestVisuals.this_dir + '/test_visuals/test_time_series.png')

    def test_single_tweet(self):
        """Tests if print by single_tweet is equal to test output."""        
        obj = Visuals()
        capturedOutput = io.StringIO()            
        sys.stdout = capturedOutput                    
        obj.single_tweet(TestVisuals.tweet_text, num = 1)                                    
        sys.stdout = sys.__stdout__  
        self.assertEqual(
            capturedOutput.getvalue(),
            ('The following Tweet: \n\n"Test of this function" ' 
            '\n\nHas been classified as: "Conservative"\n')
            )

    def test_single_tweet_print(self):
        """Captures printed output by single_tweet."""        
        obj = Visuals()
        capturedOutput = io.StringIO()            
        sys.stdout = capturedOutput                    
        obj.single_tweet(TestVisuals.tweet_text, num = 1)                                    
        sys.stdout = sys.__stdout__                     
        print('Captured', capturedOutput.getvalue()) 

if __name__ == '__main__':
    unittest.main()
