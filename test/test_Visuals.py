import unittest
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageChops 

from src.Visuals import Visuals
from test.test_Data import TestData

class TestVisuals(unittest.TestCase, TestData):
    
    this_dir, this_filename = os.path.split(__file__) 

    def test_word_density(self):
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

if __name__ == '__main__':
    unittest.main()
