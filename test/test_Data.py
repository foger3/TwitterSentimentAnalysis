from datetime import datetime

class TestData:
    tweet_text = ['Test of this function', 'Other test now']
    tweet_token = [['Test', 'of', 'this', 'function'],
                ['Other', 'test', 'now']]
    tweet_date = [datetime.strptime('May 29 2022  3:52:34PM',
                                    '%b %d %Y %I:%M:%S%p'), 
                datetime.strptime('May 26 2022  2:22:43PM', 
                                '%b %d %Y %I:%M:%S%p')]
