import pandas as pd
from datetime import datetime

class TestData:
    
    tweet_text = ['Test of this function', 'Other test now']
    tweet_token = [['Test', 'of', 'this', 'function'],
                ['Other', 'test', 'now']]
    tweet_date = [datetime.strptime('May 29 2022  3:52:34PM',
                                    '%b %d %Y %I:%M:%S%p'), 
                datetime.strptime('May 26 2022  2:22:43PM', 
                                '%b %d %Y %I:%M:%S%p')]
    tweet_df = pd.DataFrame(
        {'tweets': [['test', 'function'],['test']],
        'dates': [tweet_date[0], tweet_date[1]]}
        )
    tweet_sentiments_df = pd.DataFrame(
        {'sentiments': ['Conservative', 'Conservative'],
        'sentiments_prob': [-0.34753563238125573,-0.04166585465758832],
        'dates': [tweet_date[0], tweet_date[1]],
        '14_run_avg': [float('nan'), float('nan')]}
        ).sort_values('dates')
