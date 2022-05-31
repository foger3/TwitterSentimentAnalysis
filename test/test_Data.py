import pandas as pd
from datetime import datetime

class TestData:
    
    tweet_text = ['Test of this function', 'Other test now']
    tweet_token = [['Test', 'of', 'this', 'function'],
                ['Other', 'test', 'now']]
    tweet_date = [datetime.fromisoformat('2022-05-30T19:52:43+00:00'), 
                datetime.fromisoformat('2022-05-29T15:52:34+00:00')]
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
