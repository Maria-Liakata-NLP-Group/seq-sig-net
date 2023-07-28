import datetime
import numpy as np
import pandas as pd

class TimeFeatures:

    def __init__(self):
        #How many hours in a year?
        self.total_year_hours = 365*24

    def time_fraction(self, x):
        return x.year + abs(x - datetime.datetime(x.year, 1,1,0)).total_seconds() / 3600.0 / self.total_year_hours

    def get_time_features(self, df):

        #obtain time encoding
        df['time_encoding'] = df['datetime'].map(lambda t: self.time_fraction(t))

        df = df.sort_values(by=['timeline_id', 'datetime']).reset_index(drop=True)

        #calculate time difference between posts
        df['time_diff'] = 0
        for i in range(df.shape[0]):
            if (i > 0):
                if (df.loc[i,'timeline_id'] == df.loc[i-1,'timeline_id']):
                    df.loc[i,'time_diff'] = (df.loc[i,'datetime'] - df.loc[i-1,'datetime']).total_seconds() / 60

        #assign index for post in the timeline       
        df['timeline_index'] = 0
        timelineid_list = df['timeline_id'].unique().tolist()
        first_index = 0
        for t_id in timelineid_list:
            t_id_len = len(df[df['timeline_id']==t_id])
            last_index = first_index + t_id_len
            df['timeline_index'][first_index:last_index] = np.arange(t_id_len)
            first_index = last_index
        
        return df