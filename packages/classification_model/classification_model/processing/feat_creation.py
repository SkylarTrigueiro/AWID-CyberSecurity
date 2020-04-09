# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:58:06 2020

@author: skyst
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from classification_model.config import config

class feature_creation(BaseEstimator, TransformerMixin):
    
    def __init__(self, features=None):
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit(self, X, y=None ) -> 'feature_creation':

        self.min_time_epoch = min(X['frame.time_epoch'])
        
        return self
    
    def transform(self, X, y=None) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        X = frame_time_epoch(X,self.min_time_epoch)
        X, self.tcol = time(X)
        X = activity_count(X, self.tcol)
        X = activity_change(X, self.tcol)
        #X = wlan_ra(X)
        #X = wlan_da(X)
        #X = wlan_ta(X)
        #X = wlan_sa(X)
        #X = wlan_ba_bm(X)
        #X = wlan_bssid(X)
        #X = wlan_mgt_fixed_current_ap(X)
       
        return X

def frame_time_epoch(df, min_time_epoch):
    
    df['frame.time_epoch'] -= min_time_epoch
         
    return df

def time(df):
    
    df['passed1second'] = df['frame.time_epoch']//1
    df['passed1second'] = df['passed1second'].astype(int)
    
    tcol = ['passed1second']
    
    return df, tcol 

def activity_count(df, tcol):
    
    icol = config.ID_FEATURES
    #icol = ['wlan.ra', 'wlan.da', 'wlan.ta', 'wlan.sa', 'wlan.wep.iv', 'wlan.wep.icv']
    
    for ic in icol:
        for tc in tcol:
            new_id = ic +'_'+ tc  
            df[new_id] = df[ic] + df[tc].astype(str)
            encoder_dict = df[new_id].value_counts().to_dict()
            df[new_id+'_count'] = df[new_id].map(encoder_dict)
            df[new_id+'_count'].fillna(0, inplace=True)
            df.drop(new_id, axis=1, inplace=True)
    
    return df

def activity_change(df, tcol):
    
    icol = config.ID_FEATURES
    
    for ic in icol:
        for tc in tcol:
            map_dict = {}
            new_id = ic + '_' + tc
            time_set = list(set(df[tc]))
            for i in range(1,len(time_set)):
                temp_curr = df[df[tc]==time_set[i]][ic] + df[df[tc]==time_set[i]][tc].astype(str)
                temp_curr = temp_curr.value_counts().to_dict()

                temp_prev = df[df[tc]==time_set[i-1]][ic] + df[df[tc]==time_set[i-1]][tc].astype(str)
                temp_prev = temp_prev.value_counts().to_dict()
                
                for key_curr in temp_curr.keys():
                    ctval = str(time_set[i])
                    key_prev = key_curr[:-len(ctval)] + str(time_set[i-1])
                    if key_prev not in temp_prev.keys():
                        temp_prev[key_prev] = 0
                    map_dict[key_curr] = temp_curr[key_curr] - temp_prev[key_prev] 
                    
            df[new_id] = df[ic] + df[tc].astype(str)      
            df[new_id+ '_count_change'] = df[new_id].map(map_dict)
            df.drop(new_id, axis=1, inplace=True)
            df[new_id+ '_count_change'].fillna(-999, inplace=True)
                
    return df

def wlan_ra(df):
    
    if 'wlan.ra' in df.columns:
        df['wlan.ra'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan.ra'].str.split(':', n = 5, expand=True)
        for i in range(6):
            df['wlan.ra_' + str(i)] = temp[i]
        df.drop('wlan.ra', axis=1, inplace = True)
    
    return df

def wlan_da(df):
    
    if 'wlan.da' in df.columns:
        df['wlan.da'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan.da'].str.split(':', n = 5, expand=True)
        for i in range(6):
            df['wlan.da_' + str(i)] = temp[i]
        df.drop('wlan.da', axis=1, inplace = True)
    
    return df

def wlan_ta(df):
    
    if 'wlan.ta' in df.columns:
        df['wlan.ta'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan.ta'].str.split(':', n = 5, expand=True)
        for i in range(6):
            df['wlan.ta_' + str(i)] = temp[i]
        df.drop('wlan.ta', axis=1, inplace = True)
    
    return df

def wlan_sa(df):
    
    if 'wlan.sa' in df.columns:
        df['wlan.sa'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan.sa'].str.split(':', n = 5, expand=True)
        for i in range(6):
            df['wlan.sa_' + str(i)] = temp[i]
        df.drop('wlan.sa', axis=1, inplace = True)
    
    return df

def wlan_bssid(df):
    
    if 'wlan.bssid' in df.columns:
        df['wlan.bssid'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan.bssid'].str.split(':', n = 5, expand=True)
        for i in range(6):
            df['wlan.bssid_' + str(i)] = temp[i]
        df.drop('wlan.bssid', axis=1, inplace = True)
    
    return df

def wlan_ba_bm(df):
    
    if 'wlan.ba.bm' in df.columns:
        df['wlan.ba.bm'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan.ba.bm'].str.split(':', n = 7, expand=True)
        for i in range(8):
            df['wlan.ba.bm_' + str(i)] = temp[i]
        df.drop('wlan.ba.bm', axis=1, inplace = True)
    
    return df

def wlan_mgt_fixed_current_ap(df):
    
    if 'wlan_mgt_fixed_current_ap' in df.columns:
        df['wlan.ba.bm'].fillna( 'Missing:Missing:Missing:Missing:Missing:Missing:Missing:Missing', inplace = True )
        temp = df['wlan_mgt_fixed_current_ap'].str.split(':', n = 7, expand=True)
        for i in range(8):
            df['wlan_mgt_fixed_current_ap_' + str(i)] = temp[i]
        df.drop(['wlan_mgt_fixed_current_ap'], axis=1, inplace = True)
    
    return df

def radiotap_dbm_antsignal(df):
    
    if 'radiotap.dbm_antsignal' in df.columns:
        df['radiotap.dbm_antsignal'].fillna( 1, inplace = True)
    
    return df

if __name__ == "__main__":
    
    from data_management import load_dataset, downsample
    from feat_eng_categ import categ_missing_encoder
    
    train_orig = load_dataset(file_name='AWID-CLS-R-Trn.csv')
    train_ds = downsample(train_orig)
    
    cme = categ_missing_encoder(features=config.ID_FEATURES)
    X = cme.fit_transform(train_ds)
 
    fc = feature_creation()
    X = fc.fit_transform(X)
    
    new_vars = [var for var in X.columns if var not in train_ds]
    X_new_vars = X[new_vars]
    X_new_vars = X_new_vars.head(1000)
    
    
    