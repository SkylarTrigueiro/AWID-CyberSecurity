# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 09:58:06 2020

@author: skyst
"""
import pandas as pd
from tf_ann_model.config import config

def feature_creation(X,y):
    
        X = X.copy()
        min_time_epoch = min(X['frame.time_epoch'])
        X = frame_time_epoch(X,min_time_epoch)
        X, tcol = time(X)
        X = activity_count(X, tcol)
        X = activity_change(X, tcol)
    
        if( config.DOWNSAMPLE_DATA ):
            X,y = downsample(X, y)
        
        # Save until target is no longer needed in data.
        X.drop( config.TARGET, axis=1, inplace=True)
       
        return X, y

def frame_time_epoch(X, min_time_epoch):
    
    X['frame.time_epoch'] -= min_time_epoch
         
    return X

def time(X):
    
    X['passed1second'] = X['frame.time_epoch']//1
    X['passed1second'] = X['passed1second'].astype(int)
    
    tcol = ['passed1second']
    
    return X, tcol 

def activity_count(X, tcol):
    
    icol = config.ID_FEATURES
    #icol = ['wlan.ra', 'wlan.da', 'wlan.ta', 'wlan.sa', 'wlan.wep.iv', 'wlan.wep.icv']
    
    for ic in icol:
        for tc in tcol:
            new_id = ic +'_'+ tc  
            X[new_id] = X[ic] + X[tc].astype(str)
            encoder_dict = X[new_id].value_counts().to_dict()
            X[new_id+'_count'] = X[new_id].map(encoder_dict)
            X[new_id+'_count'].fillna(0, inplace=True)
            X.drop(new_id, axis=1, inplace=True)
    
    return X

def activity_change(X, tcol):
    
    icol = config.ID_FEATURES
    
    for ic in icol:
        for tc in tcol:
            map_dict = {}
            new_id = ic + '_' + tc
            time_set = list(set(X[tc]))
            for i in range(1,len(time_set)):
                temp_curr = X[X[tc]==time_set[i]][ic] + X[X[tc]==time_set[i]][tc].astype(str)
                temp_curr = temp_curr.value_counts().to_dict()

                temp_prev = X[X[tc]==time_set[i-1]][ic] + X[X[tc]==time_set[i-1]][tc].astype(str)
                temp_prev = temp_prev.value_counts().to_dict()
                
                for key_curr in temp_curr.keys():
                    ctval = str(time_set[i])
                    key_prev = key_curr[:-len(ctval)] + str(time_set[i-1])
                    if key_prev not in temp_prev.keys():
                        temp_prev[key_prev] = 0
                    map_dict[key_curr] = temp_curr[key_curr] - temp_prev[key_prev] 
                    
            X[new_id] = X[ic] + X[tc].astype(str)      
            X[new_id+ '_count_change'] = X[new_id].map(map_dict)
            X.drop(new_id, axis=1, inplace=True)
            X[new_id+ '_count_change'].fillna(-999, inplace=True)
                
    return X

def downsample(X,y):
    
    import numpy as np
    from sklearn.utils import resample

    X_majority = X[X[config.TARGET]=='normal']
    X_minority = X[X[config.TARGET]!='normal']
        
    n = int(np.round( len(X_minority) / 3 ))
                
    X_majority_downsampled = resample(X_majority,
                                       replace=False,
                                       n_samples=n,
                                       random_state=94019)
    
    X_downsampled = pd.concat([X_majority_downsampled, X_minority])
    X_downsampled.sort_index(inplace=True) 
    
    return X_downsampled, X_downsampled[config.TARGET]

if __name__ == "__main__":
    
    print('will update this part of feat_creation.py later.')
    
    
    