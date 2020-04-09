# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:52:52 2020

@author: skyst
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from classification_model.config import config


class outlier_capping(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, distribution='gaussian', tail='both', fold=3, features = None):
        
        if distribution not in ['gaussian', 'skewed', 'quantiles']:
            raise ValueError("distribution takes only values 'gaussian', 'skewed' or 'quantiles'")
            
        if tail not in ['right', 'left', 'both']:
            raise ValueError("tail takes only values 'right', 'left' or 'both'")
            
        if fold <=0 :
            raise ValueError("fold takes only positive numbers")
            
        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

    
    def fit(self, X, y=None):
        
        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype != 'O' and col not in config.TIME_FEATURES]
        
        # estimate the end values
        if self.tail in ['right', 'both']:
            if self.distribution == 'gaussian':
                self.right_tail_caps_ = (X[self.features].mean()+self.fold*X[self.features].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.features].quantile(0.75) - X[self.features].quantile(0.25)
                self.right_tail_caps_ = (X[self.features].quantile(0.75) + (IQR * self.fold)).to_dict()
                
            elif self.distribution == 'quantiles':
                self.right_tail_caps_ = X[self.features].quantile(0.95).to_dict()
        
        if self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.features].mean()-self.fold*X[self.features].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.features].quantile(0.75) - X[self.features].quantile(0.25)
                self.left_tail_caps_ = (X[self.features].quantile(0.25) - (IQR * self.fold)).to_dict()
                
            elif self.distribution == 'quantiles':
                self.left_tail_caps_ = X[self.features].quantile(0.05).to_dict()
        
        self.input_shape_ = X.shape
              
        return self
    
    def transform( self, X, y=None):
        
        X = X.copy()
        if self.tail == 'both':
            for feat in self.features:
                X[feat]= np.where(X[feat] > self.right_tail_caps_[feat], self.right_tail_caps_[feat],
                           np.where(X[feat] < self.left_tail_caps_[feat], self.left_tail_caps_[feat], X[feat]))
        elif self.tail == 'right':
            for feat in self.features:
                X[feat] = np.where(X[feat] > self.right_tail_caps_[feat], self.right_tail_caps_[feat], X[feat])
        elif self.tail == 'left':
            for feat in self.features:
                X[feat] = np.where(X[feat] < self.left_tail_caps_[feat], self.left_tail_caps_[feat], X[feat])
        
        return X



class ArbitraryNumberImputer(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, arbitrary_number = -999, features = None):
        
        if isinstance(arbitrary_number, int) or isinstance(arbitrary_number, float):
            self.arbitrary_number = arbitrary_number
        else:
            raise ValueError('Arbitrary number must be numeric of type int or float')
            
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        
       
    def fit(self, X, y=None):
       
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype != 'O']
                  
        self.input_shape_ = X.shape
        
        return self


    def transform(self, X):
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')
        
        X = X.copy()
        for feat in self.features:
            X[feat] = X[feat].fillna(self.arbitrary_number)
        
        return X