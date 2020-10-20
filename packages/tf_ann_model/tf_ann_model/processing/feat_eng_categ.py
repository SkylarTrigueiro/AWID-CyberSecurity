# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:10:53 2020

@author: skyst
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class rare_label_encoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, tol = 0.05, features=None):
       
        if tol < 0 or tol > 1 :
            raise ValueError("tol takes values between 0 and 1")
        
        self.tol = tol
                    
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit(self, X, y=None ) -> 'rare_label_encoder':
        
        self.encoder_dict_ = {}
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
        
        for feat in self.features:
            temp = pd.Series(X[feat].value_counts(normalize=True))
            self.encoder_dict_[feat] = temp[temp >= self.tol].index
           
        self.input_shape_ = X.shape

        return self
    
    def transform(self, X ):
      
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
            
        X = X.copy()
        for feat in self.features:
            X[feat] = np.where(X[feat].isin(self.encoder_dict_[feat]), X[feat], 'Rare')

        return X
    
class categ_missing_encoder(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, features=None):
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit(self, X, y=None):
      
        if self.features == [None]:
            self.features = [ col for col in X.columns if (X[col].isnull().sum() > 0) and X[col].dtype == 'O' ]
        self.input_shape_ = X.shape  
        
        return self

    def transform(self, X, y=None):
     
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the imputer')

        X = X.copy()
        X.loc[:, self.features] = X[self.features].fillna('Missing')
        
        return X

    
class label_encoder(BaseEstimator, TransformerMixin):
    
    
    def __init__(self, encoding_method  = 'arbitrary', features = None):
        
        if encoding_method not in ['arbitrary', 'count', 'both']:
            raise ValueError("encoding_method takes only values 'ordered' and 'arbitrary'")
            
        self.encoding_method = encoding_method
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        
    def fit(self, X, y=None):
       
        
        self.encoder_dict_ = {}
        self.encoder_dict1_ = {}
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
        
        for feat in self.features:
            
            if self.encoding_method == 'arbitrary':
                temp = X[feat].unique()
                self.encoder_dict_[feat] = {k:i for i, k in enumerate(temp, 0)}
                
            elif self.encoding_method == 'count':
                self.encoder_dict_[feat] = X[feat].value_counts().to_dict()
                
            elif self.encoder_dict_ == 'both':
                self.encoder_dict_[feat] = {k:i for i, k in enumerate(temp, 0)}
                self.encoder_dict1_[feat] = X[feat].value_counts().to_dict()
            
        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        #self.input_shape_ = X.shape
        
        return self
    
    def transform(self, X, y = None):
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        #if X.shape[1] != self.input_shape_[1]:
        #    raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
            
        X = X.copy()
        for feat in self.features:
            X[feat] = X[feat].map(self.encoder_dict_[feat])
            if( self.encoding_method == 'both' ):
                X[feat + '_count'] = X[feat].map(self.encoder_dict1_[feat])
        
        return X
    
class one_hot_encoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_labels = 9999, drop_last = False, features=None):
        
        if max_labels:
            if not isinstance(max_labels, int):
                raise ValueError("top_categories takes only integer numbers, 1, 2, 3, etc.")            
        
        if drop_last not in [True, False]:
            raise ValueError("drop_last takes only True or False")
            
        self.drop_last = drop_last
        self.max_labels = max_labels
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit( self, X, y=None ):
        
        self.encoder_dict_ = {}
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
        
        for feat in self.features:
            if not self.max_labels:
                if self.drop_last:
                    category_ls = [x for x in X[feat].unique() ]
                    self.encoder_dict_[feat] = category_ls[:-1]
                else:
                    self.encoder_dict_[feat] = X[feat].unique()
                
            else:
                self.encoder_dict_[feat] = [x for x in X[feat].value_counts().sort_values(ascending=False).head(self.max_labels).index]

        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
        
        return self
    
    def transform( self, X, y=None ):
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')
        
        X = X.copy()
        for feat in self.features:
            for label in self.encoder_dict_[feat]:
                X[str(feat) + '_' + str(label)] = np.where(X[feat] == label, 1, 0)
            
        # drop the original non-encoded variables.
        X.drop(labels=self.features, axis=1, inplace=True)
        
        return X
    
class discrete_to_categ(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_labels=10, features = None):
        
        self.max_labels = max_labels
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit( self, X, y=None ):
        
        self.encode_dict_ = {}
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
        
        for col in X.columns:
            if X[col].nunique() < self.max_labels:
                self.encode_dict_[col] = 'O'
                self.features.append(col)
        
        self.input_shape_ = X.shape
        
        return self
        
    def transform( self, X, y=None ):
        
        X = X.copy()
        X = X.astype(self.encode_dict_)
        
        return X
        
        
        
        