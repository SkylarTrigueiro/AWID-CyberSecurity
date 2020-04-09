# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 08:15:46 2019

@author: skyst
"""

from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from classification_model.config import config


def feature_selection(df, y):
    
    features_to_remove = list()
    df, new_features_to_remove = remove_constant(df)
    features_to_remove.append(new_features_to_remove)
    df, new_features_to_remove = remove_quasi_constant(df)
    features_to_remove.append(new_features_to_remove)
    df, new_features_to_remove = remove_duplicates(df)
    features_to_remove.append(new_features_to_remove)

    
    
    return df, features_to_remove

class remove_constant( BaseEstimator, TransformerMixin ):
    
    def __init__(self, features=None):
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        
    def fit(self, X, y=None):
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
            
        self.drop_features = [ feat for feat in X.columns if X[feat].std() == 0 ]
        
        return self
        
    def transform(self, X, y=None):
        
        X.drop(self.drop_features, axis=1, inplace=True)
        
        return X

class remove_quasi_constant( BaseEstimator, TransformerMixin ):
    
    def __init__(self, threshold = 0.01, features=None):
        
        self.threshold = threshold
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        
    def fit(self, X, y=None):
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
            
        sel = VarianceThreshold(threshold=0.01)  # 0.1 indicates 99% of observations approximately
        sel.fit(X)
        
        features_to_keep = X.columns[sel.get_support()]
        self.drop_features = [ feat for feat in self.features if feat not in features_to_keep ]
        
        return self
        
    def transform(self, X, y=None):
        
        X.drop(self.drop_features, axis=1, inplace=True)
        
        return X

class remove_duplicates( BaseEstimator, TransformerMixin ):
    
    def __init__(self, features=None):
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
        
    def fit(self, X, y=None):
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype == 'O']
            
        duplicated_feat = []
        for i in range(0, len(X.columns)):
            col_1 = list(X.columns)[i]
            if( col_1 not in duplicated_feat ):
                for col_2 in list(X.columns)[i + 1:]:
                    if X[col_1].equals(X[col_2]):
                        duplicated_feat.append(col_2)
                        
        self.drop_features = list(set(duplicated_feat))
        
        return self
        
    def transform(self, X, y=None):
        
        X.drop(self.drop_features, axis=1, inplace=True)
        
        return X
    
class remove_correlated_features( BaseEstimator, TransformerMixin ):
    
    def __init__(self, threshold = 0.8, features = None):
        
        if not isinstance(features, list):
            self.features=[features]
        else:
            self.features = features
            
        self.threshold = threshold
            
    def fit(self, X, y=None):
        
        if self.features == [None]:
            self.features = [col for col in X.columns if X[col].dtype != 'O']

        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = X[self.features].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        
        self.corr_feat = list(col_corr)
        
        return self
        
    def transform(self, X, y=None):
        
        X.drop(self.corr_feat, axis=1, inplace=True)
        
        return X
                     

    
class selected_drop_features( BaseEstimator, TransformerMixin ):
    
    def __init__(self, features = None):
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit(self, X, y=None):
        
        if self.features == [None]:
            self.features = config.TIME_FEATURES + config.ID_FEATURES
            
        self.input_shape_ = X.shape
        
        return self
    
    def transform(self, X, y=None):
        
        X.drop(self.features, axis=1, inplace=True)
        
        return X
    
class drop_features( BaseEstimator, TransformerMixin ):
    
    def __init__(self, features = None):
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
    def fit(self, X, y=None):
        
        if self.features == [None]:
            self.features = config.CONSTANT_FEATURES + config.DUPLICATE_FEATURES + config.QUASI_CONSTANT_FEATURES + config.ID_FEATURES
            
        self.input_shape_ = X.shape
        
        return self
    
    def transform(self, X, y=None):
        
        X.drop(self.features, axis=1, inplace=True)
        
        return X