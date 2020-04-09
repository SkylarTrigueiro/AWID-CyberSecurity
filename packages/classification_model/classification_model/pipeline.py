from sklearn.pipeline import Pipeline

from classification_model.processing.feat_eng_categ import discrete_to_categ, one_hot_encoder, categ_missing_encoder, rare_label_encoder, label_encoder
from classification_model.processing.feat_eng_num import outlier_capping, ArbitraryNumberImputer
from classification_model.processing.feat_creation import feature_creation
from classification_model.processing.feat_selection import remove_constant, remove_quasi_constant, remove_duplicates, selected_drop_features, remove_correlated_features
from classification_model.config import config
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
#import lightgbm as lgb
#from sklearn.multiclass import OneVsRestClassifier

xgboost_params = {#'bagging_fraction': 0.8993155305338455, 
          #'colsample_bytree': 0.7463058454739352, 
          #'feature_fraction': 0.7989765808988153, 
          #'gamma': 0.6665437467229817, 
          'metric': 'mlogloss',
          #'learning_rate': 0.013887824598276186, 
          #'max_depth': 16, 
          #'min_child_samples': 170,
          #'num_leaves': 220, 
          #'reg_alpha': 0.39871702770778467, 
          #'reg_lambda': 0.24309304355829786,
          #'subsample': 0.7, 
          #'missing':-999,
          'objective':'multi:softmax',
          'num_class':4
         }

PIPELINE_NAME = 'xgboost_classifier'

xgboost_pipe = Pipeline([
                ('cme1', categ_missing_encoder(config.ID_FEATURES)),
                ('fc', feature_creation()),
                ('oc', outlier_capping(distribution='quantiles')),
                ('cme2', categ_missing_encoder()),
                ('rle', rare_label_encoder(0.0001)),                
                ('ani', ArbitraryNumberImputer()),
                ('le', label_encoder()),
                ('sd', selected_drop_features()),
                ('rc', remove_constant()),
                ('rqc', remove_quasi_constant()),
                ('rcf', remove_correlated_features()),
                ('scaler', StandardScaler()),
                ('xgb', xgb.XGBClassifier(**xgboost_params, n_estimators = 10000, n_jobs=6))
            ])
'''
lightgbm_params = {
                      'objective': 'multiclass',
                      'num_class':4,
                      'metric': 'multi_logloss' }
                      #"num_leaves" : 60,
                      #"max_depth": -1,
                      #"learning_rate" : 0.01,
                      #"bagging_fraction" : 0.9,  # subsample
                      #"feature_fraction" : 0.9,  # colsample_bytree
                      #"bagging_freq" : 5,        # subsample_freq
                      #"bagging_seed" : 2018,
                      #"verbosity" : -1 }

lightgbm_pipe = Pipeline([
                ('fc', feature_creation()),
                #('d2c', discrete_to_categ()),
                ('rce', rare_label_encoder()),
                ('cme', categ_missing_encoder()),
                ('ani', ArbitraryNumberImputer()),
                ('le', label_encoder()),
                ('rc', remove_constant()),
                ('rqc', remove_quasi_constant()),
                #('rd', remove_duplicates()),
                ('rcf', remove_correlated_features(threshold=0.99)),
                ('sd', selected_drop_features()),
                ('oc', outlier_capping()),
                ('scaler', StandardScaler()),
                ('lgbm', lgb.LGBMClassifier( **lightgbm_params, n_estimators = 2616))
            ])
'''