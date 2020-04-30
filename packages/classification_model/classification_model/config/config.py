import pathlib

import pandas as pd

import classification_model
import os

pd.options.display.max_rows=20
pd.options.display.max_columns = 20

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'AWID-CLS-R-Tst.csv'
TRAINING_DATA_FILE = 'AWID-CLS-R-Trn.csv'
TARGET = 'class'

# Model Parameters
INCLUDE_VALIDATION_DATA = True
DOWNSAMPLE_DATA = True

# model
MODEL = 'xgboost'
if MODEL == 'xgboost':
    PIPELINE_SAVE_FILE = 'xgboost_classifier_model'
elif MODEL == 'lightgbm':
    PIPELINE_SAVE_FILE = 'lightgbm_classifier_model'

# ann model
BATCH_SIZE = 100


"""Real Feature Parameters"""
ARBITRARY_REAL_MISSING_VALUE = -999
OUTLIER_DISTANCE = 1.5
OUTLIER_MEASURE = 'skew'

"""Discrete Feature Parameters"""
DISC_IS_CATEG = True
MAX_CAT_CARDINALITY = 20

"""Categorical Feature Parameters"""
RARE_THRESHOLD = 0.0003
OHE_TOP_X_FEATUERS = 40

TIME_FEATURES = ['frame.time_epoch',
                 'passed1second',  
               ]

ID_FEATURES = ['wlan.wep.iv', 
               'wlan.wep.icv', 
               'wlan.ta', 
               'wlan.ra',
               'wlan.ra',
               'wlan.da']