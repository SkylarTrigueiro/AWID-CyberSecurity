import pathlib
import os

import pandas as pd
import tf_ann_model

pd.options.display.max_rows=20
pd.options.display.max_columns = 20

#PWD = os.path.dirname(os.path.abspath(__file__))
#PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
#DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')
#TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')

PACKAGE_ROOT = pathlib.Path(tf_ann_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'AWID-CLS-R-Tst.csv'
TRAINING_DATA_FILE = 'AWID-CLS-R-Trn.csv'
TARGET = 'class'

# model
MODEL_NAME = 'tf_ann_model'
PIPELINE_NAME = 'ann_pipe'
CLASSES_NAME = 'classes'
ENCODER_NAME = 'encoder'

# Model Parameters
INCLUDE_VALIDATION_DATA = True
DOWNSAMPLE_DATA = True

# ANN Parameters
BATCH_SIZE = 1000
EPOCHS = 10

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()
    
MODEL_FILE_NAME = f'{MODEL_NAME}_{_version}.h5'
#MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
#PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)

CLASSES_FILE_NAME = f'{CLASSES_NAME}_{_version}.pkl'
#CLASSES_PATH = os.path.join(TRAINED_MODEL_DIR, CLASSES_FILE_NAME)

ENCODER_FILE_NAME = f'{ENCODER_NAME}_{_version}.pkl'
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)

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

CONSTANT_FEATURES = []

QUASI_CONSTANT_FEATURES = []

DUPLICATE_FEATURES = []

TIME_FEATURES = ['frame.time_epoch',
                 'passed1second',  
               ]

ID_FEATURES = ['wlan.wep.iv', 
               'wlan.wep.icv', 
               'wlan.ta', 
               'wlan.ra',
               'wlan.ra',
               'wlan.da']