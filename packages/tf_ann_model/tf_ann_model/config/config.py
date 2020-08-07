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



# model
MODEL_NAME = 'tf_cnn_model'
PIPELINE_NAME = 'cnn_pipe'
ENCODER_NAME = 'encoder'

# ANN Parameters
BATCH_SIZE = 1000
EPOCHS = 45

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()
    
MODEL_FILE_NAME = f'{MODEL_NAME}_{_version}.h5'
#MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
#PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)

ENCODER_FILE_NAME = f'{ENCODER_NAME}_{_version}.pkl'
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)
