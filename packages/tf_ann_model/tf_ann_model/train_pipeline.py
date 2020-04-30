import numpy as np


from tf_ann_model import pipeline
from tf_ann_model.processing.data_management import load_dataset, save_pipeline_keras, prepare_data
from tf_ann_model.processing.feat_eng_categ import categ_missing_encoder
from tf_ann_model.config import config
from tf_ann_model.processing.feat_creation import feature_creation
from tf_ann_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

    
def run_training(save_result: bool = True):
    """Train the model"""
    
    #read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    
    print('data shape before', data.shape)
    X_train, y_train = prepare_data(data, True)
    print('X_train.shape after', X_train.shape)
    
    pipeline.tf_ann_pipe.fit(X_train, y_train)
    
    if save_result:
        save_pipeline_keras(pipeline.tf_ann_pipe)
    
if __name__ == '__main__':
     run_training(save_result=True)