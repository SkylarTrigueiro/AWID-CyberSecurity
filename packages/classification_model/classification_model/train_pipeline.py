from sklearn.model_selection import train_test_split

from classification_model import pipeline
from classification_model.processing.data_management import (
        load_dataset, save_pipeline, get_target )
from classification_model.config import config
from classification_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

    
def run_training() -> None:
    """Train the model"""
    
    #read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    X_train, y_train = get_target(data)
    
    pipeline.xgboost_pipe.fit(X_train, y_train)
    
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.xgboost_pipe)
    
if __name__ == '__main__':
    run_training()