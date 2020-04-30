from sklearn.model_selection import train_test_split

from classification_model import pipeline
from classification_model.processing.data_management import load_dataset, save_pipeline, prepare_data 
from classification_model.config import config
from classification_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

    
def run_training() -> None:
    """Train the model"""
    
    #read training data
    print('Loading data')
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    test = load_dataset(file_name=config.TESTING_DATA_FILE)
    X_train, y_train = prepare_data(data,True)
    print('data loaded and prepared')

    
    if( config.INCLUDE_VALIDATION_DATA == True ):
        print(' ')
        print('Training with validation set')
        X_test, y_test = prepare_data(test, False)
        pipeline.data_pipe.fit(X_train,y_train)
        X_test = pipeline.data_pipe.transform(X_test)
    
        fit_params = {
            'xgb__eval_set': [(X_test,y_test)],
            'xgb__early_stopping_rounds': 50
            }
    
        pipeline.xgboost_pipe.fit(X_train, y_train, **fit_params)
        
    else:
        print(' ')
        print('Training without validation set')
        pipeline.xgboost_pipe.fit(X_train, y_train)
    
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.xgboost_pipe)
    
if __name__ == '__main__':
    run_training()