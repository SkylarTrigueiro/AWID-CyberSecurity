# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:57:06 2020

@author: skyst
"""

import numpy as np
import pandas as pd

from tf_ann_model.preprocessing.data_management import load_pipeline
from tf_ann_model.config import config
from tf_ann_model.processing.validation import validate_inputs
from tf_ann_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_cs_pipe = load_pipeline( file_name = pipeline_file_name )

def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _cs_pipe.predict(validated_data)
    output = prediction
    
    results = {'predictions':output, 'version':_version}
    
    _logger.info(
            f'Making predictions with model versions: {_version} '
            f'Inputs: {validated_data} '
            f'Predictions: {results}')
    
    return results