import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

from tf_ann_model.config import config
from tf_ann_model.processing.feat_creation import feature_creation
from tf_ann_model.processing.feat_eng_categ import categ_missing_encoder
from tf_ann_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str ) -> pd.DataFrame:
    
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}', header = 0, names = headers, na_values=['?'])
    
    return _data

def prepare_data(df, train_data):

    X,y = get_target(df)  
    
    cme = categ_missing_encoder(config.ID_FEATURES)
    df = cme.fit_transform(df)
    X,y = get_target(df)
    X,y = feature_creation(X, y)
    y = prepare_target_keras(y, train_data)

    return X,y

def restore_target(y):
    
    import numpy as np
    from sklearn.externals import joblib
    
    from 
    y = np.argmax(y, axis=1)
    encoder = joblib.load(config.ENCODER_PATH)
    y = encoder.inverse_transform(y)
    
    return y

def get_target(df):

    y = df[config.TARGET]
    
    return df, y

def save_pipeline_keras(pipeline) -> None:
    """Persist keras model to disk."""

    # Save the Keras model first:
    pipeline.named_steps['classifier'].model.save( config.TRAINED_MODEL_DIR / config.MODEL_FILE_NAME)

    # This hack allows us to save the sklearn pipeline:
    pipeline.named_steps['classifier'].model = None
    
    # Finally, save the pipeline:
    joblib.dump(pipeline, config.TRAINED_MODEL_DIR / config.PIPELINE_FILE_NAME)

    remove_old_pipelines(
        files_to_keep=[config.MODEL_FILE_NAME, config.ENCODER_FILE_NAME,
                       config.PIPELINE_FILE_NAME, config.CLASSES_FILE_NAME])
    
    print('saved pipeline')
    
def load_pipeline_keras() -> Pipeline:
    """Load a Keras Pipeline from disk."""

    # Load the pipeline first:
    pipeline = joblib.load(config.TRAINED_MODEL_DIR /config.PIPELINE_FILE_NAME)

    # Then, load the Keras model:
    pipeline.named_steps['classifier'].model = load_model(config.TRAINED_MODEL_DIR /config.MODEL_FILE_NAME)

    return pipeline
    
def load_encoder() -> LabelEncoder:
    encoder = joblib.load(config.ENCODER_PATH)

    return encoder

def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines, models, encoders and classes.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in Path(config.TRAINED_MODEL_DIR).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def partition_features(df):
    
    FEATURES = list(df.columns)
    CATEG = [ var for var in FEATURES if df[var].dtype == 'O' ]
    NUMERIC = [var for var in FEATURES if df[var].dtype != 'O'  ]
    
    return NUMERIC, CATEG

def simple_split(data, y, test_size=0.1):
    
    n = int(np.floor(test_size*len(y)))
    
    X_train = data[:n]
    X_test  = data[n:]
    y_train = y[:n]
    y_test  = y[n:]
    
    return X_train, X_test, y_train, y_test

def prepare_target_keras(y, train_data):
    
    from sklearn.preprocessing import LabelEncoder
    import tensorflow as tf
    from sklearn.externals import joblib
    
    if( train_data ):
        # If we're training the data, then the encoder shouldn't exist yet.
        encoder = LabelEncoder()        
    else:
        # If we're not working with training data, then the encoder should already exist
        encoder = joblib.load(config.ENCODER_PATH)
    
    encoder.fit(y)
    
    y = encoder.transform(y)
    y = tf.keras.utils.to_categorical( y, num_classes=4)
    
    joblib.dump(encoder, config.ENCODER_PATH)
    
    return y
headers = [
'frame.interface_id',
'frame.dlt',
'frame.offset_shift',
'frame.time_epoch',
'frame.time_delta',
'frame.time_delta_displayed',
'frame.time_relative',
'frame.len',
'frame.cap_len',
'frame.marked',
'frame.ignored',
'radiotap.version',
'radiotap.pad',
'radiotap.length',
'radiotap.present.tsft',
'radiotap.present.flags',
'radiotap.present.rate',
'radiotap.present.channel',
'radiotap.present.fhss',
'radiotap.present.dbm_antsignal',
'radiotap.present.dbm_antnoise',
'radiotap.present.lock_quality',
'radiotap.present.tx_attenuation',
'radiotap.present.db_tx_attenuation',
'radiotap.present.dbm_tx_power',
'radiotap.present.antenna',
'radiotap.present.db_antsignal',
'radiotap.present.db_antnoise',
'radiotap.present.rxflags',
'radiotap.present.xchannel',
'radiotap.present.mcs',
'radiotap.present.ampdu',
'radiotap.present.vht',
'radiotap.present.reserved',
'radiotap.present.rtap_ns',
'radiotap.present.vendor_ns',
'radiotap.present.ext',
'radiotap.mactime',
'radiotap.flags.cfp',
'radiotap.flags.preamble',
'radiotap.flags.wep',
'radiotap.flags.frag',
'radiotap.flags.fcs',
'radiotap.flags.datapad',
'radiotap.flags.badfcs',
'radiotap.flags.shortgi',
'radiotap.datarate',
'radiotap.channel.freq',
'radiotap.channel.type.turbo',
'radiotap.channel.type.cck',
'radiotap.channel.type.ofdm',
'radiotap.channel.type.2ghz',
'radiotap.channel.type.5ghz',
'radiotap.channel.type.passive',
'radiotap.channel.type.dynamic',
'radiotap.channel.type.gfsk',
'radiotap.channel.type.gsm',
'radiotap.channel.type.sturbo',
'radiotap.channel.type.half',
'radiotap.channel.type.quarter',
'radiotap.dbm_antsignal',
'radiotap.antenna',
'radiotap.rxflags.badplcp',
'wlan.fc.type_subtype',
'wlan.fc.version',
'wlan.fc.type',
'wlan.fc.subtype',
'wlan.fc.ds',
'wlan.fc.frag',
'wlan.fc.retry',
'wlan.fc.pwrmgt',
'wlan.fc.moredata',
'wlan.fc.protected',
'wlan.fc.order',
'wlan.duration',
'wlan.ra',
'wlan.da',
'wlan.ta',
'wlan.sa',
'wlan.bssid',
'wlan.frag',
'wlan.seq',
'wlan.bar.type',
'wlan.ba.control.ackpolicy',
'wlan.ba.control.multitid',
'wlan.ba.control.cbitmap',
'wlan.bar.compressed.tidinfo',
'wlan.ba.bm',
'wlan.fcs_good',
'wlan_mgt.fixed.capabilities.ess',
'wlan_mgt.fixed.capabilities.ibss',
'wlan_mgt.fixed.capabilities.cfpoll.ap',
'wlan_mgt.fixed.capabilities.privacy',
'wlan_mgt.fixed.capabilities.preamble',
'wlan_mgt.fixed.capabilities.pbcc',
'wlan_mgt.fixed.capabilities.agility',
'wlan_mgt.fixed.capabilities.spec_man',
'wlan_mgt.fixed.capabilities.short_slot_time',
'wlan_mgt.fixed.capabilities.apsd',
'wlan_mgt.fixed.capabilities.radio_measurement',
'wlan_mgt.fixed.capabilities.dsss_ofdm',
'wlan_mgt.fixed.capabilities.del_blk_ack',
'wlan_mgt.fixed.capabilities.imm_blk_ack',
'wlan_mgt.fixed.listen_ival',
'wlan_mgt.fixed.current_ap',
'wlan_mgt.fixed.status_code',
'wlan_mgt.fixed.timestamp',
'wlan_mgt.fixed.beacon',
'wlan_mgt.fixed.aid',
'wlan_mgt.fixed.reason_code',
'wlan_mgt.fixed.auth.alg',
'wlan_mgt.fixed.auth_seq',
'wlan_mgt.fixed.category_code',
'wlan_mgt.fixed.htact',
'wlan_mgt.fixed.chanwidth',
'wlan_mgt.fixed.fragment',
'wlan_mgt.fixed.sequence',
'wlan_mgt.tagged.all',
'wlan_mgt.ssid',
'wlan_mgt.ds.current_channel',
'wlan_mgt.tim.dtim_count',
'wlan_mgt.tim.dtim_period',
'wlan_mgt.tim.bmapctl.multicast',
'wlan_mgt.tim.bmapctl.offset',
'wlan_mgt.country_info.environment',
'wlan_mgt.rsn.version',
'wlan_mgt.rsn.gcs.type',
'wlan_mgt.rsn.pcs.count',
'wlan_mgt.rsn.akms.count',
'wlan_mgt.rsn.akms.type',
'wlan_mgt.rsn.capabilities.preauth',
'wlan_mgt.rsn.capabilities.no_pairwise',
'wlan_mgt.rsn.capabilities.ptksa_replay_counter',
'wlan_mgt.rsn.capabilities.gtksa_replay_counter',
'wlan_mgt.rsn.capabilities.mfpr',
'wlan_mgt.rsn.capabilities.mfpc',
'wlan_mgt.rsn.capabilities.peerkey',
'wlan_mgt.tcprep.trsmt_pow',
'wlan_mgt.tcprep.link_mrg',
'wlan.wep.iv',
'wlan.wep.key',
'wlan.wep.icv',
'wlan.tkip.extiv',
'wlan.ccmp.extiv',
'wlan.qos.tid',
'wlan.qos.priority',
'wlan.qos.eosp',
'wlan.qos.ack',
'wlan.qos.amsdupresent',
'wlan.qos.buf_state_indicated_1',
'wlan.qos.bit4',
'wlan.qos.txop_dur_req',
'wlan.qos.buf_state_indicated_2',
'data.len',
'class'
]


