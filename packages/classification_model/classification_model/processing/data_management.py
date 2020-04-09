import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from classification_model.config import config
from classification_model import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str ) -> pd.DataFrame:
    
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}', header = 0, names = headers, na_values=['?'])
    
    return _data

def get_target(df):

    y = df[config.TARGET]
    df.drop(config.TARGET, axis = 1, inplace = True)
    
    return df, y

def save_pipeline(*, pipeline_to_persist) -> None:
    
    """Persist the pipeline.
    
    Saves the versioned moel, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can
    be called, and we know exactly how it was build.
    """
    
    # Prepare versioned save file name
    save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    
    print('saved pipeline')
    
def load_pipeline(*, file_name: str ) -> Pipeline:
    """Load a persisted pipeline"""
    
    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    
    return saved_pipeline

def partition_features(df):
    
    FEATURES = list(df.columns)
    CATEG = [ var for var in FEATURES if df[var].dtype == 'O' ]
    NUMERIC = [var for var in FEATURES if df[var].dtype != 'O'  ]
    
    return NUMERIC, CATEG

def downsample(df):
    
    from sklearn.utils import resample
    
    n = sum(df[config.TARGET].value_counts()[1:])
    
    df_majority = df[df[config.TARGET]=='normal']
    df_minority = df[df[config.TARGET]!='normal']
    
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=n,
                                       random_state=94019)
    
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_downsampled.sort_index(inplace=True)
    
    return df_downsampled

def simple_split(data, y, test_size=0.1):
    
    n = int(np.floor(test_size*len(y)))
    
    X_train = data[:n]
    X_test  = data[n:]
    y_train = y[:n]
    y_test  = y[n:]
    
    return X_train, X_test, y_train, y_test

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


