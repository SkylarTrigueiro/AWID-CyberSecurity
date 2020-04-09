import pathlib

import pandas as pd

import classification_model

pd.options.display.max_rows=20
pd.options.display.max_columns = 20

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = 'AWID-CLS-R-Tst.csv'
TRAINING_DATA_FILE = 'AWID-CLS-R-Trn.csv'
TARGET = 'class'

# model
MODEL = 'xgboost'
if MODEL == 'xgboost':
    PIPELINE_SAVE_FILE = 'xgboost_classifier_model'
elif MODEL == 'lightgbm':
    PIPELINE_SAVE_FILE = 'lightgbm_classifier_model'
elif MODEL == 'ann':
    PIPELINE_SAVE_FILE = 'ann_classifier_model'
    
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

CONSTANT_FEATURES = ['frame.interface_id',
 'frame.dlt',
 'frame.offset_shift',
 'frame.marked',
 'frame.ignored',
 'radiotap.version',
 'radiotap.pad',
 'radiotap.present.rate',
 'radiotap.present.fhss',
 'radiotap.present.dbm_antnoise',
 'radiotap.present.lock_quality',
 'radiotap.present.tx_attenuation',
 'radiotap.present.db_tx_attenuation',
 'radiotap.present.dbm_tx_power',
 'radiotap.present.db_antsignal',
 'radiotap.present.db_antnoise',
 'radiotap.present.xchannel',
 'radiotap.present.mcs',
 'radiotap.present.ampdu',
 'radiotap.present.vht',
 'radiotap.present.reserved',
 'radiotap.present.rtap_ns',
 'radiotap.present.vendor_ns',
 'radiotap.present.ext',
 'wlan.fc.version',
 'wlan.fc.order',
 'wlan.qos.buf_state_indicated_1']

QUASI_CONSTANT_FEATURES = ['radiotap.length',
 'wlan.fc.frag',
 'wlan.fc.pwrmgt',
 'wlan.fc.moredata',
 'wlan.bar.type',
 'wlan.ba.control.ackpolicy',
 'wlan.ba.control.multitid',
 'wlan_mgt.fixed.listen_ival',
 'wlan_mgt.fixed.current_ap',
 'wlan_mgt.fixed.status_code',
 'wlan_mgt.fixed.aid',
 'wlan_mgt.fixed.auth.alg',
 'wlan_mgt.fixed.auth_seq',
 'wlan_mgt.fixed.category_code',
 'wlan_mgt.fixed.htact',
 'wlan_mgt.fixed.chanwidth',
 'wlan_mgt.fixed.fragment',
 'wlan_mgt.fixed.sequence',
 'wlan_mgt.country_info.environment',
 'wlan_mgt.tcprep.trsmt_pow',
 'wlan_mgt.tcprep.link_mrg',
 'wlan.ba.bm_0',
 'wlan.ba.bm_1',
 'wlan.ba.bm_2',
 'wlan.ba.bm_3',
 'wlan.ba.bm_4',
 'wlan.ba.bm_5']

DUPLICATE_FEATURES = ['radiotap.channel.type.quarter',
 'radiotap.present.tsft',
 'radiotap.flags.datapad',
 'radiotap.channel.type.5ghz',
 'radiotap.flags.preamble',
 'frame.time_delta_displayed',
 'radiotap.channel.type.sturbo',
 'radiotap.rxflags.badplcp',
 'radiotap.channel.type.passive',
 'radiotap.flags.wep',
 'wlan.ba.control.cbitmap',
 'radiotap.flags.fcs',
 'radiotap.present.dbm_antsignal',
 'wlan.bar.compressed.tidinfo',
 'radiotap.channel.type.dynamic',
 'radiotap.flags.badfcs',
 'wlan.qos.buf_state_indicated_2',
 'radiotap.channel.type.turbo',
 'radiotap.channel.type.ofdm',
 'radiotap.channel.type.2ghz',
 'wlan.qos.txop_dur_req',
 'radiotap.flags.shortgi',
 'radiotap.channel.type.gsm',
 'frame.cap_len',
 'radiotap.channel.type.half',
 'radiotap.antenna',
 'wlan.fcs_good',
 'radiotap.flags.frag',
 'radiotap.present.antenna',
 'radiotap.channel.type.gfsk',
 'radiotap.present.flags',
 'radiotap.flags.cfp',
 'radiotap.present.channel',
 'radiotap.present.rxflags',
 'wlan.qos.priority']

TIME_FEATURES = ['frame.time_epoch',
                 'passed1second',  
               ]

ID_FEATURES = ['wlan.wep.iv', 
               'wlan.wep.icv', 
               'wlan.ta', 
               'wlan.ra',
               'wlan.ra',
               'wlan.da']