from sklearn.pipeline import Pipeline

from tf_ann_model.processing.feat_eng_categ import discrete_to_categ, one_hot_encoder, categ_missing_encoder, rare_label_encoder, label_encoder
from tf_ann_model.processing.feat_eng_num import outlier_capping, ArbitraryNumberImputer
from tf_ann_model.processing.feat_creation import feature_creation
from tf_ann_model.processing.feat_selection import remove_constant, remove_quasi_constant, remove_duplicates, selected_drop_features, remove_correlated_features
from tf_ann_model.config import config
from sklearn.preprocessing import StandardScaler
from tf_ann_model.model import ann_classifier
from sklearn.multiclass import OneVsRestClassifier

tf_ann_pipe = Pipeline([
                ('oc', outlier_capping(distribution='quantiles')),
                ('cme2', categ_missing_encoder()),
                ('rle', rare_label_encoder(0.0001)),                
                ('ani', ArbitraryNumberImputer()),
                ('sd', selected_drop_features()),
                #('ohe', one_hot_encoder(max_labels=256)),
                ('le', label_encoder()),
                ('rc', remove_constant()),
                ('rqc', remove_quasi_constant()),
                ('rcf', remove_correlated_features()),
                ('scaler', StandardScaler()),
                ('classifier', ann_classifier )
            ])