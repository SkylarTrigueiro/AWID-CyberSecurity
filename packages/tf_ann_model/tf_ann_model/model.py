from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tf_ann_model.config import config
from tf_ann_model.processing.data_management import load_dataset, prepare_data
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.externals import joblib

from tf_ann_model.processing.feat_eng_categ import categ_missing_encoder, rare_label_encoder, one_hot_encoder, label_encoder
from tf_ann_model.processing.feat_eng_num import outlier_capping, ArbitraryNumberImputer
from tf_ann_model.processing.feat_creation import feature_creation
from tf_ann_model.processing.feat_selection import remove_constant, remove_quasi_constant, selected_drop_features, remove_correlated_features
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import tensorflow as tf

def ann_model( input_shape=(40,), optimizer = 'adam', loss='categorical_crossentropy', metrics = 'accuracy'):
    
    model = models.Sequential()
    model.add( layers.Dense(32, activation = 'relu', input_shape=(input_shape)))
    model.add( layers.Dense(32, activation = 'relu'))
    model.add( layers.Dense(4, activation='softmax'))
    model.compile( optimizer=optimizer, loss=loss, metrics=[metrics])
    
    return model

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose = 0 )

checkpoint = ModelCheckpoint(config.TRAINED_MODEL_DIR,
                             monitor='acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

reduce_lr = ReduceLROnPlateau(monitor='acc',
                              factor=0.5,
                              patience=2,
                              verbose=1,
                              mode='max',
                              min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

fe_pipe = Pipeline([
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
                ('scaler', StandardScaler())
            ])

if config.INCLUDE_VALIDATION_DATA:
    
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    val = load_dataset(file_name=config.TESTING_DATA_FILE)

    X_train, y_train = prepare_data(data, True)
    X_val, y_val = prepare_data(val, False)


    fe_pipe.fit( X_train, y_train )
    X_val =fe_pipe.transform(X_val)

    ann_classifier = KerasClassifier(build_fn=ann_model,
                          batch_size= config.BATCH_SIZE,
                          epochs=config.EPOCHS,
                          validation_data = (X_val, y_val),
                          verbose=1,  # progress bar - required for CI job
                          callbacks=callbacks_list
                          )
    
else:
    
    ann_classifier = KerasClassifier(build_fn=ann_model,
                          batch_size= config.BATCH_SIZE,
                          epochs=config.EPOCHS,
                          verbose=1,  # progress bar - required for CI job
                          callbacks=callbacks_list
                          )

if __name__ == '__main__':
    model = ann_model()
    model.summary()
