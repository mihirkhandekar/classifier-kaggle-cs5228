import math
import os
import sys
import time

import keras.backend as K
import lightgbm as gbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf
import random
from keras import regularizers
from keras.callbacks.callbacks import (EarlyStopping, LearningRateScheduler,
                                       ModelCheckpoint, ReduceLROnPlateau,
                                       TerminateOnNaN)
from keras.layers import (LSTM, Activation, BatchNormalization, Bidirectional,
                          Concatenate, Conv1D, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D, Input, MaxPooling1D, Multiply)
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.regularizers import l2
from scipy import stats
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import (RFE, SelectKBest, SelectPercentile,
                                       chi2, f_classif)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier

sparse_index = [i for i in range(40)]
sparse_index = [i for i in sparse_index if i not in [0, 4, 10, 25]]

prefix_path = '../data'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
print('Labels', labels.describe())
iterations = 6


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'binary_logloss'},
    'num_leaves': 196,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'num_threads': 4,
    'bagging_freq': 5,
    'verbose': 0,
    "tree_learner": "feature"
}

ar_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l1', 'l2', 'binary_logloss'},
    'num_leaves': 196,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'num_threads': 4,
    'bagging_freq': 5,
    'verbose': 0,
    "tree_learner": "feature"
}



def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    return sparse_x


def __get_model(lgb_train, lgb_eval, x_train, y_train, param):
    model = gbm.train(param, 
                      lgb_train,
                      num_boost_round=800,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=60)
    return model


def __extract_features(features):
    sparse_x = __preprocess_feature(np.array(features))

    # For each feature, we find average of all values and replace all NaN with that value
    sparse_means = np.nanmean(
        np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_max = np.nanmax(sparse_x, axis=0)
    
    sparse_medians = np.nanmedian(
        np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_nans = np.count_nonzero(np.isnan(sparse_x), axis=0)
    sparse_vars = np.nanvar(sparse_x, axis=0)
    sparse_modes = stats.mode(sparse_x)[0][0]


    # sp_features = np.concatenate([sparse_modes, sparse_vars, sparse_max, sparse_x[0], sparse_x[-1]])
    sp_features = np.concatenate([sparse_modes, sparse_vars, sparse_max, sparse_medians, sparse_x[0], sparse_x[-1]])
    return np.nan_to_num(sp_features)


test_Y = []


batch_size = 256
train_samples = 30336
test_samples = 10000
no_epochs = 100
max_time = 50



for it in range(iterations):    
    print('Starting Iteration ', it)
    X0 = []
    y0 = []
    X1 = []
    y1 = []
    Xn = []
    yn = []
    ones = len(labels.loc[labels['label'] == 1])

    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])

    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        if label == 0 and ones > 0:
            ones = ones - 0.65
        if ones <= 0 and label == 0:
            continue
        features = np.load(prefix_path + '/train/train/' +
                           str(train_label['Id']) + '.npy')

        sp_features = __extract_features(features)
        if features[0][0] == 0:
            X0.append(sp_features)
            y0.append(label)
        elif features[0][0] == 1:
            X1.append(sp_features)
            y1.append(label)
        else:
            Xn.append(sp_features)
            yn.append(label)

    from sklearn.feature_selection import VarianceThreshold

    sel0 = VarianceThreshold(threshold=0.15)
    sel0.fit_transform(X0)
    print('Variances0', [(num, item) for (num, item) in enumerate(sel0.variances_)])
    sel1 = VarianceThreshold(threshold=0.15)
    sel1.fit_transform(X1)
    print('Variances1', [(num, item) for (num, item) in enumerate(sel0.variances_)])
    seln = VarianceThreshold(threshold=0.15)
    seln.fit_transform(Xn)
    print('Variancesn', [(num, item) for (num, item) in enumerate(sel0.variances_)])

    keep_feat = [index for index, variances in enumerate(zip(sel0.variances_, sel1.variances_, seln.variances_)) if variances[0] > 0.8 and variances[1] > 0.8 and variances[2] > 0.8]

    print('Useful features', keep_feat)

    X0 = np.array(X0)[:, keep_feat]
    y0 = np.array(y0)
    

    X1 = np.array(X1)[:, keep_feat]
    y1 = np.array(y1)
    
    
    Xn = np.array(Xn)[:, keep_feat]
    yn = np.array(yn)

    x_train_0, x_val_0, y_train_0, y_val_0 = train_test_split(X0, y0, test_size=0.20)
    x_train_1, x_val_1, y_train_1, y_val_1 = train_test_split(X1, y1, test_size=0.20)
    x_train_n, x_val_n, y_train_n, y_val_n = train_test_split(Xn, yn, test_size=0.20)

    lgb_train_0 = gbm.Dataset(x_train_0, y_train_0)
    gbm_model_0 = __get_model(lgb_train_0, gbm.Dataset(x_val_0, y_val_0, reference=lgb_train_0), x_train_0, y_train_0, params)
    lgb_train_1 = gbm.Dataset(x_train_1, y_train_1)
    gbm_model_1 = __get_model(lgb_train_1, gbm.Dataset(x_val_1, y_val_1, reference=lgb_train_1), x_train_1, y_train_1, ar_params)
    lgb_train_n = gbm.Dataset(x_train_n, y_train_n)
    gbm_model_n = __get_model(lgb_train_n, gbm.Dataset(x_val_n, y_val_n, reference=lgb_train_n), x_train_n, y_train_n, params)
    
    '''
    xgb_model_0 = XGBClassifier(alpha=4, base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.1,
              learning_rate=0.05, max_delta_step=0, max_depth=9,
              min_child_weight=1, missing=-1, n_estimators=500, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.9, tree_method='hist', verbosity=1)
    
    xgb_model_1 = XGBClassifier(n_estimators=2000,
                max_depth=12, 
                learning_rate=0.02, 
                subsample=0.8,
                colsample_bytree=0.4, 
                missing=-1, 
                eval_metric='auc',)
    
    xgb_model_n = XGBClassifier(alpha=4, base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.1,
              learning_rate=0.05, max_delta_step=0, max_depth=9,
              min_child_weight=1, missing=-1, n_estimators=500, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.9, tree_method='hist', verbosity=1)
    
    xgb_model_0.fit(x_train_0, y_train_0, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(x_val_0, y_val_0)], verbose=True)
    
    xgb_model_1.fit(x_train_1, y_train_1, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(x_val_1, y_val_1)], verbose=True)
    
    xgb_model_n.fit(x_train_n, y_train_n, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(x_val_n, y_val_n)], verbose=True)
    '''

    '''
    cat_0 = CatBoostClassifier(iterations=5)
    cat_0.fit(x_train_0, y_train_0)
    
    cat_1 = CatBoostClassifier(iterations=5, class_weights=[0.9, 1.2])
    cat_1.fit(x_train_1, y_train_1)
    
    cat_n = CatBoostClassifier(iterations=5)
    cat_n.fit(x_train_n, y_train_n)
    '''

    '''

    bag_model_1 = BaggingClassifier(n_estimators=64, )

    def get_wt(sample):
        if sample == 0:
            return 0.9
        else:
            return 1.2
    #sample_weight = [get_wt(sample) for sample in y_train_1]
    #bag_model_1.fit(x_train_1, y_train_1, sample_weight=sample_weight)
    '''

    y_pred_0 = gbm_model_0.predict(x_val_0) 
    xg_predictions_0 = [int(round(value)) for value in y_pred_0]

    y_pred_1 = gbm_model_1.predict(x_val_1)
    #y_pred_1 = gbm_model_1.predict(x_val_1)
    xg_predictions_1 = [int(round(value)) for value in y_pred_1]

    y_pred_n = gbm_model_n.predict(x_val_n)
    xg_predictions_n = [int(round(value)) for value in y_pred_n]

    test_X_0 = []
    test_X_1 = []
    test_X_n = []
    
    y_test_dl = []
    count_0 = 0
    count_1 = 0
    count_n = 0
    for fileno in range(10000):
        test_features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
        
        sp_features = __extract_features(test_features)
        tX = np.array([sp_features])[:, keep_feat]
        if test_features[0][0] == 0:
            prediction = gbm_model_0.predict(tX)
            y_test_dl.extend(prediction)
            count_0 += 1
        elif test_features[0][0] == 1:
            prediction = gbm_model_1.predict(tX)
            # prediction = gbm_model_1.predict(tX)
            y_test_dl.extend(prediction)
            count_1 += 1
        else:
            prediction = gbm_model_n.predict(tX)
            y_test_dl.extend(prediction)
            count_n += 1

    test_Y.append(y_test_dl)

    print('### . 0={}, 1={}, n={}'.format(count_0, count_1, count_n))

    print('0 Round validation ROC-AUC = {}, accuracy = {}, recall = {}, precision = {}'.format(roc_auc_score(y_val_0, y_pred_0),
                                                                                            accuracy_score(y_val_0, xg_predictions_0), recall_score(
    y_val_0, xg_predictions_0),
    precision_score(y_val_0, xg_predictions_0)))
    print('1 Round validation ROC-AUC = {}, accuracy = {}, recall = {}, precision = {}'.format(roc_auc_score(y_val_1, y_pred_1),
                                                                                             accuracy_score(y_val_1, xg_predictions_1), recall_score(
        y_val_1, xg_predictions_1),
        precision_score(y_val_1, xg_predictions_1)))
    print('n Round validation ROC-AUC = {}, accuracy = {}, recall = {}, precision = {}'.format(roc_auc_score(y_val_n, y_pred_n),
                                                                                             accuracy_score(y_val_n, xg_predictions_n), recall_score(
        y_val_n, xg_predictions_n),
        precision_score(y_val_n, xg_predictions_n)))


test_Y = np.array(test_Y)
print('Results', test_Y.shape)

print(test_Y)
test_Y = np.average(test_Y, axis=0)
print(test_Y.shape, test_Y)

df = pd.DataFrame()
df["Predicted"] = test_Y
df.to_csv('outputs/ml-output-' + str(time.time()) + '.csv', index_label="Id")
