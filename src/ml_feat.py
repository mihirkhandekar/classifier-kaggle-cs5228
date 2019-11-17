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

sparse_index = [i for i in range(40)]
sparse_index = [i for i in sparse_index if i not in [4, 10, 25]]

prefix_path = 'data'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
print('Labels', labels.describe())
iterations = 6

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    'num_leaves': 256,
    'min_data_in_leaf': 106,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'num_threads': 4,
    'bagging_freq': 10,
    'verbose': 0,
    "tree_learner": "feature"
}


def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    return sparse_x


def __get_model(lgb_train, lgb_eval, x_train, y_train):
    model = gbm.train(params, 
                      lgb_train,
                      num_boost_round=400,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=30)
    return model
    '''
    clf = AdaBoostRegressor(n_estimators=500)
    clf.fit(X, y)
    return clf'''


def __extract_features(features, rand_indices):
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
    sp_features = np.concatenate([sparse_modes, sparse_vars, [sparse_x.shape[0]], sparse_x[0], sparse_x[-1]])
    return np.nan_to_num(sp_features, 0)


test_Y = []


batch_size = 256
train_samples = 30336
test_samples = 10000
no_epochs = 100
max_time = 50



for it in range(iterations):
    rand_indices = random.sample(range(1, 37), 25)
    test_X = []
    # Read test file
    test_X_features = []
    for fileno in range(10000):
        test_features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
        
        sp_features = __extract_features(test_features, rand_indices)
        
        test_X.append(sp_features)
        test_X_features.extend(test_features[:, 0])

    test_X = np.array(test_X)
    print('TEX shape', np.array(test_X_features).shape, np.unique(np.array(test_X_features)))
    
    print('Starting Iteration ', it)
    X = []
    y = []
    ones = len(labels.loc[labels['label'] == 1])

    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])

    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        if label == 0 and ones > 0:
            ones = ones - 0.8
        if ones <= 0 and label == 0:
            continue
        features = np.load(prefix_path + '/train/train/' +
                           str(train_label['Id']) + '.npy')

        sp_features = __extract_features(features, rand_indices)
        X.append(sp_features)
        y.append(label)


    X = np.array(X)
    y = np.array(y)


    round_test_X = test_X#[:, top_features]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    lgb_train = gbm.Dataset(x_train, y_train)
    lgb_eval = gbm.Dataset(x_val, y_val, reference=lgb_train)

    gbm_model = __get_model(lgb_train, lgb_eval, x_train, y_train)


    xgb_model = XGBClassifier(learning_rate=0.1, scale_pos_weight = 9, max_depth=7, min_child_weight=1, subsample=0.6, n_estimators=800, gamma=0.8, colsample_bytree=0.8)
    '''xgb_model = XGBClassifier(alpha=4, base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.1,
              learning_rate=0.05, max_delta_step=0, max_depth=9,
              min_child_weight=1, missing=-1, n_estimators=500, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=0.9, tree_method='hist', verbosity=1)
    '''
    #xgb_model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(x_val, y_val)], verbose=True)

    cat_model = CatBoostClassifier()
    #cat_model.fit(x_train, y_train)

    #from sklearn.ensemble import BaggingClassifier

    #bag_model = BaggingClassifier()
    #bag_model.fit(x_train, y_train)

    y_pred = gbm_model.predict(x_val)
    print(y_pred.shape)

    xg_predictions = [int(round(value)) for value in y_pred]
    print('Round validation ROC-AUC = {}, accuracy = {}, recall = {}, precision = {}'.format(roc_auc_score(y_val, y_pred),
                                                                                             accuracy_score(y_val, xg_predictions), recall_score(
        y_val, xg_predictions),
        precision_score(y_val, xg_predictions)))

    y_test_dl = gbm_model.predict(round_test_X)
    test_Y.append(y_test_dl)

test_Y = np.array(test_Y)
print('Results', test_Y.shape)

print(test_Y)
test_Y = np.average(test_Y, axis=0)
print(test_Y.shape, test_Y)

df = pd.DataFrame()
df["Predicted"] = test_Y
df.to_csv('outputs/ml-output.csv', index_label="Id")
