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

sparse_index = [i for i in range(40)]
sparse_index = [i for i in sparse_index if i not in [4, 10, 25]]

prefix_path = '..'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
print('Labels', labels.describe())
iterations = 6

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    'num_leaves': 128,
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
    sp_features = np.concatenate([sparse_modes, sparse_vars, [sparse_x.shape[0]], sparse_x[0], sparse_x[-1]])
    return np.nan_to_num(sp_features, -1)


test_X = []
test_Y = []

# Read test file
for fileno in range(10000):
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
    sp_features = __extract_features(features)
    test_X.append(sp_features)

test_X = np.array(test_X)

batch_size = 256
train_samples = 30336
test_samples = 10000
no_epochs = 100
max_time = 50


for it in range(iterations):
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

        sp_features = __extract_features(features)
        X.append(sp_features)
        y.append(label)

    #print(np.nan_to_num(X), y)
    # X_best = f_classif(np.nan_to_num(X), y) #.fit_transform(X, y)

    #print(X_best, np.array(X_best).shape)
    #mask = X_best.get_support()
    # print(mask)
    # for bool, feature in zip(mask, X):
    #    if bool:
    #        new_X.append(feature)

    '''if incorrect_x is not None:
        X_sparse.extend(incorrect_x)
        y.extend(incorrect_y)
    '''
    X = np.array(X)
    y = np.array(y)

    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=0.15)
    sel.fit_transform(X)
    #print('Variances', sel.variances_, len(sel.variances_))
    top_features = np.argsort(sel.variances_)[::-1][0:155]
    #fs = SelectPercentile(f_classif, percentile=70)
    #fs.fit_transform(X, y)
    #top_features = np.argsort(fs.scores_)[::-1][0:150]
    #print('Top features in round : ', top_features)

    X = X[:, top_features]

    round_test_X = test_X[:, top_features]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    lgb_train = gbm.Dataset(x_train, y_train)
    lgb_eval = gbm.Dataset(x_val, y_val, reference=lgb_train)

    gbm_model = __get_model(lgb_train, lgb_eval, x_train, y_train)
    #xgb_model = XGBRegressor(objective="binary:logistic", subsample=0.5,
    #                         learning_rate=0.005, max_depth=8,
    #                         min_child_weight=1.8, n_estimators=4000,
    #                         reg_alpha=0.1, reg_lambda=0.3, gamma=0.01,
    #                         silent=1, random_state=7, nthread=-1)
    # xgb_model.fit(x_train, y_train)
    # from sklearn import tree
    # dt_model = tree.DecisionTreeRegressor()
    # dt_model.fit(x_train, y_train)

    xgb_model = XGBClassifier(learning_rate=0.1, scale_pos_weight = 9, max_depth=7, min_child_weight=1, subsample=0.6, n_estimators=800, gamma=0.8, colsample_bytree=0.8)
    xgb_model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(x_val, y_val)], verbose=True)


    y_pred = (gbm_model.predict(x_val) + xgb_model.predict(x_val)) / 2
    print(y_pred.shape)

    xg_predictions = [int(round(value)) for value in y_pred]
    print('Round validation ROC-AUC = {}, accuracy = {}, recall = {}, precision = {}'.format(roc_auc_score(y_val, y_pred),
                                                                                             accuracy_score(y_val, xg_predictions), recall_score(
        y_val, xg_predictions),
        precision_score(y_val, xg_predictions)))

    y_test_dl = (gbm_model.predict(round_test_X) + xgb_model.predict(round_test_X)) / 2
    test_Y.append(y_test_dl)

test_Y = np.array(test_Y)
print('Results', test_Y.shape)

print(test_Y)
test_Y = np.average(test_Y, axis=0)
print(test_Y.shape, test_Y)

df = pd.DataFrame()
df["Predicted"] = test_Y
df.to_csv('outputs/ml-output-' + str(time.time()) + '.csv', index_label="Id")
