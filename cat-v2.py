import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier

import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Conv2D, \
    Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
import lightgbm as gbm
from xgboost import XGBRegressor
import time

## Below path hardcoded. TODO: Change this
prefix_path = 'data1'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())

iterations = 6

test_X = []

# parse_index = [0, 2, 1, 4, 6, 8, 9, 10, 14, 16, 19, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 38]
sparse_index = [i for i in range(40)]
# sparse_index = [i for i in sparse_index if i not in [4, 10, 25]]
dense_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]


# sparse_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]

def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    dense_x = feat[:, dense_index]
    return sparse_x, dense_x


for fileno in range(10000):
    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')

    sparse_x, dense_x = __preprocess_feature(np.array(features))

    ## For each feature, we find average of all values and replace all NaN with that value
    sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_max = np.max(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_medians = np.nanmedian(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_nans = np.count_nonzero(np.isnan(sparse_x), axis=0)
    from scipy import stats
    sparse_mode = stats.mode(sparse_x)
    sparse_mode = sparse_mode[0][0]
    sp_features = np.concatenate([sparse_mode, sparse_max, sparse_medians, sparse_x[0], sparse_x[-1]])

    # sdense_means = np.nanmean(np.where(sparse_x!=0, sparse_x, np.nan), axis=0)
    test_X.append(sp_features)

test_set_results = []

incorrect_x = None

for it in range(iterations):
    print('Starting Iteration ', it)
    X = []
    y = []
    ## ones count kept to balance number of zeros and ones in data to be equal
    ones = len(labels.loc[labels['label'] == 1])

    max_len = 340
    batch_size = 512
    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])
    ## For each sample in the file
    X_sparse = []
    zero_test = []
    zero_test_y = []
    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        ## Checking below if number of zeros matches total number of ones, then stop adding zeros to data
        if label == 0 and ones > 0:
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            continue
        ## features is a (N, 40) matrix
        features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')

        sparse_x, dense_x = __preprocess_feature(features)

        sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        sparse_max = np.max(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        sparse_medians = np.nanmedian(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        sparse_nans = np.count_nonzero(np.isnan(sparse_x), axis=0)
        sparse_mode = stats.mode(sparse_x)
        sparse_mode = sparse_mode[0][0]
        sp_features = np.concatenate([sparse_mode, sparse_max, sparse_medians, sparse_x[0], sparse_x[-1]])

        X_sparse.append(sp_features)
        y.append(label)
    if incorrect_x is not None:
        X_sparse.extend(incorrect_x)
        y.extend(incorrect_y)

    X_sparse = np.array(X_sparse)
    y = np.array(y)

    print('X Shape', X_sparse.shape)
    print('y shape', y.shape)

    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.20)

    x_test_2 = np.concatenate([x_test])
    y_test_2 = np.concatenate([y_test])

    print('X_train Shape', x_train.shape)
    print('X_test shape', x_test.shape)
    print('y_train Shape', y_train.shape)
    print('y_test shape', y_test.shape)

    print('Starting training')

    lgb_train = gbm.Dataset(x_train, y_train)
    lgb_eval = gbm.Dataset(x_test, y_test, reference=lgb_train)


    model = CatBoostClassifier(iterations=1200, learning_rate=0.01, l2_leaf_reg=3.5, depth=8, rsm=0.98,
                               loss_function='CrossEntropy', eval_metric='AUC', use_best_model=True, random_seed=42)

    model.fit(x_train, y_train, cat_features=[], eval_set=(x_test_2, y_test_2))

    # model.fit(x_train, y_train)
    incorrect_x = []
    incorrect_y = []
    y_pred = model.predict_proba(x_test_2)

    print("======>", y_pred)
    # for i, x_sample in enumerate(x_test_2):
    #     if int(round(y_pred[i])) != y_test_2[i]:
    #         incorrect_x.append(x_sample)
    #         incorrect_y.append(y_test_2[i])
    # incorrect_x = np.array(incorrect_x)
    # incorrect_y = np.array(incorrect_y)
    # print('{} out of {} incorrect'.format(incorrect_x.shape, x_test_2.shape))
    ####
    y_pred = y_pred[:,1]
    print(y_pred)
    xg_predictions = [int(round(value)) for value in y_pred]
    print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_test_2, y_pred),
          accuracy_score(y_test_2, xg_predictions), recall_score(y_test_2, xg_predictions),
          precision_score(y_test_2, xg_predictions))

    y_xg_1 = model.predict_proba(test_X)
    y_xg_1 = y_xg_1[:, 1]
    test_set_results.append(y_xg_1)

test_set_results = np.array(test_set_results)
print('Results', test_set_results.shape)

print(test_set_results)
final_y = np.average(test_set_results, axis=0)
print(final_y.shape, final_y)

import pandas as pd

df = pd.DataFrame()
df["Predicted"] = final_y
df.to_csv('lgbm_200-' + str(time.time()) + '.csv', index_label="Id")
