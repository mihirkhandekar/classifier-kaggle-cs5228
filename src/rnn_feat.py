import math
import os
import sys
import time

import keras.backend as K
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle

sparse_index = [i for i in range(40)]

prefix_path = '../data'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
print('Labels', labels.describe())
iterations = 6


def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    return sparse_x


def __get_model():
    left = Sequential()
    left.add(BatchNormalization())
    left.add(Activation('relu'))
    right = Sequential()
    right.add(BatchNormalization())
    right.add(Activation('sigmoid'))
    model = Multiply([left, right])
    model.add(Dense(256, activation='relu', input_shape=(160, )))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
    '''
    data_input = Input(shape=(160))

    X = BatchNormalization()(data_input)

    sig_conv = Conv1D(128, (1), activation='sigmoid', padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(X)
    rel_conv = Conv1D(128, (1), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Multiply()([sig_conv, rel_conv])

    X = Activation("relu")(X)

    X = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.0005))(X)

    X = Dropout(0.5)(X)
    X = Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.0005))(X)

    X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Activation("sigmoid")(X)
    model = Model(input=data_input, output=X)
    return model
    '''

def __extract_features(features):
    sparse_x = __preprocess_feature(np.array(features))

    # For each feature, we find average of all values and replace all NaN with that value
    sparse_means = np.nanmean(
        np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_max = np.max(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_medians = np.nanmedian(
        np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    sparse_nans = np.count_nonzero(np.isnan(sparse_x), axis=0)

    sp_features = np.concatenate(
        [sparse_max, sparse_medians, sparse_x[0], sparse_x[-1]])
    return np.nan_to_num(sp_features)


test_X = []
test_Y = []

# Read test file
for fileno in range(10000):
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
    sp_features = __extract_features(features)
    test_X.append(sp_features)


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
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            continue
        features = np.load(prefix_path + '/train/train/' +
                           str(train_label['Id']) + '.npy')

        sp_features = __extract_features(features)
        X.append(sp_features)
        y.append(label)

    '''if incorrect_x is not None:
        X_sparse.extend(incorrect_x)
        y.extend(incorrect_y)
    '''
    X = np.array(X)
    y = np.array(y)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    lstm_model = __get_model()
    lstm_model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss='binary_crossentropy',
                       metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    model_checkpoint = ModelCheckpoint(
        "cp1", monitor='loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=32, mode='auto')

    lstm_model.fit(
        x_train, y_train,
        epochs=no_epochs,
        shuffle=True,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=([model_checkpoint, terminate_on_nan, early_stopping]))

    y_pred = lstm_model.predict(x_val)
    print(y_pred.shape)

    xg_predictions = [int(round(value[0])) for value in y_pred]
    print('Round validation ROC-AUC = {}, accuracy = {}, recall = {}, precision = {}'.format(roc_auc_score(y_val, y_pred),
                                                                                             accuracy_score(y_val, xg_predictions), recall_score(
        y_val, xg_predictions),
        precision_score(y_val, xg_predictions)))

    y_test_dl = lstm_model.predict(test_X)
    test_Y.append(y_test_dl)

test_set_results = np.array(test_set_results)
print('Results', test_set_results.shape)

print(test_Y)
test_Y = np.average(test_Y, axis=0)
print(test_Y.shape, test_Y)

df = pd.DataFrame()
df["Predicted"] = test_Y
df.to_csv('dl-output-' + str(time.time()) + '.csv', index_label="Id")
