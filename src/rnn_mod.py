import math
import os
import sys

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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, shuffle


def classification_evaluation(y_ture, y_pred):
    acc = accuracy_score(y_ture, (y_pred > 0.5).astype('int'))
    auc = roc_auc_score(y_ture, y_pred)
    fpr, tpr, thresholds = roc_curve(y_ture, y_pred)

    print('Accuracy:', acc)
    print('ROC AUC Score:', auc)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('FPR')
    plt.ylabel('Recall rate')
    plt.show()


plt.style.use('seaborn')

max_len = 336
batch_size = 128
train_samples = 30336
test_samples = 10000
no_epochs = 15
max_time = 60
deleted_cols = [2]

print('Reading labels CSV')
labels = pd.read_csv("data/train_kaggle.csv")
ones = len(labels.loc[labels['label'] == 1])
zeros = len(labels.loc[labels['label'] == 0])
X_t = []
y_t = []
print('Ones, zeros', ones, zeros)

bar = progressbar.ProgressBar(maxval=len(labels),
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


bar.start()
print("Reading and preprocessing data...")
for index, train_label in labels.iterrows():
    label = train_label['label']
    data = np.load("data/train/train/" + str(train_label['Id']) + '.npy')

    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])

    data = np.delete(data, deleted_cols, axis=1)

    zero_mat = np.zeros((max_time, data.shape[1]))
    zero_mat[:min(max_time, data.shape[0]),
             :] = data[:min(max_time, data.shape[0]), :]
    X_t.append(zero_mat)
    y_t.append(label)
    bar.update(index+1)

bar.finish()

X_t = np.array(X_t)
y_t = np.array(y_t)


X_t, y_t = shuffle(X_t, y_t)

X = np.nan_to_num(X_t)
print(y_t)
y = np.array([y_t.T, (1-y_t).T]).T
print(y.shape, y)

print("After preprocessing", X.shape, y.shape)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, y, shuffle=True, test_size=0.15)

print("Split set shapes", X_train.shape,
      Y_train.shape, X_val.shape, Y_val.shape)

print('Creating batches')
bar = progressbar.ProgressBar(maxval=no_epochs,
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])


bar.start()

def create_batches(x_data, y_data, b_size):
    batches = []
    for it in range(no_epochs):
        shuffle(x_data, y_data)
        max_ones = np.sum(y_data[:, 0])
        one_count = 0
        x_batch = []
        y_batch = []
        for index, data in enumerate(x_data):
            if y_data[index, 0] == 1:
                one_count += 1
            if one_count > max_ones and y_data[index, 0] == 1:
                continue
            x_batch.append(data)
            y_batch.append(y_data[index])
            if index >= b_size:
                break
        batches.append((np.array(x_batch), np.array(y_batch)))
        bar.update(it+1)
bar.finish()

batches = create_batches(X_train, Y_train, batch_size)

def generate_data(x_data, y_data, b_size):
    i = 0
    while True:
        yield batches[i % b_size][0], batches[i % b_size][1]
        i = i + 1


def get_model():
    data_input = Input(shape=(None, 39))

    X = BatchNormalization()(data_input)

    sig_conv = Conv1D(128, (1), activation='sigmoid', padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(X)
    rel_conv = Conv1D(128, (1), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.0005))(X)
    a = Multiply()([sig_conv, rel_conv])
    # print(X)

    b_sig = Conv1D(filters=128, kernel_size=(5), strides=1, kernel_regularizer=regularizers.l2(
        0.0005), activation="sigmoid", padding="same")(X)
    b_relu = Conv1D(filters=128, kernel_size=(5), strides=1, kernel_regularizer=regularizers.l2(
        0.0005), activation="relu", padding="same")(X)
    b = Multiply()([b_sig, b_relu])

    X = Concatenate()([a, b])
    #X = BatchNormalization()(X)
    X1 = Bidirectional(LSTM(256))(X)
    X = Activation("relu")(X1)
    #X2 = Bidirectional(LSTM(128))(X)
    #X2 = Activation("relu")(X2)
    #X = Concatenate()([X1, X2])

    X = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.0005))(X)
    # X = GlobalMaxPooling1D()(X)
    X = Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.0005))(X)
    # X = Bidirectional(LSTM(32))(X)
    X = Dropout(0.5)(X)
    X = Dense(2, kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Activation("softmax")(X)
    model = Model(input=data_input, output=X)
    return model


model = get_model()
print(model.summary())


def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 0.5
    y_pred = tf.maximum(y_pred, 1e-15)
    log_y_pred = tf.log(y_pred)
    focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
    focal_loss = tf.multiply(y_true, tf.multiply(focal_scale, log_y_pred))
    return -tf.reduce_sum(focal_loss, axis=-1)


'''
    gamma = 2.0
    alpha = 0.5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
'''

# model = load_model("cp1")


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss=[focal_loss],
              metrics=['accuracy', f1_m, precision_m, recall_m])

generator2 = generate_data(X_train, Y_train, batch_size)

reduce_lr = ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
terminate_on_nan = TerminateOnNaN()
model_checkpoint = ModelCheckpoint(
    "cp1", monitor='loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=6, mode='auto')

#class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

model.fit_generator(
    generator2,
    steps_per_epoch=math.ceil(len(X_train) / batch_size),
    epochs=no_epochs,
    shuffle=True,
    # class_weight=class_weights,
    verbose=1,
    # initial_epoch=86,
    validation_data=(X_val, Y_val),
    callbacks=([model_checkpoint, terminate_on_nan, reduce_lr, early_stopping]))

loss, accuracy, f1_score, precision, recall = model.evaluate(
    X_val, Y_val, verbose=0)
print("EVALUATION loss:", loss, "accuracy:", accuracy, "f1_score:", f1_score, "precision:", precision, "recall:",
      recall)
'''
    STEP 4 : Prepare test samples
'''

print("Starting test - ")
X_test = []
for i in range(0, test_samples):
    data = np.load("data/test/test/" + str(i) + ".npy")
    col_mean = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_mean, inds[1])

    data = np.delete(data, deleted_cols, axis=1)

    zero_mat = np.zeros((max_time, data.shape[1]))
    zero_mat[:min(max_time, data.shape[0]),
             :] = data[:min(max_time, data.shape[0]), :]
    X_test.append(zero_mat)

X_test = np.nan_to_num(np.array(X_test))
print(X_test.shape)

print("Predicting test data")
pred = model.predict(X_test)
print(pred.shape, pred)
p = pd.DataFrame()
p['Predicted'] = pred.T[0]
p.to_csv('rnn_v11.csv', index=True)

pred2 = pd.DataFrame()
pred2['1'] = pred.T[0]
pred2['2'] = pred.T[1]
pred2.to_csv('rnn_v12_pred.csv', index=True)
