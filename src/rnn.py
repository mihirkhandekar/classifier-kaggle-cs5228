import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.models import load_model

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Input, \
    BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

from keras.callbacks.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping


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

max_len = 340
# 336
batch_size = 64
# 128
train_samples = 30336
# 30336
test_samples = 10000
# 10000
no_epochs = 88

labels = pd.read_csv("data/train_kaggle.csv")
ones = len(labels.loc[labels['label'] == 1])
zeros = ones
X_t = []
y_t = []
print(ones, zeros)

for index, train_label in labels.iterrows():
    label = train_label['label']
    zero_mat = np.zeros((50, 40))
    data = np.load("data/train/train/" + str(train_label['Id']) + '.npy')
    zero_mat[:data.shape[0], :] = data[:min(50, data.shape[0]), :]
    X_t.append(zero_mat)
    y_t.append(label)

x_sampled = []
y_sampled = []
X_t = np.array(X_t)
y_t = np.array(y_t)

from sklearn.utils import shuffle

X_t, y_t = shuffle(X_t, y_t, random_state=0)
print("Stage 1", X_t.shape, y_t.shape)
# shuffle data
for index in range(y_t.shape[0]):
    label = y_t[index]
    if label == 0 and zeros > 0:
        zeros = zeros - 1
    if label == 1 and ones > 0:
        ones = ones - 1
    if (zeros == 0 and label == 0) or (ones == 0 and label == 1):
        continue
    df1 = pd.DataFrame(data=X_t[index])
    # df1.fillna(0, inplace=True)
    for feature in range(40):
        # average_value = np.nanmean(X_t[index][:, feature])
        # print(average_value)
        # X_t[index][:, feature] = np.nan_to_num(X_t[index][:, feature], nan=average_value)
        # X_t[index][:, feature] = np.nan_to_num(X_t[index][:, feature], nan=0)
        # df1 = pd.DataFrame(data=X_t[index])
        median = df1[feature].mode()
        df1[feature].fillna(median, inplace=True)
    # df1.fillna(0, inplace=True)
    m = np.array(df1)
    m = np.delete(m, [2, 34, 16, 10], axis=1)
    # df1 = pd.DataFrame(data=m)
    # df1 = df1.fillna(method='bfill')

    x_sampled.append(m)
    y_sampled.append(y_t[index])

X = np.nan_to_num(np.array(x_sampled))
y = np.array(y_sampled)

print("Stage 2", X.shape, y.shape)
# model = load_model("cp1")
# print(y)
# df = pd.read_csv("data/train_kaggle.csv")
# Y_t = df[:train_samples]
# y = Y_t.values

X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.15)

# print("Trainig set", X_train, X_val)
print("Trainig set shapes", X_train.shape, Y_train, X_val.shape, Y_val.shape)


def generate_data(x_data, y_data, b_size):
    samples_per_epoch = x_data.shape[0]
    number_of_batches = samples_per_epoch / b_size
    counter = 0
    while 1:
        x_batch = np.array(x_data[batch_size * counter:batch_size * (counter + 1)])
        y_batch = np.array(y_data[batch_size * counter:batch_size * (counter + 1)])
        counter += 1
        yield x_batch, y_batch

        if counter >= number_of_batches:
            counter = 0


data_input = Input(shape=(None, 36))

X = BatchNormalization()(data_input)

sig_conv = Conv1D(64, (1), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
rel_conv = Conv1D(64, (1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
a = Multiply()([sig_conv, rel_conv])

b_sig = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid",
               padding="same")(X)
b_relu = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="relu",
                padding="same")(X)
b = Multiply()([b_sig, b_relu])

X = Concatenate()([a, b])
X = BatchNormalization()(X)
X = Bidirectional(LSTM(64))(X)
# X = GlobalMaxPooling1D()(X)
X = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
# X = Bidirectional(LSTM(32))(X)
X = Dropout(0.5)(X)
X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
X = Activation("sigmoid")(X)
model = Model(input=data_input, output=X)


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


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

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
terminate_on_nan = TerminateOnNaN()
model_checkpoint = ModelCheckpoint("cp1", monitor='loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=12, mode='auto')

class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

model.fit_generator(
    generator2,
    steps_per_epoch=math.ceil(len(X_train) / batch_size),
    epochs=no_epochs,
    shuffle=True,
    class_weight=class_weights,
    verbose=1,
    # initial_epoch=86,
    validation_data=(X_val, Y_val),
    callbacks=([model_checkpoint, terminate_on_nan, reduce_lr, early_stopping]))

loss, accuracy, f1_score, precision, recall = model.evaluate(X_val, Y_val, verbose=0)
print("EVALUATION loss:", loss, "accuracy:", accuracy, "f1_score:", f1_score, "precision:", precision, "recall:",
      recall)
'''
    STEP 4 : Prepare test samples
'''
X_test = []
for i in range(0, test_samples):
    data = np.load("data/test/test/" + str(i) + ".npy")
    zero_mat = np.zeros((50, 40))
    zero_mat[:data.shape[0], :] = data[:min(50, data.shape[0]), :]
    df1 = pd.DataFrame(data=zero_mat)
    for feature in range(40):
        #    average_value = np.nanmean(zero_mat[:, feature])
        # zero_mat[:, feature]= np.nan_to_num(zero_mat[:, feature], nan=average_value)
        #    zero_mat[:, feature] = np.nan_to_num(zero_mat[:, feature], nan=0)
        mod = df1[feature].mode()
        df1[feature].fillna(mod, inplace=True)

    zero_mat = np.array(df1)
    zero_mat = np.delete(zero_mat, [2, 34, 16, 10], axis=1)
    # df1 = pd.DataFrame(data=zero_mat)
    # zero_mat = df1.fillna(method='bfill')
    # 11, 33, 35
    # 1,3, 4, 15, 17,  22, 24, 36
    # 1, 3, 4, 6, 8, 15, 17, 18, 20, 22, 23, 24, 29, 31, 36
    # df1 = pd.DataFrame(data=X_t[index])
    # median = df1[feature].mode()
    # df1[feature].fillna(median, inplace=True)

    X_test.append(np.array(zero_mat))

X_test = np.nan_to_num(np.array(X_test))
print(X_test.shape)
# model = load_model("cp1")
# model = load_model("cp1")

pred = model.predict(X_test)
print(pred.shape, pred)
pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('rnn_v11.csv', index=True)
