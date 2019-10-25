import keras.backend as K
import math
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
                          Concatenate, Conv1D, Conv2D, Dense, Dropout,
                          Embedding, GlobalMaxPooling1D, Input, MaxPooling1D,
                          Multiply)
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight, shuffle
from xgboost import XGBRegressor

## Below path hardcoded. TODO: Change this
prefix_path = '..'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())


test_data_x_1 = []
test_data_x_2 = []
max_len = 50
num_epochs = 20

sparse_index = [0, 1, 4, 6, 8, 9, 10, 14, 16, 19, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 38]
dense_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]

def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    dense_x = feat[:, dense_index]
    return sparse_x, dense_x

def __get_training_x(sparse_x, dense_x, zeros_array):
    ### Preprocess sparse_array
    sparse_means = np.nanmean(np.where(sparse_x!=0, sparse_x, np.nan), axis=0)
    sparse_means = np.nan_to_num(sparse_means)
    
    ### Preprocess dense_array : replace NaN with mean ##
    dense_array = dense_x[:min(max_len, dense_x.shape[0]), :]
    col_mean = np.nanmean(dense_array, axis=0)
    inds = np.where(np.isnan(dense_array))
    dense_array[inds] = np.take(col_mean, inds[1])
    zeros_array[:dense_x.shape[0], :] = dense_array
    zeros_array = np.nan_to_num(zeros_array)

    return sparse_means, zeros_array

def __get_model():
    input_sparse = Input(shape=(len(sparse_index)))
    sparse_relu = Activation('relu')(input_sparse)
    sparse_sigmoid = Activation('sigmoid')(input_sparse)
    combine_sparse = Multiply()([sparse_relu, sparse_sigmoid])

    input_dense = Input(shape=(None, len(dense_index)))
    dense_conv_sig = Conv1D(128, (1), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.0005))(input_dense)
    dense_conv_relu = Conv1D(128, (1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(input_dense)
    combine_dense = Multiply()([dense_conv_relu, dense_conv_sig])

    combined = Concatenate()([combine_sparse, combine_dense])

    dense_1 = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.0005))(combined)

    dense_2 = Dense(128, activation='relu',
              kernel_regularizer=regularizers.l2(0.0005))(dense_1)

    output = Dense(1, activation='sigmoid',
              kernel_regularizer=regularizers.l2(0.0005))(dense_2)

    model = Model(input=[input_sparse, input_dense], output=output)

    return model

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def generate_data(x_data_1, x_data_2, y_data, b_size):
    samples_per_epoch = x_data_1.shape[0]
    number_of_batches = samples_per_epoch / b_size
    counter = 0
    while 1:
        x_batch_1 = np.array(x_data_1[batch_size * counter:batch_size * (counter + 1)])
        x_batch_2 = np.array(x_data_2[batch_size * counter:batch_size * (counter + 1)])
        y_batch = np.array(y_data[batch_size * counter:batch_size * (counter + 1)])
        counter += 1
        yield [x_batch_1, y_batch_1], y_batch

        if counter >= number_of_batches:
            counter = 0


generator = generate_data(x_train_1, x_train_2, y_train, batch_size)

for fileno in range(10000):
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    zeros_array = np.zeros((max_len, len(dense_index)))

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
    
    sparse_x, dense_x = __preprocess_feature(np.array(features))

    sparse_means, zeros_array = __get_training_x(sparse_x, dense_x, zeros_array)

    test_data_x_1.append(sparse_means)
    test_data_x_2.append(zeros_array)


X_1 = []
X_2 = []
y = []
## ones count kept to balance number of zeros and ones in data to be equal
ones = len(labels.loc[labels['label']==1])

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

    sparse_means, zeros_array = __get_training_x(sparse_x, dense_x, zeros_array)

    X_1.append(sparse_means)
    X_2.append(zeros_array)

    y.append(label)

X_1 = np.array(X_1)
X_2 = np.array(X_2)
y = np.array(y)


print('X_1 Shape', X_1.shape)
print('X_2 Shape', X_2.shape)
print('y shape', y.shape)

## Split into train and test datasets
x_train_1, x_test_1, x_train_2, x_test_2, y_train, y_test = train_test_split(X_1, X_2, y, test_size=0.20)

print('X1_train Shape', x_train_1.shape)
print('X1_test shape', x_test_1.shape)
print('X2_train Shape', x_train_2.shape)
print('X2_test shape', x_test_1.shape)
print('y_train Shape', y_train.shape)
print('y_test shape', y_test.shape)

exit(0)


###### DL Training
model = __get_model()
print(model.summary())

model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss=[focal_loss],
              metrics=['accuracy', roc_auc_score, precision_score, recall_score])

early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=6, mode='auto')

model.fit_generator(
    generator,
    steps_per_epoch=math.ceil(len(x_train_1) / batch_size),
    epochs=num_epochs,
    shuffle=True,
    #class_weight=class_weights,
    verbose=1,
    # initial_epoch=86,
    validation_data=([x_test_1, x_test_2], y_test),
    callbacks=([early_stopping]))



y_pred = model.predict([x_test_1, x_test_2])

print(y_pred)

df = pd.DataFrame()
df["Predicted"] = y_pred
df.to_csv('output-1.csv', index_label="Id")
'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

'''
