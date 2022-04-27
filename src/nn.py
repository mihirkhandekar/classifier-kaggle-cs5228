import numpy as np
import pandas as pd
import math
from keras import regularizers
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Concatenate, Bidirectional, Dense, Activation, Dropout, \
    Input, LSTM, Conv1D, BatchNormalization, GlobalMaxPooling1D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks.callbacks import TerminateOnNaN, ReduceLROnPlateau, EarlyStopping

prefix = 'data'
labels = pd.read_csv(f'{prefix}/train_kaggle.csv')

sparse_index = list(range(40))
def __preprocess_feature(feat):
    return feat[:, sparse_index]

def generate_data(x_data, y_data, b_size):
    x_data = np.array(x_data)
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


predictions = []
batch_size = 128
no_epochs = 100
ml = 80

def mymodel():
    X_input = Input(shape=(ml, 40))
    X = BatchNormalization()(X_input)
    a = Conv1D(filters=64, kernel_size=(1), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid",
               padding="same")(X)
    b = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid",
               padding="same")(X)
    X = Concatenate()([a, b])  # Normalization 2
    X = BatchNormalization()(X)
    X = Bidirectional(LSTM(32, return_sequences=True))(X)
    X = GlobalMaxPooling1D()(X)
    X = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Dropout(0.5)(X)
    X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
    final_output = Activation("sigmoid")(X)
    model = Model(input=X_input, output=final_output)
    model.summary()
    return model


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


X_test = []
for fileno in range(10000):
    features = np.load(f'{prefix}/test/test/{str(fileno)}.npy')
    zero_mat = np.zeros((ml, 40))
    features = np.load(f'{prefix}/test/test/{str(fileno)}.npy')
    if features.shape[0] > ml:
        s = np.random.choice(range(21, features.shape[0] - 20), 40, replace=False)
        s.sort()
        mid_f = features[s, :]
        last_f = features[-20:, :]
        zero_mat = np.concatenate([features[:20, :], mid_f, last_f], axis=0)
    else:
        zero_mat[:features.shape[0], :] = features[:min(ml, features.shape[0]), :]
    X_test.append(zero_mat)

X_test = np.nan_to_num(np.array(X_test), 0)
for i in range(10):
    print("Iteration no.:", i + 1)
    ones = len(labels.loc[labels['label'] == 1])
    shuffled_labels = shuffle(labels)
    X_data = []
    y = []
    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        zero_mat = np.zeros((ml, 40))
        if label == 0 and ones > 0:
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            continue
        features = np.load(f'{prefix}/train/train/' + str(train_label['Id']) + '.npy')

        if features.shape[0] > ml:
            s = np.random.choice(range(21, features.shape[0] - 20), 40, replace=False)
            s.sort()
            mid_f = features[s, :]
            last_f = features[-20:, :]
            zero_mat = np.concatenate([features[:20, :], mid_f, last_f], axis=0)

        else:
            zero_mat[:features.shape[0], :] = features[:min(ml, features.shape[0]), :]

        X_data.append(zero_mat)
        y.append(label)

    X_data = np.nan_to_num(np.array(X_data), 0)
    y = np.array(y)
    print(("===X_data==>", X_data.shape))
    print("y shape", y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X_data, y, shuffle=True, test_size=0.20)

    model = mymodel()
    model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss="binary_crossentropy",
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    generator2 = generate_data(x_train, y_train, batch_size)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    model.fit_generator(
        generator2,
        steps_per_epoch=math.ceil(len(x_train) / batch_size),
        epochs=no_epochs,
        shuffle=True,
        class_weight=class_weights,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=([terminate_on_nan, reduce_lr, early_stopping]))

    y_pred = model.predict(x_test)
    y_pred = np.array(y_pred)
    print(y_pred)
    xg_predictions = [int(np.round(value)) for value in y_pred]
    print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_test, y_pred),
          accuracy_score(y_test, xg_predictions), recall_score(y_test, xg_predictions),
          precision_score(y_test, xg_predictions))

    pred = model.predict(X_test)
    predictions.append(pred)

predictions = np.array(predictions)
predictions = np.mean(predictions, axis=0)
pred = pd.DataFrame(
    data=predictions,
    index=list(range(predictions.shape[0])),
    columns=["Predicted"],
)

pred.index.name = 'Id'
pred.to_csv('outputs/dl-output.csv', index=True)
