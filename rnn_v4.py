import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from keras.callbacks.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping
import time
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Multiply, LSTM, Conv1D
from keras.layers import Bidirectional, BatchNormalization, Concatenate, GlobalMaxPooling1D
from keras.optimizers import Adam

## Below path hardcoded. TODO: Change this
prefix_path = 'data1'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
iterations = 6
test_X = []

sparse_index = [i for i in range(40)]
dense_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]

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

def malware_detection_model_3():
    data_input = Input(shape=(None, 200))
    X = BatchNormalization()(data_input)

    sig_conv = Conv1D(64, (1), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
    # rel_conv = Conv1D(64, (1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
    # a = Multiply()([sig_conv, rel_conv])

    # b_sig = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid", padding="same")(X)
    # b_relu = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="relu", padding="same")(X)
    # b = Multiply()([b_sig, b_relu])

    # X = Concatenate()([a, b])
    # X = BatchNormalization()(X)
    # X = Bidirectional(LSTM(64))(sig_conv)
    X = LSTM(64)(sig_conv)

    # X = Bidirectional(LSTM(64))(X)
    # X = GlobalMaxPooling1D()(X)
    X = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Dropout(0.5)(X)
    X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Activation("sigmoid")(X)


    model = Model(inputs=data_input, outputs=X)
    opt = Adam(learning_rate=0.001, decay=1e-8)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

model = malware_detection_model_3()
test_X = np.array(test_X)
test_X = np.nan_to_num(test_X.reshape(test_X.shape[0], 1, test_X.shape[1]))
print('test_Xy shape', test_X.shape)

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


    X_sparse = np.nan_to_num(X_sparse.reshape(X_sparse.shape[0], 1, X_sparse.shape[1]))


    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.20)

    x_test_2 = np.concatenate([x_test])
    y_test_2 = np.concatenate([y_test])

    print('X_train Shape', x_train.shape)
    print('X_test shape', x_test_2.shape)
    print('y_train Shape', y_train.shape)
    print('y_test shape', y_test_2.shape)
    print('main test_X shape', y_test_2.shape)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    early_stopping = EarlyStopping(monitor='recall_m', patience=5, mode='auto')

    print('Starting training')
    model.fit(x=x_train, y=y_train, epochs=30, batch_size=32,
              shuffle=True,
              callbacks=([terminate_on_nan, reduce_lr, early_stopping]))
    # lgb_train = gbm.Dataset(x_train, y_train)
    # lgb_eval = gbm.Dataset(x_test, y_test, reference=lgb_train)
    #
    # model = XGBRegressor(objective="binary:logistic", eval_metric="auc", subsample=0.5,
    #                      learning_rate=0.005, max_depth=8,
    #                      min_child_weight=5, n_estimators=3000,
    #                      reg_alpha=0.1, reg_lambda=0.3, gamma=0.1,
    #                      silent=1, random_state=8, nthread=-1)
    # model.fit(x_train, y_train)


     ####herrere
    incorrect_x = []
    incorrect_y = []
    y_pred = model.predict(x_test_2)
    # for i, x_sample in enumerate(x_test_2):
    #     if int(round(y_pred[i])) != y_test_2[i]:
    #         incorrect_x.append(x_sample)
    #         incorrect_y.append(y_test_2[i])
    # incorrect_x = np.array(incorrect_x)
    # incorrect_y = np.array(incorrect_y)
    # print('{} out of {} incorrect'.format(incorrect_x.shape, x_test_2.shape))
    ####
    # y_pred = np.array(y_pred).tolist()
    # xg_predictions = [int(round(value)) for value in y_pred]
    # print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_test_2, y_pred),
    #       accuracy_score(y_test_2, xg_predictions), recall_score(y_test_2, xg_predictions),
    #       precision_score(y_test_2, xg_predictions))

    y_xg_1 = model.predict(test_X)
    test_set_results.append(y_xg_1)

test_set_results = np.array(test_set_results)
print('Results', test_set_results.shape)

print(test_set_results)
final_y = np.average(test_set_results, axis=0)
print(final_y.shape, final_y)

import pandas as pd

final_pred = pd.DataFrame(data=final_y, index=[i for i in range(final_y.shape[0])], columns=["Predicted"])
final_pred.index.name = 'Id'
final_pred.to_csv('rnn_complex.csv', index=True)
