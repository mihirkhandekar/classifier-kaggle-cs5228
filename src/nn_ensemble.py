import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Conv2D, Multiply, Activation, MaxPooling1D, Masking
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from keras.callbacks.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from keras.regularizers import l2
from keras import regularizers
import math
## Below path hardcoded. TODO: Change this
prefix_path = '../data'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())

iterations = 1
max_len = 336

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

def get_model():
    data_input = Input(shape=(None, 35))
    #X = Masking(mask_value=-100, input_shape=(None, 35))(data_input)
    X = BatchNormalization()(data_input)
    X = Bidirectional(LSTM(512))(X)
    # sig_conv = Conv1D(64, (1), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
    #rel_conv = Conv1D(64, (1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
    #a = Multiply()([sig_conv, rel_conv])

    #b_sig = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid",
                   #padding="same")(X)
    #b_relu = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="relu",
                    #padding="same")(X)
    #b = Multiply()([b_sig, b_relu])

    #X = Concatenate()([a, b])
    X = BatchNormalization()(X)
    # X = Bidirectional(LSTM(64))(X)
    # X = GlobalMaxPooling1D()(X)
    X = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
    # X = Bidirectional(LSTM(32))(X)
    X = Dropout(0.5)(X)
    X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Activation("sigmoid")(X)
    return Model(input=data_input, output=X)


test_X = []
for fileno in range(10000):
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    zeros_array = np.zeros((max_len, 40)) #- 100
    # print(zeros_array)

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
    ## We add it to zeros_array to make all samples as (400, 40) matrix
    # zeros_array[0:len(features)] = features

    #print('orig shape', features.shape)
    ## For each feature, we find average of all values and replace all NaN with that value
    for feature in range(40):
        feature_values = features[:, feature]
        nonnan_feature_values = feature_values[~np.isnan(feature_values)]
        avg = mode(nonnan_feature_values)
        temp = features[:, feature]
        temp[np.isnan(temp)] = avg[0] or np.nan
        features[:, feature] = temp
    
    zeros_array[0:len(features)] = features
    # print(zeros_array)
    test_X.append(zeros_array)

test_X = np.delete(test_X, [2, 3, 11, 33, 35], axis=2)
# test_X = test_X.reshape((test_X.shape[0], -1))
test_X = np.nan_to_num(test_X)
print(test_X.shape)

test_set_results = []

for it in range(iterations):
    print('Starting NN Iteration ', it)
    X = []
    y = []
    ## ones count kept to balance number of zeros and ones in data to be equal
    ones = len(labels.loc[labels['label']==1])

    batch_size = 512
    shuffled_labels = shuffle(labels)
    ## For each sample in the file
    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        ## Checking below if number of zeros matches total number of ones, then stop adding zeros to data
        if label == 0 and ones > 0:
            ones = ones - 0.7
        if ones <= 0 and label == 0:
            continue
        ## zeros_array used to keep the maximum number of sequences constant to max_len
        zeros_array = np.zeros((max_len, 40))# - 100

        ## features is a (N, 40) matrix
        features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')

        ## For each feature, we find average of all values and replace all NaN with that value
        for feature in range(40):
            feature_values = features[:, feature]
            nonnan_feature_values = feature_values[~np.isnan(feature_values)]
            avg = mode(nonnan_feature_values)
            temp = features[:, feature]
            temp[np.isnan(temp)] = avg[0] or np.nan
            features[:, feature] = temp

        zeros_array[0:len(features)] = features

        X.append(zeros_array)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = np.delete(X, [2, 3, 11, 33, 35], axis=2)
    X = np.nan_to_num(X)


    print('X Shape', X.shape)
    print('y shape', y.shape)

    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

    print('X_train Shape', x_train.shape)
    print('X_test shape', x_test.shape)
    print('y_train Shape', y_train.shape)
    print('y_test shape', y_test.shape)
    
    print('Starting training')
    model = get_model()
    model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss=[focal_loss],
              metrics=['accuracy', f1_m, precision_m, recall_m])
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    model_checkpoint = ModelCheckpoint("cp1", monitor='loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='accuracy', patience=12, mode='auto')
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    no_epochs = 88
    print('len', len(x_train))
    def get_data():
        yield x_train, y_train
    generator = get_data()
    '''model.fit_generator(
        generator,
        steps_per_epoch=math.ceil(len(x_train) / batch_size),
        epochs=no_epochs,
        shuffle=True,
        class_weight=class_weights,
        verbose=1,
        # initial_epoch=86,
        validation_data=(x_test, y_test),
        callbacks=([model_checkpoint, terminate_on_nan, reduce_lr, early_stopping]))'''
    print('NUmber of NaNs', np.count_nonzero(np.isnan(x_train)), np.count_nonzero(np.isnan(y_train)))
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=no_epochs, callbacks=([model_checkpoint, terminate_on_nan, reduce_lr, early_stopping]))

    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    print("EVALUATION loss:", loss, "accuracy:", accuracy, "f1_score:", f1_score, "precision:", precision, "recall:",
          recall)

    y_xg_1 = model.predict(test_X)
    print(x_xg_1)
    test_set_results.append(y_xg_1)

test_set_results = np.array(test_set_results)
print('Results', test_set_results.shape)

##accuracy = accuracy_score(final_score, xg_predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

final_y = np.average(test_set_results, axis=0)
print(final_y.shape, final_y)
#final_y = [round(value) for value in final_y]

indices = np.array([i for i in range(10000)])
#print(indices.T.shape, test_Y.shape)
np.savetxt("output.csv", final_y, delimiter=",")
import pandas as pd
df = pd.DataFrame()
df["Predicted"] = final_y.T
df.to_csv('output-1.csv')
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

