import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Conv2D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
## Below path hardcoded. TODO: Change this
prefix_path = '..'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())


X = []
y = []


## ones count kept to balance number of zeros and ones in data to be equal
ones = len(labels.loc[labels['label']==1])

max_len = 340
batch_size = 512

## For each sample in the file
for index, train_label in labels.iterrows():
    label = train_label['label']
    ## Checking below if number of zeros matches total number of ones, then stop adding zeros to data
    if label == 0 and ones > 0:
        ones = ones - 0.7
    if ones <= 0 and label == 0:
        continue
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    zeros_array = np.zeros((max_len, 40))

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')
    
    ## We add it to zeros_array to make all samples as (400, 40) matrix
    zeros_array[0:len(features)] = features


    ## For each feature, we find average of all values and replace all NaN with that value
    for feature in range(40):
        average_value = np.average(zeros_array[:feature][np.nan_to_num(zeros_array[:feature]) != 0])
        zeros_array[:feature] = np.nan_to_num(zeros_array[:feature], average_value)

    X.append(zeros_array)
    y.append(label)

X = np.array(X)
y = np.array(y)

print('X Shape', X.shape)
print('y shape', y.shape)

## Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

print('X_train Shape', x_train.shape)
print('X_test shape', x_test.shape)
print('y_train Shape', y_train.shape)
print('y_test shape', y_test.shape)
print('y_train', y_train, np.count_nonzero(y_train==1))
x_train = np.delete(x_train, 2, axis=2)
x_train = np.delete(x_train, 11, axis=2)
x_train = np.delete(x_train, 33, axis=2)
x_train = np.delete(x_train, 35, axis=2)
x_test = np.delete(x_test, 2, axis=2)
x_test = np.delete(x_test, 11, axis=2)
x_test = np.delete(x_test, 33, axis=2)
x_test = np.delete(x_test, 35, axis=2)

x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))
############# Keras ###############
'''print('Starting Keras training')
data_input = Input(shape=(None, 36))

normalize_input = BatchNormalization()(data_input)

sig_conv = Conv1D(36, (1), activation='sigmoid', padding='same')(normalize_input)
rel_conv = Conv1D(36, (1), activation='tanh', padding='same')(normalize_input)
mul_conv = Multiply()([sig_conv, rel_conv])

lstm = Bidirectional(LSTM(64))(mul_conv)

dense_1 = Dense(16, activation='relu')(lstm)
dense_1 = Dropout(0.1)(dense_1)
dense_2 = Dense(1)(dense_1)
out = Activation('sigmoid')(dense_2)
model_ker = Model(input=data_input, output=out)

model_ker.compile(optimizer=Adam(lr=0.005), loss= 'binary_crossentropy', metrics=['accuracy'])
x_train_zer = np.nan_to_num(x_train)
model_ker.fit(x_train_zer, y_train, epochs=20, validation_data=[x_test, y_test])
x_test_zer = np.nan_to_num(x_test)
ker_output = model_ker.predict(x_test_zer)
ker_output = [round(value) for value in ker_output]
print('Ker acc', accuracy_score(ker_output, y_test))
'''
test_X = []
for fileno in range(10000):
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    zeros_array = np.zeros((max_len, 40))

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
    ## We add it to zeros_array to make all samples as (400, 40) matrix
    zeros_array[0:len(features)] = features


    ## For each feature, we find average of all values and replace all NaN with that value
    for feature in range(40):
        average_value = np.average(zeros_array[:feature][np.nan_to_num(zeros_array[:feature]) != 0])
        #zeros_array[:feature] = np.nan_to_num(zeros_array[:feature], average_value)

    test_X.append(zeros_array)

test_X = np.delete(test_X, 2, axis=2)
test_X = np.delete(test_X, 11, axis=2)
test_X = np.delete(test_X, 33, axis=2)
test_X = np.delete(test_X, 35, axis=2)
test_X_flat = test_X.reshape((test_X.shape[0], -1))
test_X_flat_nan = np.nan_to_num(test_X_flat)


from sklearn.metrics import accuracy_score

############ XGBoost ##############
print('Starting XGB training')
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train_flat, y_train)
y_pred_1 = model.predict(x_test_flat)
xg_predictions = [round(value) for value in y_pred_1]
y_xg_1 = model.predict(test_X_flat_nan)

print('XG acc', accuracy_score(xg_predictions, y_test))
###################################
X = []
y = []

ones = len(labels.loc[labels['label']==1])
for index, train_label in labels[::-1].iterrows():
    label = train_label['label']
    ## Checking below if number of zeros matches total number of ones, then stop adding zeros to data
    if label == 0 and ones > 0:
        ones = ones - 0.7
    if ones <= 0 and label == 0:
        continue
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    zeros_array = np.zeros((max_len, 40))

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')
    
    ## We add it to zeros_array to make all samples as (400, 40) matrix
    zeros_array[0:len(features)] = features


    ## For each feature, we find average of all values and replace all NaN with that value
    for feature in range(40):
        average_value = np.average(zeros_array[:feature][np.nan_to_num(zeros_array[:feature]) != 0])
        zeros_array[:feature] = np.nan_to_num(zeros_array[:feature], average_value)

    X.append(zeros_array)
    y.append(label)

X = np.array(X)
y = np.array(y)

print('X Shape', X.shape)
print('y shape', y.shape)

## Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

print('X_train Shape', x_train.shape)
print('X_test shape', x_test.shape)
print('y_train Shape', y_train.shape)
print('y_test shape', y_test.shape)
print('y_train', y_train, np.count_nonzero(y_train==1))
x_train = np.delete(x_train, 2, axis=2)
x_train = np.delete(x_train, 11, axis=2)
x_train = np.delete(x_train, 33, axis=2)
x_train = np.delete(x_train, 35, axis=2)
x_test = np.delete(x_test, 2, axis=2)
x_test = np.delete(x_test, 11, axis=2)
x_test = np.delete(x_test, 33, axis=2)
x_test = np.delete(x_test, 35, axis=2)

x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))
#final_score = [round(value) for value in (np.array(rforest_predictions) + np.array(xg_predictions) + #np.array(ada_predictions) + np.array(svm_predictions) + np.array(per_predictions))/5]

print('Starting XGB training')
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train_flat, y_train)
y_pred_2 = model.predict(x_test_flat)
xg_predictions = [round(value) for value in y_pred_2]
y_xg_2 = model.predict(test_X_flat_nan)

print('XG acc', accuracy_score(xg_predictions, y_test))


##accuracy = accuracy_score(final_score, xg_predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

final_y = (y_xg_1 + y_xg_2)/2
final_y = [round(value+0.001) for value in final_y]

#print('Predicting test data')
#np.set_printoptions(precision=25)
#test_Y = model.predict(np.nan_to_num(np.array(test_X)))
indices = np.array([i for i in range(10000)])
#print(indices.T.shape, test_Y.shape)
np.savetxt("output.csv", final_y, delimiter=",")
import pandas as pd
df = pd.DataFrame()
df["Predicted"] = final_y
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

