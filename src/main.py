import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

## Below path hardcoded. TODO: Change this
prefix_path = '/home/m/mihir/cs5228/data'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())


X = []
y = []


## ones count kept to balance number of zeros and ones in data to be equal
ones = len(labels.loc[labels['label']==1])

max_len = 350
batch_size = 512

## For each sample in the file
for index, train_label in labels.iterrows():
    label = train_label['label']
    ## Checking below if number of zeros matches total number of ones, then stop adding zeros to data
    if label == 0 and ones > 0:
        ones = ones - 1
    if ones == 0 and label == 0:
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

X = np.nan_to_num(np.array(X))
y = np.array(y)

print('X Shape', X.shape)
print('y shape', y.shape)

## Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

print('X_train Shape', x_train.shape)
print('X_test shape', x_test.shape)
print('y_train Shape', y_train.shape)
print('y_test shape', y_test.shape)

#print('X_train', x_train)
print('y_train', y_train, np.count_nonzero(y_train==1))
#print('X_test', x_test)
#print('y_test', y_test, np.count_nonzero(y_test==1))


data_input = Input(shape=(None, 40))

# sig_input = Activation('sigmoid')(data_input)
normalize_input = BatchNormalization()(data_input)

sig_conv = Conv1D(40, (1), activation='sigmoid', padding='same')(normalize_input)
rel_conv = Conv1D(40, (1), activation='tanh', padding='same')(normalize_input)
mul_conv = Multiply()([sig_conv, rel_conv])

#max_pooling = MaxPooling1D(pool_size=10)(mul_conv)
#max_pooling = MaxPooling1D(pool_size=4)(data_input)

lstm = Bidirectional(LSTM(64))(mul_conv)

dense_1 = Dense(16, activation='relu')(lstm)
#dense_1 = Dropout(0.8)(dense_1)
dense_2 = Dense(1)(dense_1)
out = Activation('sigmoid')(dense_2)
model = Model(input=data_input, output=out)

model.compile(optimizer=Adam(lr=0.005), loss= 'binary_crossentropy', metrics=['accuracy'])

print('Model', model.summary())


#training_generator = BalancedBatchGenerator(x_train, y_train, sampler=NearMiss(), batch_size=50, random_state=42)

#callback_history = model.fit_generator(generator=training_generator, epochs=10, verbose=1)

model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs = 22,
		  validation_data=[x_test, y_test])

print('Saving model')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
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
        zeros_array[:feature] = np.nan_to_num(zeros_array[:feature], average_value)

    test_X.append(zeros_array)
    
print('Predicting test data')
np.set_printoptions(precision=25)
test_Y = model.predict(np.nan_to_num(np.array(test_X)))
indices = np.array([i for i in range(10000)])
print(indices.T.shape, test_Y.shape)
np.savetxt("output.csv", test_Y, delimiter=",")

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

