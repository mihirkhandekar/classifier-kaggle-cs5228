import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Multiply, Activation

prefix_path = '/kaggle/input/1910-cs5228-knowledge-discovery-and-data-mining'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
print('Labels', labels.describe())
X = []
y = []
limit = 100
max_len = 1000
batch_size = 50
for index, train_label in labels.iterrows():
	zeros_array = np.zeros((max_len, 40))
	features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')
	zeros_array[0:len(features)] = features
	X.append(zeros_array)
	y.append(train_label['label'])
	limit = limit - 1
	if limit == 0:
		break
	
X = np.nan_to_num(np.array(X))
y = np.array(y)
print('X Shape', X.shape)
print('y shape', y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print('X_train Shape', x_train.shape)
print('X_test shape', x_test.shape)
print('y_train Shape', y_train.shape)
print('y_test shape', y_test.shape)

model = Sequential()

data_input = Input(shape=(None, 40))
normalize_input = BatchNormalization()(data_input)

sig_conv = Conv1D(128, (2), activation='sigmoid', padding='same')(normalize_input)
rel_conv = Conv1D(128, (2), activation='relu', padding='same')(normalize_input)
mul_conv = Multiply()([sig_conv, rel_conv])

lstm = Bidirectional(LSTM(64))(mul_conv)
dense_1 = Dense(64, activation='relu')(lstm)
dense_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(1)(dense_1)
out = Activation('softmax')(dense_2)
model = Model(input=data_input, output=out)

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Model', model.summary())

model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs = 100,
		  validation_data=[x_test, y_test])
