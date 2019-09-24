import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

prefix_path = '/home/m/mihir/cs5228/data'
labels = pd.read_csv(prefix_path + '/train_kaggle.csv')
print('Labels', labels.describe())
X = []
y = []
ones = len(labels.loc[labels['label']==1])

max_len = 400
batch_size = 500
for index, train_label in labels.iterrows():
	label = train_label['label']
    if label == 0:
        ones -= 1
    if ones == 0:
        break
    zeros_array = np.zeros((max_len, 40))
    features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')
	zeros_array[0:len(features)] = features
	X.append(zeros_array)
	y.append(label)
	
X = np.nan_to_num(np.array(X))
y = np.array(y)
print('X Shape', X.shape)
print('y shape', y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

print('X_train Shape', x_train.shape)
print('X_test shape', x_test.shape)
print('y_train Shape', y_train.shape)
print('y_test shape', y_test.shape)

#print('X_train', x_train)
#print('y_train', y_train, np.count_nonzero(y_train==1))
#print('X_test', x_test)
#print('y_test', y_test, np.count_nonzero(y_test==1))

# model = Sequential()
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

data_input = Input(shape=(None, 40))
normalize_input = BatchNormalization()(data_input)

#sig_conv = Conv1D(128, (2), activation='sigmoid', padding='same')(normalize_input)
#rel_conv = Conv1D(128, (2), activation='relu', padding='same')(normalize_input)
#mul_conv = Multiply()([sig_conv, rel_conv])

#max_pooling = MaxPooling1D(pool_size=10)(mul_conv)
max_pooling = MaxPooling1D(pool_size=8)(normalize_input)

lstm = Bidirectional(LSTM(64))(max_pooling)


dense_1 = Dense(64, kernel_regularizer=l2(0.002), activation='relu')(lstm)
dense_1 = Dropout(0.5)(dense_1)
dense_2 = Dense(1)(dense_1)
out = Activation('sigmoid')(dense_2)
model = Model(input=data_input, output=out)

model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

print('Model', model.summary())

from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss

#training_generator = BalancedBatchGenerator(x_train, y_train, sampler=NearMiss(), batch_size=50, random_state=42)

#callback_history = model.fit_generator(generator=training_generator, epochs=10, verbose=1)

model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs = 25,
		  validation_data=[x_test, y_test])
