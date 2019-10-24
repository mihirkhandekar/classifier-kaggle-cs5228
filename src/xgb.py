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
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score

## Below path hardcoded. TODO: Change this
prefix_path = '..'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())

iterations = 10
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

test_X = []
for fileno in range(10000):
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    #zeros_array = np.zeros((max_len, 40))

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')
    
    ## For each feature, we find average of all values and replace all NaN with that value
    for feature in range(40):
        average_value = np.average(features[:feature][np.nan_to_num(features[:feature]) != 0])
        features[:feature] = np.nan_to_num(features[:feature], average_value)
    feature_avg = np.average(features, axis=0)
    test_X.append(feature_avg)
deleting_index = [0,1,4,6,8,9,10,14,16,19,21,22,23,25,26,27,28,30,32,34,36,38]


test_X = np.delete(test_X, deleting_index, axis=1)
#est_X = np.delete(test_X, 3, axis=1)
##test_X = np.delete(test_X, 11, axis=1)
#test_X = np.delete(test_X, 33, axis=1)
#test_X = np.delete(test_X, 35, axis=1)

test_set_results = []

for it in range(iterations):
    print('Starting XGBoost Iteration ', it)
    X = []
    y = []
    ## ones count kept to balance number of zeros and ones in data to be equal
    ones = len(labels.loc[labels['label']==1])

    max_len = 340
    batch_size = 512
    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])
    ## For each sample in the file

    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        ## Checking below if number of zeros matches total number of ones, then stop adding zeros to data
        if label == 0 and ones > 0:
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            continue
        # zeros_array used to keep the maximum number of sequences constant to max_len
        # zeros_array = np.zeros((max_len, 40))

        ## features is a (N, 40) matrix
        features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')

        ## We add it to zeros_array to make all samples as (400, 40) matrix
        # zeros_array[0:len(features)] = features
        
        ## For each feature, we find average of all values and replace all NaN with that value
        for feature in range(40):
            average_value = np.average(features[:feature][np.nan_to_num(features[:feature]) != 0])
            features[:feature] = np.nan_to_num(features[:feature], average_value)
        
        feature_avg = np.average(features, axis=0)
        assert feature_avg.shape == (features.shape[1],)
        if len(X) == 0:
            X = np.array([feature_avg])
        else:
            X = np.concatenate((X, [feature_avg]))
        y.append(label)

    X = np.array(X)
    y = np.array(y)


    X = np.delete(X, deleting_index, axis=1)
    #X = np.delete(X, 3, axis=1)
    #X = np.delete(X, 11, axis=1)
    #X = np.delete(X, 33, axis=1)
    #X = np.delete(X, 35, axis=1)

    print('X Shape', X.shape)
    print('y shape', y.shape)

    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

    #weight_ratio = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 
#1]))
    #w_array = np.array([1.0]*y_train.shape[0])  
    #w_array[y_train==1] = 8
    #w_array[y_train==0] = 1

    #print('Weights', w_array.shape, w_array)
    # class_weight = {0: 1., 1: 10}

    print('X_train Shape', x_train.shape)
    print('X_test shape', x_test.shape)
    print('y_train Shape', y_train.shape)
    print('y_test shape', y_test.shape)
    
    print('Starting XGB training')
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=200, max_depth=8, objective="binary:logistic", silent=False)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    xg_predictions = [int(round(value)) for value in y_pred]
    print('Round validation accuracy, recall, precision', accuracy_score(y_test, xg_predictions), recall_score(y_test, xg_predictions), precision_score(y_test, xg_predictions))

    y_xg_1 = model.predict(test_X)
    test_set_results.append(y_xg_1)

test_set_results = np.array(test_set_results)
print('Results', test_set_results.shape)

##accuracy = accuracy_score(final_score, xg_predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

final_y = np.average(test_set_results, axis=0)
print(final_y.shape, final_y)

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

