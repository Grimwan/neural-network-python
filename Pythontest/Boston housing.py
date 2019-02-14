from keras.datasets import boston_housing
from keras.callbacks import TensorBoard
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#print(train_data.shape)
#print(train_data[0])

mean = train_data.mean(axis=0)
train_data -= mean
#print(train_data[0])
std = train_data.std(axis=0)
train_data /= std
#print(train_data[0])
test_data -= mean
test_data/= std

from keras import models
from keras import layers

def build_model():
    model = modelayers.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k) :
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: ]