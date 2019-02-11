import keras as kr
from keras.datasets import reuters
import numpy as np

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) #Only the top 10000 words
#finns vectorize_sequnces already exists in keras as to_categorical. 
# x_test = to_categorical(x_test)
def vectorize_sequences(sequences, dimension=10000):  
   results = np.zeros((len(sequences), dimension))
   for i, sequence in enumerate(sequences):
       results[i, sequence] = 1
   return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#y_train = vectorize_sequences(train_labels)
#y_test = vectorize_sequences(test_labels)

#x_train = kr.utils.to_categorical(train_data)
#x_test = kr.utils.to_categorical(test_data)
y_train = kr.utils.to_categorical(train_labels)
y_test = kr.utils.to_categorical(test_labels)


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,partial_y_train, epochs=20,batch_size=512,validation_data=(x_val, y_val))
