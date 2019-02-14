from keras.callbacks import TensorBoard
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #Only the top 10000 words

#print(len(train_data[0]))
#print(train_data[0])
import numpy as np

#creates  a 10000 zero array, for every value that exists eg 5 then result[5] = 1. 
def vectorize_sequences(sequences, dimension=10000):  
   results = np.zeros((len(sequences), dimension))
   for i, sequence in enumerate(sequences):
       results[i, sequence] = 1
   return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

import time
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
#print(len(x_train[0]))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy', metrics = ['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

history = model.fit(x_train,y_train, epochs=3,batch_size=512,validation_data=(x_val, y_val),callbacks=[tensorboard])
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)

import matplotlib.pyplot as plt

history_dict = history.history
history_dict.keys()
[u'acc', u'loss', u'val_acc', u'val_loss']

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation  accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#print(x_train[0