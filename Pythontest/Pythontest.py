import tensorflow as tf

print("hello world")

mnist = tf.keras.datasets.mnist

#extract the image data into xtrain and y train and x test and y test
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 28x28 images of hand written digits 0-9

#normalising the values for the nn to train faster
x_train = tf.keras.utils.normalize(x_train, axis=1) # makes it between 0-1 easier to learn for network
x_test = tf.keras.utils.normalize(x_test, axis=1)

#start of the neural network
model = tf.keras.models.Sequential() # feed forward one of two
#input layer of neural network
#x_train = x_train.reshape((60000, 28 * 28))
#x_train = x_train.astype('float32') / 255
#x_test = x_test.reshape((10000, 28 * 28))
#x_test = x_test.astype('float32') / 255
#model.add(tf.keras.layers.Dense(512,activation = 'relu', input_shape=(28 * 28,))) #input and specifys next layers output
model.add(tf.keras.layers.Flatten()) # flattens a mutli array in to single array from 3d to 1d basically
#First hidden layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # tf.nn.relu = a kind of activation like sigmoid function
#second hidden layer 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#output layer of neural network
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # output layer using softmax for probability distribution instead of relu.  

#nn done set up 

#setting up how to train the neural network specifiyng optimiser and loss function
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy']) #lossfunction the degree of error minimise loss, optimizer is the backprogation type. keras supports alot, adam is one of them. for loss the most popular one is categorical crossentropy  

#time to train the model
model.fit(x_train,y_train, epochs = 3)
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)
#import matplotlib.pyplot as plt
#print(x_train[0])
#plt.imshow(x_train[0])
#plt.show()

#how to save and load a model/nn
#save
#model.save('NumberSaveModel')
#new_model = tf.keras.models.load_model('NumberSaveModel')

#prediciton
#predictions = new_model.predict([x_test])
#print(predictions)

#import numpy as np
#print(np.argmax(predictions[0]))
#plt.imshow(x_test[0])
#plt.show()
