import pickle 
import tensorflow as tf
pickle_in = open("x.TotalTrainingData","rb")
x_train = pickle.load(pickle_in)
pickle_in = open("y.TotalTrainingData","rb")
y_train = pickle.load(pickle_in)

print("hello world")

#normalising the values for the nn to train faster
#x_train = tf.keras.utils.normalize(x_train, axis=1) # makes it between 0-1 easier to learn for network

#start of the neural network
model = tf.keras.models.Sequential() # feed forward one of two
model.add(tf.keras.layers.Flatten()) # flattens a mutli array in to single array from 3d to 1d basically
#First hidden layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # tf.nn.relu = a kind of activation like sigmoid function
#second hidden layer 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
#output layer of neural network
model.add(tf.keras.layers.Dense(1200, activation = tf.nn.softmax)) # output layer using softmax for probability distribution instead of relu.  

#nn done set up 

#setting up how to train the neural network specifiyng optimiser and loss function
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy']) #lossfunction the degree of error minimise loss, optimizer is the backprogation type. keras supports alot, adam is one of them. for loss the most popular one is categorical crossentropy  
x_train = x_train.reshape(60,20*20*3)
x_train = x_train.astype('float32') / 255
y_train = y_train.reshape(60,20*20*3)
y_train = y_train.astype('float32') / 255
#time to train the model
model.fit(x_train,y_train, epochs = 3)

